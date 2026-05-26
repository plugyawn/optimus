#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from optimus.core.perturbations import PerturbationSpec
from optimus.modeling import AdapterSpec, save_seed_adapter
from optimus.modeling.qwen import load_qwen_lora_config, qwen_lora_shapes
from optimus.serving.runtime import (
    import_vllm_lora_request,
    make_sampling_params,
    optional_vllm_kwargs,
    runtime_environment,
    write_json,
)
from optimus.serving.search import base_eval, mixed_eval, safe_name
from optimus.subspace.reference import config_hash, git_commit, git_dirty, sha256_bytes
from optimus.tasks.countdown import load_examples


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_overwrite(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _source_example_ids(source: Path) -> set[int]:
    out: set[int] = set()
    for name in ("per_prompt.jsonl", "holdout_per_prompt.jsonl"):
        path = source / name
        if not path.exists():
            continue
        for row in _jsonl(path):
            if "example_id" in row:
                out.add(int(row["example_id"]))
    return out


def _candidate_from_adapter(row: dict[str, Any]) -> PerturbationSpec:
    return PerturbationSpec.from_record({"candidate": row["candidate"]}, default_method="lora")


def _load_candidates(source: Path, wanted: set[str] | None) -> list[PerturbationSpec]:
    candidates = []
    for row in _jsonl(source / "adapters.jsonl"):
        candidate = _candidate_from_adapter(row)
        if wanted is not None and candidate.key not in wanted and str(row.get("name")) not in wanted:
            continue
        candidates.append(candidate)
    if not candidates:
        raise SystemExit("no LoRA candidates selected from adapters.jsonl")
    return candidates


def _write_adapters(
    *,
    out: Path,
    model: str,
    rank: int,
    targets: list[str],
    adapter_dtype: str,
    candidates: list[PerturbationSpec],
    local_files_only: bool,
) -> list[AdapterSpec]:
    config = load_qwen_lora_config(model, local_files_only=local_files_only)
    qwen_lora_shapes(config, targets)
    adapter_root = out / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)
    specs = []
    for idx, candidate in enumerate(candidates):
        name = safe_name(candidate)
        path = adapter_root / f"{idx:05d}_{name}"
        save_seed_adapter(
            path,
            model=model,
            candidate=candidate,
            rank=rank,
            targets=targets,
            config=config,
            tensor_dtype=adapter_dtype,
        )
        specs.append(
            AdapterSpec(
                index=idx,
                name=name,
                lora_int_id=idx + 1,
                path=str(path.resolve()),
                candidate=candidate.key,
                seed=candidate.seed,
                sigma=candidate.sigma,
                sign=candidate.sign,
            )
        )
    return specs


def _candidate_score_record(row: dict[str, Any], *, base_exact: float, sample_count: int, stage: str) -> dict[str, Any]:
    return {
        "candidate_id": row["candidate"],
        "split": "final",
        "selection_stage": stage,
        "aggregate_metrics": {
            "exact": float(row["exact_mean"]),
            "delta_vs_base": float(row["exact_mean"]) - float(base_exact),
            "malformed": float(row["malformed_mean"]),
            "cap_hit": float(row["cap_hit_mean"]),
        },
        "sample_count": sample_count,
        "output_tokens": int(row["output_tokens"]),
        "elapsed_s": float(row["elapsed_s"]),
        "adapter_index": int(row["adapter_index"]),
        "adapter": row["adapter"],
        "seed": int(row["seed"]),
        "sigma": float(row["sigma"]),
        "sign": int(row["sign"]),
    }


def _top_specs(specs: list[AdapterSpec], candidate_rows: list[dict[str, Any]], top_k: int) -> list[AdapterSpec]:
    if top_k <= 0:
        return []
    by_candidate = {spec.candidate: spec for spec in specs}
    ordered_rows = sorted(candidate_rows, key=lambda row: (float(row["exact_mean"]), str(row["candidate"])), reverse=True)
    selected: list[AdapterSpec] = []
    seen = set()
    for row in ordered_rows:
        candidate = str(row["candidate"])
        if candidate in seen:
            continue
        selected.append(by_candidate[candidate])
        seen.add(candidate)
        if len(selected) >= top_k:
            break
    return selected


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay deterministic vLLM LoRA candidates on a fresh Countdown split.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-id-file", action="append")
    parser.add_argument("--confirm-top-k", type=int, default=0)
    parser.add_argument("--model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--prompt-input", choices=["text", "token_ids"])
    parser.add_argument("--prompt-variants")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--stop-at-answer", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exclude-source-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype")
    parser.add_argument("--adapter-dtype")
    parser.add_argument("--rank", type=int)
    parser.add_argument("--targets")
    parser.add_argument("--max-loras", type=int)
    parser.add_argument("--max-cpu-loras", type=int)
    parser.add_argument("--chunk-adapters", type=int)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--keep-adapters", action="store_true")
    parser.add_argument("--vllm-kwarg", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    source_summary = json.loads((source / "summary.json").read_text())

    wanted: set[str] | None = None
    wanted_items = list(args.candidate_id or [])
    for item in args.candidate_id_file or []:
        wanted_items.extend(line.strip() for line in Path(item).read_text().splitlines() if line.strip())
    if wanted_items:
        wanted = set(wanted_items)

    model = args.model or source_summary.get("model") or "Qwen/Qwen3-4B"
    rank = int(args.rank or source_summary.get("rank") or 8)
    targets = [item.strip() for item in (args.targets or ",".join(source_summary.get("targets") or [])).split(",") if item.strip()]
    if not targets:
        raise SystemExit("could not infer LoRA targets")
    dtype = args.dtype or source_summary.get("dtype") or "bfloat16"
    adapter_dtype = args.adapter_dtype or source_summary.get("adapter_dtype") or dtype
    prompt_input = args.prompt_input or source_summary.get("prompt_input") or "text"
    prompt_variants = args.prompt_variants or ",".join(source_summary.get("prompt_variants") or ["tight"])
    prompt_variant = prompt_variants.split(",", 1)[0].strip() or "tight"
    max_new_tokens = int(args.max_new_tokens or source_summary.get("max_new_tokens") or 64)
    use_chat_template = bool(source_summary.get("use_chat_template", False)) if args.use_chat_template is None else bool(args.use_chat_template)
    stop_at_answer = bool(source_summary.get("stop_at_answer", False)) if args.stop_at_answer is None else bool(args.stop_at_answer)
    max_loras = int(args.max_loras or source_summary.get("max_loras") or 32)
    max_cpu_loras = int(args.max_cpu_loras or source_summary.get("max_cpu_loras") or 2048)
    chunk_adapters = int(args.chunk_adapters or source_summary.get("chunk_adapters") or max_loras)

    candidates = _load_candidates(source, wanted)
    adapter_started = time.perf_counter()
    specs = _write_adapters(
        out=out,
        model=model,
        rank=rank,
        targets=targets,
        adapter_dtype=adapter_dtype,
        candidates=candidates,
        local_files_only=bool(args.local_files_only),
    )
    adapter_build_s = time.perf_counter() - adapter_started
    _write_jsonl_overwrite(out / "adapters.jsonl", [asdict(spec) for spec in specs])

    exclude_ids = _source_example_ids(source) if args.exclude_source_splits else set()
    examples = load_examples(args.data, args.prompts, args.seed, exclude_ids=exclude_ids)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    LLM, SamplingParams, LoRARequest = import_vllm_lora_request()
    sampling = make_sampling_params(SamplingParams, max_new_tokens, stop_at_answer)
    vllm_args = SimpleNamespace(
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        enable_chunked_prefill=None,
        kv_cache_dtype="",
        vllm_kwarg=args.vllm_kwarg or [],
    )
    llm_kwargs = optional_vllm_kwargs(vllm_args)
    if args.max_model_len:
        llm_kwargs.setdefault("max_model_len", args.max_model_len)
    llm_kwargs.setdefault("enable_prefix_caching", False)
    llm = LLM(
        model=model,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_loras=max_loras,
        max_lora_rank=rank,
        max_cpu_loras=max(max_cpu_loras, len(specs)),
        enforce_eager=bool(args.enforce_eager),
        **llm_kwargs,
    )
    eval_args = SimpleNamespace(
        chunk_adapters=chunk_adapters,
        max_loras=max_loras,
        use_chat_template=use_chat_template,
        prompt_input=prompt_input,
        max_new_tokens=max_new_tokens,
    )

    base_rows, base_metrics = base_eval(llm, sampling, examples, eval_args, mode="base_final", prompt_variant=prompt_variant)
    started = time.perf_counter()
    rows, candidate_rows, aggregate = mixed_eval(
        llm,
        LoRARequest,
        sampling,
        examples,
        specs,
        eval_args,
        mode="final",
        prompt_variant=prompt_variant,
    )
    elapsed = time.perf_counter() - started
    candidate_scores = {row["candidate"]: float(row["exact_mean"]) for row in candidate_rows}
    best = max(candidate_rows, key=lambda row: (float(row["exact_mean"]), str(row["candidate"])))
    confirmed_rows: list[dict[str, Any]] = []
    confirmed_candidate_rows: list[dict[str, Any]] = []
    confirmed_aggregate: dict[str, Any] | None = None
    confirmed_elapsed = 0.0
    confirm_specs = _top_specs(specs, candidate_rows, int(args.confirm_top_k))
    if confirm_specs:
        confirm_eval_args = SimpleNamespace(
            chunk_adapters=1,
            max_loras=1,
            use_chat_template=use_chat_template,
            prompt_input=prompt_input,
            max_new_tokens=max_new_tokens,
        )
        confirm_started = time.perf_counter()
        confirmed_rows, confirmed_candidate_rows, confirmed_aggregate = mixed_eval(
            llm,
            LoRARequest,
            sampling,
            examples,
            confirm_specs,
            confirm_eval_args,
            mode="final_confirmed",
            prompt_variant=prompt_variant,
        )
        confirmed_elapsed = time.perf_counter() - confirm_started
    confirmed_candidate_scores = {row["candidate"]: float(row["exact_mean"]) for row in confirmed_candidate_rows}
    confirmed_best = (
        max(confirmed_candidate_rows, key=lambda row: (float(row["exact_mean"]), str(row["candidate"])))
        if confirmed_candidate_rows
        else None
    )
    score_rows = [
        {
            "candidate_id": "base",
            "split": "final",
            "selection_stage": "base_final",
            "aggregate_metrics": {"exact": float(base_metrics["exact_mean"])},
            "sample_count": len(examples),
            "output_tokens": int(base_metrics["output_tokens"]),
            "elapsed_s": float(base_metrics["elapsed_s"]),
        }
    ]
    for row in candidate_rows:
        score_rows.append(_candidate_score_record(row, base_exact=float(base_metrics["exact_mean"]), sample_count=len(examples), stage="k1_final_replay"))
    confirmed_score_rows = [
        _candidate_score_record(row, base_exact=float(base_metrics["exact_mean"]), sample_count=len(examples), stage="k1_final_confirmed_chunk1")
        for row in confirmed_candidate_rows
    ]
    scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
    confirmed_scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in confirmed_score_rows)
    _write_jsonl_overwrite(
        out / "per_prompt.jsonl",
        [dict(row, split="final", candidate_id="base") for row in base_rows]
        + [dict(row, split="final", candidate_id=row["candidate"]) for row in rows],
    )
    (out / "candidate_scores.jsonl").write_text(scores_text)
    if confirm_specs:
        _write_jsonl_overwrite(
            out / "confirmed_per_prompt.jsonl",
            [dict(row, split="final_confirmed", candidate_id=row["candidate"]) for row in confirmed_rows],
        )
        (out / "confirmed_candidate_scores.jsonl").write_text(confirmed_scores_text)
    summary = {
        "kind": "vllm_lora_k1_final_replay",
        "source_run": str(source),
        "model": model,
        "data": args.data,
        "seed": args.seed,
        "prompts": len(examples),
        "excluded_source_example_ids": len(exclude_ids),
        "population": len(specs),
        "rank": rank,
        "targets": targets,
        "dtype": dtype,
        "adapter_dtype": adapter_dtype,
        "max_loras": max_loras,
        "chunk_adapters": chunk_adapters,
        "base_final_score": float(base_metrics["exact_mean"]),
        "candidate_final_scores": candidate_scores,
        "best_candidate_final_score": float(best["exact_mean"]),
        "best_candidate_id": str(best["candidate"]),
        "candidate_replay_sec": elapsed / max(len(specs), 1),
        "mixed_candidate_sec": len(specs) / max(elapsed, 1e-9),
        "aggregate": aggregate,
        "confirm_top_k": int(args.confirm_top_k),
        "confirmed_population": len(confirm_specs),
        "confirmed_candidate_final_scores": confirmed_candidate_scores,
        "confirmed_best_candidate_final_score": float(confirmed_best["exact_mean"]) if confirmed_best is not None else None,
        "confirmed_best_candidate_id": str(confirmed_best["candidate"]) if confirmed_best is not None else None,
        "confirmed_candidate_replay_sec": confirmed_elapsed / max(len(confirm_specs), 1) if confirm_specs else None,
        "confirmed_mixed_candidate_sec": len(confirm_specs) / max(confirmed_elapsed, 1e-9) if confirm_specs else None,
        "confirmed_aggregate": confirmed_aggregate,
        "adapter_build_s": adapter_build_s,
        "candidate_scores_hash": sha256_bytes(scores_text.encode("utf-8")),
        "confirmed_candidate_scores_hash": sha256_bytes(confirmed_scores_text.encode("utf-8")) if confirm_specs else None,
        "runtime_environment": runtime_environment(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "prompt_variant": prompt_variant,
        "decode_config_hash": config_hash({"max_new_tokens": max_new_tokens, "stop_at_answer": stop_at_answer}),
    }
    write_json(out / "summary.json", summary)
    if not args.keep_adapters:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
