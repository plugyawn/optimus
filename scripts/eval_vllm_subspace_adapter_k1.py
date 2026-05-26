#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from optimus.modeling.qwen import load_qwen_lora_config, qwen_lora_shapes
from optimus.modeling.subspace_lora import (
    load_subspace_candidates,
    specs_to_jsonl,
    write_subspace_adapter_specs,
)
from optimus.serving.runtime import (
    import_vllm_lora_request,
    make_sampling_params,
    optional_vllm_kwargs,
    runtime_environment,
    write_json,
)
from optimus.serving.search import base_eval, mixed_eval
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


def _wanted(args: argparse.Namespace) -> set[str] | None:
    wanted_items = list(args.candidate_id or [])
    for item in args.candidate_id_file or []:
        wanted_items.extend(line.strip() for line in Path(item).read_text().splitlines() if line.strip())
    return set(wanted_items) if wanted_items else None


def _default_targets(source_summary: dict[str, Any], policy: str) -> list[str]:
    preset = str(source_summary.get("target_preset") or "")
    if policy == "fused-qkv-exact" and preset in {"qv", "attn-qkvo"}:
        return ["q_proj", "k_proj", "v_proj"]
    if preset == "qv":
        return ["q_proj", "v_proj"]
    if preset == "attn-qkvo":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if preset == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    if preset == "transformer-linears":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "v_proj"]


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


def _top_specs(specs: list[Any], candidate_rows: list[dict[str, Any]], top_k: int) -> list[Any]:
    if top_k <= 0:
        return []
    by_candidate = {spec.candidate: spec for spec in specs}
    ordered_rows = sorted(candidate_rows, key=lambda row: (float(row["exact_mean"]), str(row["candidate"])), reverse=True)
    selected = []
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
    parser = argparse.ArgumentParser(description="Replay activation-subspace candidates through native vLLM LoRA adapters.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-id-file", action="append")
    parser.add_argument("--adapter-policy", choices=["fused-qkv-exact", "target-split"], default="target-split")
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
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
    parser.add_argument("--exclude-run", action="append", default=[])
    parser.add_argument("--dtype")
    parser.add_argument("--adapter-dtype")
    parser.add_argument("--targets")
    parser.add_argument("--max-loras", type=int, default=16)
    parser.add_argument("--max-cpu-loras", type=int, default=512)
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
    state_summary = json.loads((source / "subspace_state_summary.json").read_text())
    state_payload = torch.load(source / "subspace_state.pt", map_location="cpu")
    model = args.model or source_summary.get("model") or state_summary.get("model_id_or_path") or "Qwen/Qwen3-4B"
    dtype = args.dtype or source_summary.get("dtype") or "bfloat16"
    adapter_dtype = args.adapter_dtype or dtype
    prompt_input = args.prompt_input or source_summary.get("prompt_input") or "text"
    prompt_variant = (args.prompt_variants or ",".join(source_summary.get("prompt_variants") or ["tight"])).split(",", 1)[0].strip() or "tight"
    max_new_tokens = int(args.max_new_tokens or source_summary.get("max_new_tokens") or 64)
    use_chat_template = bool(source_summary.get("use_chat_template", False)) if args.use_chat_template is None else bool(args.use_chat_template)
    stop_at_answer = bool(source_summary.get("stop_at_answer", True)) if args.stop_at_answer is None else bool(args.stop_at_answer)
    targets = [item.strip() for item in (args.targets or ",".join(_default_targets(source_summary, args.adapter_policy))).split(",") if item.strip()]

    config = load_qwen_lora_config(model, local_files_only=bool(args.local_files_only))
    qwen_lora_shapes(config, targets)
    candidates = load_subspace_candidates(source / "candidates.jsonl", _wanted(args))
    basis_rank = int(candidates[0].basis_rank)
    if any(int(candidate.basis_rank) != basis_rank for candidate in candidates):
        raise SystemExit("mixed basis ranks are not supported in one adapter replay")
    adapter_rank = int(args.adapter_rank or basis_rank)
    if adapter_rank <= 0 or adapter_rank > basis_rank:
        raise SystemExit(f"--adapter-rank must be in [1, {basis_rank}], got {adapter_rank}")
    if args.scale_multiplier <= 0:
        raise SystemExit("--scale-multiplier must be positive")

    adapter_started = time.perf_counter()
    specs = write_subspace_adapter_specs(
        out=out,
        model=model,
        config=config,
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidates=candidates,
        targets=targets,
        policy=args.adapter_policy,
        tensor_dtype=adapter_dtype,
        adapter_rank=adapter_rank,
        scale_multiplier=float(args.scale_multiplier),
    )
    adapter_build_s = time.perf_counter() - adapter_started
    (out / "adapters.jsonl").write_text(specs_to_jsonl(specs))

    exclude_ids = _source_example_ids(source) if args.exclude_source_splits else set()
    for exclude_run in args.exclude_run:
        exclude_ids.update(_source_example_ids(Path(exclude_run)))
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
        max_loras=int(args.max_loras),
        max_lora_rank=adapter_rank,
        max_cpu_loras=max(int(args.max_cpu_loras), len(specs)),
        enforce_eager=bool(args.enforce_eager),
        **llm_kwargs,
    )
    eval_args = SimpleNamespace(
        chunk_adapters=int(args.chunk_adapters or args.max_loras),
        max_loras=int(args.max_loras),
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
        "kind": "vllm_subspace_adapter_k1_final_replay",
        "source_run": str(source),
        "model": model,
        "data": args.data,
        "seed": args.seed,
        "prompts": len(examples),
        "excluded_source_example_ids": len(exclude_ids),
        "population": len(specs),
        "basis_rank": basis_rank,
        "adapter_rank": adapter_rank,
        "scale_multiplier": float(args.scale_multiplier),
        "targets": targets,
        "adapter_policy": args.adapter_policy,
        "dtype": dtype,
        "adapter_dtype": adapter_dtype,
        "max_loras": int(args.max_loras),
        "chunk_adapters": int(args.chunk_adapters or args.max_loras),
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
