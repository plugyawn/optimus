#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from optimus.backends.vllm_lazy_hook import (
    _score_outputs,
    discover_targets,
    find_vllm_model,
    find_vllm_model_runner,
    install_hooks,
    install_model_runner_routing,
    remove_hooks,
    remove_model_runner_routing,
)
from optimus.backends.vllm_lora_hook import FusedQKVSpec, LazyLoraHookRuntime
from optimus.core.perturbations import PerturbationSpec
from optimus.modeling.qwen import load_qwen_lora_config
from optimus.serving.prompting import make_vllm_prompt_inputs
from optimus.serving.runtime import (
    configure_vllm_logging,
    make_sampling_params,
    optional_vllm_kwargs,
    runtime_environment,
    write_json,
)
from optimus.subspace.reference import config_hash, git_commit, git_dirty, sha256_bytes
from optimus.tasks.countdown import load_examples
from optimus.tasks.prompt_variants import make_variant_prompts


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


def _target_preset_for(targets: list[str]) -> str:
    target_set = set(targets)
    if target_set <= {"q_proj", "v_proj"}:
        return "qv"
    if target_set <= {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return "attn-qkvo"
    if target_set <= {"gate_proj", "up_proj", "down_proj"}:
        return "mlp"
    return "transformer-linears"


def _fused_qkv_spec(model: str, *, local_files_only: bool) -> FusedQKVSpec:
    config = load_qwen_lora_config(model, local_files_only=local_files_only)
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    return FusedQKVSpec(q_out=heads * head_dim, kv_out=kv_heads * head_dim)


def _candidate_id(candidate: PerturbationSpec) -> str:
    return candidate.key


@contextmanager
def _candidate_batch_context(
    runtime: LazyLoraHookRuntime,
    llm: Any,
    candidates: list[PerturbationSpec],
    prompt_count: int,
    *,
    routing: str,
):
    if not candidates:
        runtime.set_candidate(None)
        yield
        return
    if routing == "input-order":
        runtime.set_candidate_batch_by_order(candidates, prompt_count=prompt_count)
        try:
            yield
        finally:
            runtime.set_candidate(None)
        return
    start_request_id = int(getattr(llm.request_counter, "counter"))
    request_candidate_by_id: dict[str, PerturbationSpec] = {}
    for candidate_index, candidate in enumerate(candidates):
        for prompt_index in range(prompt_count):
            request_candidate_by_id[str(start_request_id + candidate_index * prompt_count + prompt_index)] = candidate
    runtime.set_candidate_batch(request_candidate_by_id)
    try:
        yield
    finally:
        runtime.set_candidate(None)


def _split_candidate_outputs(outputs: list[Any], candidates: list[PerturbationSpec], prompt_count: int) -> dict[str, list[Any]]:
    if len(outputs) != prompt_count * len(candidates):
        raise RuntimeError(f"candidate-batched vLLM output count mismatch: expected {prompt_count * len(candidates)}, got {len(outputs)}")
    split: dict[str, list[Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        start = candidate_index * prompt_count
        split[_candidate_id(candidate)] = outputs[start : start + prompt_count]
    return split


def _chunked(items: list[PerturbationSpec], size: int) -> list[list[PerturbationSpec]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay deterministic LoRA candidates through adapter-free vLLM forward hooks.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-id-file", action="append")
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
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument("--candidate-routing", choices=["request-id", "input-order"], default="request-id")
    parser.add_argument("--preload-factors", action="store_true")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--local-files-only", action="store_true")
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
    candidates = _load_candidates(source, wanted)

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
    candidate_batch_size = max(1, int(args.candidate_batch_size or 1))

    exclude_ids = _source_example_ids(source) if args.exclude_source_splits else set()
    examples = load_examples(args.data, args.prompts, args.seed, exclude_ids=exclude_ids)

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/vllm-config")
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    configure_vllm_logging()
    from vllm import LLM, SamplingParams

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
    llm_kwargs.setdefault("tensor_parallel_size", int(args.tensor_parallel_size or 1))
    llm_kwargs.setdefault("enable_prefix_caching", False)
    llm_kwargs.setdefault("enforce_eager", bool(args.enforce_eager))
    llm_kwargs.setdefault("trust_remote_code", True)
    llm_kwargs.setdefault("gpu_memory_utilization", float(args.gpu_memory_utilization))
    llm = LLM(model=model, dtype=dtype, **llm_kwargs)
    _, torch_model = find_vllm_model(llm)
    _, model_runner = find_vllm_model_runner(llm)
    target_set = set(targets)
    hook_targets = [
        target
        for target in discover_targets(torch_model, preset=_target_preset_for(targets), layers=None)
        if target.suffix in target_set or (target.suffix == "qkv_proj" and bool(target_set & {"q_proj", "k_proj", "v_proj"}))
    ]
    if len(hook_targets) == 0:
        raise RuntimeError(f"no hook targets found for {targets}")
    runtime = LazyLoraHookRuntime(
        hook_targets,
        rank=rank,
        adapter_dtype=adapter_dtype,
        fused_qkv=_fused_qkv_spec(model, local_files_only=bool(args.local_files_only)),
        preserve_factor_cache=bool(args.preload_factors),
    )
    handles = install_hooks(runtime)
    routing_handle = install_model_runner_routing(runtime, model_runner)
    try:
        tokenizer = llm.get_tokenizer()
        texts = make_variant_prompts(examples, prompt_variant, tokenizer=tokenizer, use_chat_template=use_chat_template)
        prompt_inputs = make_vllm_prompt_inputs(texts, tokenizer, prompt_input)
        sampling = make_sampling_params(SamplingParams, max_new_tokens, stop_at_answer)
        factor_preload_s = 0.0
        if args.preload_factors:
            preload_started = time.perf_counter()
            for candidate in candidates:
                runtime.preload_candidate(candidate)
            factor_preload_s = time.perf_counter() - preload_started

        runtime.set_candidate(None)
        base_started = time.perf_counter()
        base_outputs = llm.generate(prompt_inputs, sampling, use_tqdm=False)
        base_elapsed = time.perf_counter() - base_started
        base_score, base_tokens, base_prompt_rows = _score_outputs(examples, base_outputs, max_new_tokens=max_new_tokens)
        per_prompt_rows = [{"split": "final", "candidate_id": "base", **row} for row in base_prompt_rows]
        score_rows = [
            {
                "candidate_id": "base",
                "split": "final",
                "selection_stage": "base_final",
                "aggregate_metrics": {"exact": base_score},
                "elapsed_s": base_elapsed,
                "output_tokens": base_tokens,
                "sample_count": len(examples),
            }
        ]

        total_tokens = base_tokens
        total_qx = 0.0
        total_delta = 0.0
        total_delta_rows = 0
        total_delta_calls = 0
        started_all = time.perf_counter()
        for candidate_chunk in _chunked(candidates, candidate_batch_size):
            runtime.reset_timing()
            chunk_started = time.perf_counter()
            if len(candidate_chunk) == 1:
                runtime.set_candidate(candidate_chunk[0])
                outputs = llm.generate(prompt_inputs, sampling, use_tqdm=False)
                outputs_by_candidate = {_candidate_id(candidate_chunk[0]): outputs}
            else:
                batched_inputs = []
                for _candidate in candidate_chunk:
                    batched_inputs.extend(prompt_inputs)
                with _candidate_batch_context(runtime, llm, candidate_chunk, len(prompt_inputs), routing=args.candidate_routing):
                    outputs = llm.generate(batched_inputs, sampling, use_tqdm=False)
                outputs_by_candidate = _split_candidate_outputs(outputs, candidate_chunk, len(prompt_inputs))
            chunk_elapsed = time.perf_counter() - chunk_started
            if runtime.delta_rows <= 0:
                raise RuntimeError("vLLM lazy LoRA hook did not apply any perturbation rows")
            total_qx += runtime.qx_time_s
            total_delta += runtime.delta_time_s
            total_delta_rows += runtime.delta_rows
            total_delta_calls += runtime.delta_calls
            for candidate in candidate_chunk:
                candidate_id = _candidate_id(candidate)
                candidate_outputs = outputs_by_candidate[candidate_id]
                score, tokens, rows = _score_outputs(examples, candidate_outputs, max_new_tokens=max_new_tokens)
                total_tokens += tokens
                per_prompt_rows.extend({"split": "final", "candidate_id": candidate_id, **row} for row in rows)
                score_rows.append(
                    {
                        "candidate_id": candidate_id,
                        "split": "final",
                        "selection_stage": "k1_final_replay",
                        "aggregate_metrics": {"exact": score, "delta_vs_base": score - base_score},
                        "output_tokens": tokens,
                        "sample_count": len(examples),
                        "elapsed_s": chunk_elapsed / max(len(candidate_chunk), 1),
                        "seed": candidate.seed,
                        "sigma": candidate.sigma,
                        "sign": candidate.sign,
                    }
                )
        elapsed_all = time.perf_counter() - started_all
        candidate_rows = [row for row in score_rows if row["candidate_id"] != "base"]
        best = max(candidate_rows, key=lambda row: (float(row["aggregate_metrics"]["exact"]), str(row["candidate_id"])))
        scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
        summary = {
            "kind": "vllm_lora_hook_k1_final_replay",
            "source_run": str(source),
            "model": model,
            "data": args.data,
            "seed": args.seed,
            "prompts": len(examples),
            "excluded_source_example_ids": len(exclude_ids),
            "population": len(candidates),
            "rank": rank,
            "targets": targets,
            "dtype": dtype,
            "adapter_dtype": adapter_dtype,
            "candidate_batch_size": candidate_batch_size,
            "candidate_routing": args.candidate_routing,
            "enforce_eager": bool(args.enforce_eager),
            "base_final_score": base_score,
            "candidate_final_scores": {
                row["candidate_id"]: row["aggregate_metrics"]["exact"]
                for row in candidate_rows
            },
            "best_candidate_final_score": float(best["aggregate_metrics"]["exact"]),
            "best_candidate_id": str(best["candidate_id"]),
            "candidate_replay_sec": elapsed_all / max(len(candidates), 1),
            "mixed_candidate_sec": len(candidates) / max(elapsed_all, 1e-9),
            "candidate_scores_hash": sha256_bytes(scores_text.encode("utf-8")),
            "runtime_environment": runtime_environment(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "prompt_variant": prompt_variant,
            "decode_config_hash": config_hash({"max_new_tokens": max_new_tokens, "stop_at_answer": stop_at_answer}),
            "output_tokens": total_tokens,
            "factor_preload_s": factor_preload_s,
            "hook_timing": {
                "qx_time_s": total_qx,
                "lazy_delta_time_s": total_delta,
                "delta_rows": total_delta_rows,
                "delta_calls": total_delta_calls,
            },
        }
        write_json(out / "summary.json", summary)
        (out / "candidate_scores.jsonl").write_text(scores_text)
        _write_jsonl_overwrite(out / "per_prompt.jsonl", per_prompt_rows)
        _write_jsonl_overwrite(out / "candidates.jsonl", [candidate.to_record() for candidate in candidates])
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        runtime.set_candidate(None)
        remove_model_runner_routing(routing_handle)
        remove_hooks(handles)


if __name__ == "__main__":
    raise SystemExit(main())
