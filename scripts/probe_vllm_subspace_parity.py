#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoTokenizer

from optimus.backends.vllm_lazy_hook import (
    LazyHookRuntime,
    discover_targets,
    find_vllm_model,
    find_vllm_model_runner,
    install_hooks,
    install_model_runner_routing,
    remove_hooks,
    remove_model_runner_routing,
)
from optimus.modeling.qwen import load_qwen_lora_config, qwen_lora_shapes
from optimus.modeling.subspace_lora import (
    load_subspace_candidates,
    specs_to_jsonl,
    write_subspace_adapter_specs,
)
from optimus.serving.prompting import make_vllm_prompt_inputs
from optimus.serving.runtime import (
    configure_vllm_logging,
    import_vllm_lora_request,
    optional_vllm_kwargs,
    runtime_environment,
    write_json,
)
from optimus.subspace import SubspaceCandidate
from optimus.subspace.reference import config_hash, git_commit, git_dirty, sha256_bytes
from optimus.tasks.countdown import load_examples
from optimus.tasks.prompt_variants import make_variant_prompts

from scripts.eval_vllm_lazy_k1 import (
    _exclude_ids_from_source,
    _filter_targets,
    _load_basis,
    _load_betas,
    _parse_targets,
    _qkv_dims_from_config,
    _single_radius,
)
from scripts.eval_vllm_subspace_adapter_k1 import _default_targets as _default_adapter_targets


def _main_tensor(value: Any) -> torch.Tensor | None:
    main = value[0] if isinstance(value, tuple) and value and torch.is_tensor(value[0]) else value
    return main if torch.is_tensor(main) else None


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _wanted(args: argparse.Namespace) -> set[str] | None:
    wanted_items = list(args.candidate_id or [])
    for item in args.candidate_id_file or []:
        wanted_items.extend(line.strip() for line in Path(item).read_text().splitlines() if line.strip())
    return set(wanted_items) if wanted_items else None


def _logprob_value(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, dict):
        return float(value["logprob"])
    return float(getattr(value, "logprob"))


def _token_text(value: Any) -> str | None:
    if isinstance(value, dict):
        token = value.get("decoded_token", value.get("token"))
        return None if token is None else str(token)
    token = getattr(value, "decoded_token", None)
    if token is None:
        token = getattr(value, "token", None)
    return None if token is None else str(token)


def _top_logprobs(entry: Any) -> dict[str, dict[str, Any]]:
    if entry is None:
        return {}
    out: dict[str, dict[str, Any]] = {}
    for token_id, value in dict(entry).items():
        out[str(int(token_id))] = {
            "logprob": _logprob_value(value),
            "token": _token_text(value),
        }
    return out


def _first_generated_signature(output: Any) -> dict[str, Any]:
    completions = list(getattr(output, "outputs", []) or [])
    if not completions:
        return {"token_id": None, "logprob": None, "top_logprobs": {}}
    completion = completions[0]
    token_ids = list(getattr(completion, "token_ids", []) or [])
    logprobs = list(getattr(completion, "logprobs", []) or [])
    token_id = int(token_ids[0]) if token_ids else None
    top = _top_logprobs(logprobs[0] if logprobs else None)
    selected = None
    if token_id is not None and str(token_id) in top:
        selected = top[str(token_id)]["logprob"]
    return {"token_id": token_id, "logprob": selected, "top_logprobs": top}


def _prompt_tail_signature(output: Any, *, tail_tokens: int) -> list[dict[str, Any]]:
    prompt_ids = list(getattr(output, "prompt_token_ids", []) or [])
    prompt_logprobs = list(getattr(output, "prompt_logprobs", []) or [])
    rows: list[dict[str, Any]] = []
    if not prompt_ids or not prompt_logprobs:
        return rows
    start = max(0, len(prompt_ids) - int(tail_tokens))
    for pos in range(start, len(prompt_ids)):
        entry = prompt_logprobs[pos] if pos < len(prompt_logprobs) else None
        token_id = int(prompt_ids[pos])
        top = _top_logprobs(entry)
        rows.append(
            {
                "position": pos,
                "token_id": token_id,
                "logprob": top.get(str(token_id), {}).get("logprob"),
                "top_logprobs": top,
            }
        )
    return rows


def _signature_rows(
    outputs: list[Any],
    *,
    candidate_id: str,
    examples: list[Any],
    backend: str,
    prompt_tail_tokens: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for idx, output in enumerate(outputs):
        rows.append(
            {
                "backend": backend,
                "candidate_id": candidate_id,
                "prompt_index": idx,
                "example_id": int(examples[idx].id),
                "generated": _first_generated_signature(output),
                "prompt_tail": _prompt_tail_signature(output, tail_tokens=prompt_tail_tokens),
            }
        )
    return rows


def _max_common_logprob_diff(left: dict[str, Any], right: dict[str, Any]) -> tuple[float | None, int]:
    diffs: list[float] = []
    for section in ("generated",):
        left_top = left.get(section, {}).get("top_logprobs", {}) or {}
        right_top = right.get(section, {}).get("top_logprobs", {}) or {}
        for token_id in set(left_top) & set(right_top):
            diffs.append(abs(float(left_top[token_id]["logprob"]) - float(right_top[token_id]["logprob"])))
    for left_tail, right_tail in zip(left.get("prompt_tail", []) or [], right.get("prompt_tail", []) or []):
        left_top = left_tail.get("top_logprobs", {}) or {}
        right_top = right_tail.get("top_logprobs", {}) or {}
        for token_id in set(left_top) & set(right_top):
            diffs.append(abs(float(left_top[token_id]["logprob"]) - float(right_top[token_id]["logprob"])))
    return (max(diffs), len(diffs)) if diffs else (None, 0)


def _compare_rows(adapter_rows: list[dict[str, Any]], lazy_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (row["candidate_id"], int(row["prompt_index"])): row
        for row in adapter_rows
    }
    comparisons: list[dict[str, Any]] = []
    for lazy in lazy_rows:
        key = (lazy["candidate_id"], int(lazy["prompt_index"]))
        adapter = by_key[key]
        adapter_generated = adapter["generated"]
        lazy_generated = lazy["generated"]
        max_common, common_count = _max_common_logprob_diff(adapter, lazy)
        comparisons.append(
            {
                "candidate_id": key[0],
                "prompt_index": key[1],
                "example_id": lazy["example_id"],
                "generated_token_match": adapter_generated.get("token_id") == lazy_generated.get("token_id"),
                "adapter_generated_token_id": adapter_generated.get("token_id"),
                "lazy_generated_token_id": lazy_generated.get("token_id"),
                "adapter_generated_logprob": adapter_generated.get("logprob"),
                "lazy_generated_logprob": lazy_generated.get("logprob"),
                "generated_logprob_abs_diff": (
                    None
                    if adapter_generated.get("logprob") is None or lazy_generated.get("logprob") is None
                    else abs(float(adapter_generated["logprob"]) - float(lazy_generated["logprob"]))
                ),
                "max_common_top_logprob_abs_diff": max_common,
                "common_logprob_count": common_count,
            }
        )
    generated_matches = [bool(row["generated_token_match"]) for row in comparisons]
    generated_diffs = [
        float(row["generated_logprob_abs_diff"])
        for row in comparisons
        if row.get("generated_logprob_abs_diff") is not None
    ]
    top_diffs = [
        float(row["max_common_top_logprob_abs_diff"])
        for row in comparisons
        if row.get("max_common_top_logprob_abs_diff") is not None
    ]
    summary = {
        "comparisons": len(comparisons),
        "generated_token_match_count": sum(generated_matches),
        "generated_token_match_rate": sum(generated_matches) / max(len(generated_matches), 1),
        "max_generated_logprob_abs_diff": max(generated_diffs) if generated_diffs else None,
        "max_common_top_logprob_abs_diff": max(top_diffs) if top_diffs else None,
        "common_logprob_count": sum(int(row["common_logprob_count"]) for row in comparisons),
    }
    return comparisons, summary


def _make_sampling(SamplingParams: Any, *, max_logprobs: int, prompt_logprobs: int) -> Any:
    kwargs = {
        "max_tokens": 1,
        "temperature": 0.0,
        "logprobs": int(max_logprobs),
        "prompt_logprobs": int(prompt_logprobs),
    }
    return SamplingParams(**kwargs)


def _llm_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    vllm_args = SimpleNamespace(
        tensor_parallel_size=args.tensor_parallel_size,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enable_prefix_caching=False,
        enable_chunked_prefill=None,
        kv_cache_dtype="",
        vllm_kwarg=args.vllm_kwarg or [],
    )
    kwargs = optional_vllm_kwargs(vllm_args)
    if args.max_model_len:
        kwargs.setdefault("max_model_len", int(args.max_model_len))
    kwargs.setdefault("enable_prefix_caching", False)
    return kwargs


def _parse_capture_layers(text: str | None) -> set[int] | None:
    if text is None or text.strip().lower() in {"", "all"}:
        return None
    out = {int(item.strip()) for item in text.split(",") if item.strip()}
    if not out:
        return None
    return out


class _TargetOutputCapture:
    def __init__(
        self,
        *,
        backend: str,
        max_rows: int,
        max_calls_per_target: int,
    ) -> None:
        self.backend = backend
        self.max_rows = max(1, int(max_rows))
        self.max_calls_per_target = max(1, int(max_calls_per_target))
        self.current_candidate_id: str | None = None
        self.rows: list[dict[str, Any]] = []
        self._call_counts: dict[tuple[str, str], int] = {}

    def hook_for(self, target: Any) -> Any:
        def hook(_module: Any, _args: tuple[Any, ...], output: Any, *, __target: Any = target) -> None:
            candidate_id = self.current_candidate_id
            if candidate_id is None:
                return
            key = (candidate_id, __target.target_id)
            call_index = self._call_counts.get(key, 0)
            if call_index >= self.max_calls_per_target:
                return
            main = _main_tensor(output)
            if main is None:
                return
            flat = main.detach().reshape(-1, main.shape[-1])
            sample = flat[: self.max_rows].detach().float().cpu().contiguous()
            self.rows.append(
                {
                    "backend": self.backend,
                    "candidate_id": candidate_id,
                    "target_id": str(__target.target_id),
                    "module_name": str(__target.module_name),
                    "layer_index": int(__target.layer_index),
                    "suffix": str(__target.suffix),
                    "call_index": int(call_index),
                    "original_shape": tuple(int(dim) for dim in main.shape),
                    "captured_rows": int(sample.shape[0]),
                    "output_dim": int(sample.shape[-1]),
                    "tensor": sample,
                }
            )
            self._call_counts[key] = call_index + 1

        return hook


def _install_target_output_capture(
    args: argparse.Namespace,
    targets: list[Any],
    *,
    backend: str,
) -> tuple[_TargetOutputCapture | None, list[Any]]:
    if not bool(args.capture_target_outputs):
        return None, []
    wanted_layers = _parse_capture_layers(args.capture_layers)
    selected = [target for target in targets if wanted_layers is None or int(target.layer_index) in wanted_layers]
    capture = _TargetOutputCapture(
        backend=backend,
        max_rows=int(args.capture_max_rows),
        max_calls_per_target=int(args.capture_max_calls_per_target),
    )
    handles = [target.module.register_forward_hook(capture.hook_for(target)) for target in selected]
    return capture, handles


def _remove_capture_hooks(handles: list[Any]) -> None:
    for handle in handles:
        handle.remove()


def _capture_path(out: Path, backend: str) -> Path:
    return out / f"{backend}_target_outputs.pt"


def _save_capture(out: Path, backend: str, capture: _TargetOutputCapture | None) -> None:
    if capture is None:
        return
    torch.save(capture.rows, _capture_path(out, backend))


def _tensor_diff_stats(left: torch.Tensor, right: torch.Tensor) -> dict[str, Any]:
    left_f = left.detach().float()
    right_f = right.detach().float()
    if tuple(left_f.shape) != tuple(right_f.shape):
        return {
            "shape_match": False,
            "left_shape": tuple(int(dim) for dim in left_f.shape),
            "right_shape": tuple(int(dim) for dim in right_f.shape),
            "max_abs": None,
            "mean_abs": None,
            "rms_abs": None,
            "relative_rms": None,
        }
    diff = left_f - right_f
    abs_diff = diff.abs()
    rms = torch.sqrt(torch.mean(diff * diff)) if diff.numel() else torch.tensor(0.0)
    denom = torch.sqrt(torch.mean(left_f * left_f) + torch.mean(right_f * right_f)) if diff.numel() else torch.tensor(0.0)
    relative = float((rms / denom).item()) if float(denom.item()) > 0 else 0.0
    return {
        "shape_match": True,
        "left_shape": tuple(int(dim) for dim in left_f.shape),
        "right_shape": tuple(int(dim) for dim in right_f.shape),
        "max_abs": float(abs_diff.max().item()) if abs_diff.numel() else 0.0,
        "mean_abs": float(abs_diff.mean().item()) if abs_diff.numel() else 0.0,
        "rms_abs": float(rms.item()),
        "relative_rms": relative,
    }


def _compare_capture_rows(adapter_rows: list[dict[str, Any]], lazy_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_key = {
        (str(row["candidate_id"]), str(row["target_id"]), int(row["call_index"])): row
        for row in adapter_rows
    }
    comparisons: list[dict[str, Any]] = []
    missing_adapter = 0
    for lazy in lazy_rows:
        key = (str(lazy["candidate_id"]), str(lazy["target_id"]), int(lazy["call_index"]))
        adapter = by_key.get(key)
        if adapter is None:
            missing_adapter += 1
            continue
        stats = _tensor_diff_stats(adapter["tensor"], lazy["tensor"])
        comparisons.append(
            {
                "candidate_id": key[0],
                "target_id": key[1],
                "call_index": key[2],
                "layer_index": int(lazy.get("layer_index", adapter.get("layer_index", -1))),
                "suffix": str(lazy.get("suffix", adapter.get("suffix", ""))),
                "adapter_shape": list(stats["left_shape"]),
                "lazy_shape": list(stats["right_shape"]),
                "shape_match": bool(stats["shape_match"]),
                "max_abs": stats["max_abs"],
                "mean_abs": stats["mean_abs"],
                "rms_abs": stats["rms_abs"],
                "relative_rms": stats["relative_rms"],
            }
        )
    numeric = [row for row in comparisons if row.get("shape_match")]
    max_abs_values = [float(row["max_abs"]) for row in numeric if row.get("max_abs") is not None]
    mean_abs_values = [float(row["mean_abs"]) for row in numeric if row.get("mean_abs") is not None]
    rms_values = [float(row["rms_abs"]) for row in numeric if row.get("rms_abs") is not None]
    worst = max(numeric, key=lambda row: float(row.get("max_abs") or 0.0), default=None)
    summary = {
        "comparisons": len(comparisons),
        "shape_match_count": sum(1 for row in comparisons if row.get("shape_match")),
        "missing_adapter_count": missing_adapter,
        "missing_lazy_count": max(0, len(adapter_rows) - len(comparisons)),
        "max_abs": max(max_abs_values) if max_abs_values else None,
        "mean_abs_mean": sum(mean_abs_values) / max(len(mean_abs_values), 1) if mean_abs_values else None,
        "rms_abs_mean": sum(rms_values) / max(len(rms_values), 1) if rms_values else None,
        "worst": None if worst is None else {key: worst[key] for key in ("candidate_id", "target_id", "call_index", "layer_index", "max_abs", "mean_abs", "rms_abs", "relative_rms")},
    }
    return comparisons, summary


def _write_capture_comparisons(out: Path, comparisons: list[dict[str, Any]]) -> None:
    _write_jsonl(out / "target_output_drift.jsonl", comparisons)
    csv_path = out / "target_output_drift.csv"
    if not comparisons:
        csv_path.write_text("")
        return
    fieldnames = list(comparisons[0].keys())
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(comparisons)


def _run_adapter_signatures(
    args: argparse.Namespace,
    *,
    source: Path,
    out: Path,
    source_summary: dict[str, Any],
    state_summary: dict[str, Any],
    state_payload: dict[str, Any],
    candidates: list[SubspaceCandidate],
    examples: list[Any],
    prompt_inputs: list[Any],
    targets: list[str],
    adapter_rank: int,
    dtype: str,
    adapter_dtype: str,
) -> tuple[list[dict[str, Any]], float]:
    config = load_qwen_lora_config(args.model, local_files_only=bool(args.local_files_only))
    qwen_lora_shapes(config, targets)
    started = time.perf_counter()
    specs = write_subspace_adapter_specs(
        out=out,
        model=args.model,
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
    (out / "adapters.jsonl").write_text(specs_to_jsonl(specs))
    adapter_build_s = time.perf_counter() - started

    LLM, SamplingParams, LoRARequest = import_vllm_lora_request()
    sampling = _make_sampling(SamplingParams, max_logprobs=args.logprobs, prompt_logprobs=args.prompt_logprobs)
    llm = LLM(
        model=args.model,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=adapter_rank,
        max_cpu_loras=max(16, len(specs)),
        enforce_eager=bool(args.enforce_eager),
        **_llm_kwargs(args),
    )
    capture: _TargetOutputCapture | None = None
    capture_handles: list[Any] = []
    if args.capture_target_outputs:
        _, torch_model = find_vllm_model(llm)
        discovered = discover_targets(torch_model, preset=source_summary.get("target_preset") or "qv", layers=None)
        filtered = _filter_targets(
            discovered,
            targets,
            qkv_dims=_qkv_dims_from_config(args.model, local_files_only=bool(args.local_files_only)),
        )
        capture, capture_handles = _install_target_output_capture(args, filtered, backend="adapter")
    rows: list[dict[str, Any]] = []
    try:
        for spec, candidate in zip(specs, candidates):
            if capture is not None:
                capture.current_candidate_id = candidate.candidate_id
            request = LoRARequest(spec.name, spec.lora_int_id, spec.path)
            outputs = llm.generate(prompt_inputs, sampling, lora_request=request, use_tqdm=False)
            rows.extend(
                _signature_rows(
                    outputs,
                    candidate_id=candidate.candidate_id,
                    examples=examples,
                    backend="vllm_subspace_adapter",
                    prompt_tail_tokens=args.prompt_tail_tokens,
                )
            )
    finally:
        if capture is not None:
            capture.current_candidate_id = None
        _remove_capture_hooks(capture_handles)
        _save_capture(out, "adapter", capture)
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    if not args.keep_adapters:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    return rows, adapter_build_s


def _run_lazy_signatures(
    args: argparse.Namespace,
    *,
    source_summary: dict[str, Any],
    candidates: list[SubspaceCandidate],
    examples: list[Any],
    prompt_inputs: list[Any],
    targets: list[str],
    dtype: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    os.environ["OPTIMUS_LAZY_DELTA_BACKEND"] = args.lazy_delta_backend
    os.environ["OPTIMUS_LAZY_FIELD_POLICY"] = args.lazy_field_policy
    os.environ["OPTIMUS_LAZY_QKV_KERNEL_POLICY"] = args.lazy_qkv_kernel_policy
    if args.lazy_compute_dtype:
        os.environ["OPTIMUS_LAZY_COMPUTE_DTYPE"] = args.lazy_compute_dtype
    from vllm import LLM, SamplingParams

    sampling = _make_sampling(SamplingParams, max_logprobs=args.logprobs, prompt_logprobs=args.prompt_logprobs)
    llm = LLM(
        model=args.model,
        dtype=dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        enable_lora=False,
        enforce_eager=bool(args.enforce_eager),
        **_llm_kwargs(args),
    )
    _, torch_model = find_vllm_model(llm)
    _, model_runner = find_vllm_model_runner(llm)
    discovered = discover_targets(torch_model, preset=source_summary.get("target_preset") or "qv", layers=None)
    filtered = _filter_targets(
        discovered,
        targets,
        qkv_dims=_qkv_dims_from_config(args.model, local_files_only=bool(args.local_files_only)),
    )
    runtime = LazyHookRuntime(filtered, sync_timing=bool(args.sync_lazy_timing))
    runtime.basis_by_site.update(_load_basis(Path(args.source_run), effective_rank=args.adapter_rank))
    radius = _single_radius(candidates)
    runtime.beta_by_target.update(
        _load_betas(source_summary, radius=radius, scale_multiplier=float(args.scale_multiplier))
    )
    route_handle = install_model_runner_routing(runtime, model_runner)
    hook_handles = install_hooks(runtime)
    capture, capture_handles = _install_target_output_capture(args, filtered, backend="lazy")
    rows: list[dict[str, Any]] = []
    try:
        for candidate in candidates:
            runtime.set_candidate(candidate)
            if capture is not None:
                capture.current_candidate_id = candidate.candidate_id
            runtime.reset_timing()
            outputs = llm.generate(prompt_inputs, sampling, use_tqdm=False)
            if runtime.delta_rows <= 0:
                raise RuntimeError("lazy parity probe did not apply any delta rows")
            rows.extend(
                _signature_rows(
                    outputs,
                    candidate_id=candidate.candidate_id,
                    examples=examples,
                    backend="vllm_lazy_hook",
                    prompt_tail_tokens=args.prompt_tail_tokens,
                )
            )
    finally:
        if capture is not None:
            capture.current_candidate_id = None
        _remove_capture_hooks(capture_handles)
        _save_capture(Path(args.out), "lazy", capture)
        remove_hooks(hook_handles)
        remove_model_runner_routing(route_handle)
    timing = {
        "qx_time_s": runtime.qx_time_s,
        "qx_cache_hits": runtime.qx_cache_hits,
        "qx_cache_misses": runtime.qx_cache_misses,
        "row_mapping_cache_hits": runtime.row_mapping_cache_hits,
        "row_mapping_cache_misses": runtime.row_mapping_cache_misses,
        "lazy_delta_time_s": runtime.delta_time_s,
        "lazy_stack_time_s": runtime.stack_time_s,
        "lazy_meta_time_s": runtime.meta_time_s,
        "lazy_kernel_time_s": runtime.kernel_time_s,
        "delta_rows": runtime.delta_rows,
        "delta_calls": runtime.delta_calls,
    }
    del llm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return rows, timing


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe native vLLM subspace-adapter vs true-lazy hook parity on matched logprob signatures.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-id-file", action="append")
    parser.add_argument("--adapter-policy", choices=["fused-qkv-exact", "target-split"], default="target-split")
    parser.add_argument("--adapter-rank", type=int)
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--prompts", type=int, default=8)
    parser.add_argument("--prompt-input", choices=["text", "token_ids"])
    parser.add_argument("--prompt-variants")
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--exclude-source-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dtype")
    parser.add_argument("--adapter-dtype")
    parser.add_argument("--targets")
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    parser.add_argument("--max-model-len", type=int, default=0)
    parser.add_argument("--max-num-batched-tokens", type=int, default=0)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--keep-adapters", action="store_true")
    parser.add_argument("--vllm-kwarg", action="append")
    parser.add_argument("--logprobs", type=int, default=20)
    parser.add_argument("--prompt-logprobs", type=int, default=20)
    parser.add_argument("--prompt-tail-tokens", type=int, default=8)
    parser.add_argument("--lazy-delta-backend", default="vllm-lora-kernel", choices=["torch", "triton", "triton-counter", "triton-counter-inplace", "vllm-lora-kernel", "vllm-lora"])
    parser.add_argument("--lazy-field-policy", default="target-split", choices=["target-split", "fused-qkv-exact"])
    parser.add_argument(
        "--lazy-qkv-kernel-policy",
        default="split-launches",
        choices=["split-launches", "packed-qkv", "packed-qkv-from-x"],
    )
    parser.add_argument("--lazy-compute-dtype", default="bfloat16")
    parser.add_argument("--sync-lazy-timing", action="store_true")
    parser.add_argument("--capture-target-outputs", action="store_true")
    parser.add_argument("--capture-layers", default="all")
    parser.add_argument("--capture-max-rows", type=int, default=16)
    parser.add_argument("--capture-max-calls-per-target", type=int, default=2)
    parser.add_argument("--strict-token-match", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-logprob-diff", type=float, default=0.0)
    parser.add_argument("--mode", choices=["both", "adapter", "lazy", "compare"], default="both")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/vllm-config")
    configure_vllm_logging()

    source_summary = json.loads((source / "summary.json").read_text())
    state_summary = json.loads((source / "subspace_state_summary.json").read_text())
    state_payload = torch.load(source / "subspace_state.pt", map_location="cpu")
    model = args.model or source_summary.get("model") or state_summary.get("model_id_or_path") or "Qwen/Qwen3-4B"
    args.model = model
    dtype = args.dtype or source_summary.get("dtype") or "bfloat16"
    adapter_dtype = args.adapter_dtype or dtype
    prompt_input = args.prompt_input or source_summary.get("prompt_input") or "text"
    prompt_variant = (args.prompt_variants or ",".join(source_summary.get("prompt_variants") or ["tight"])).split(",", 1)[0].strip() or "tight"
    use_chat_template = bool(source_summary.get("use_chat_template", False)) if args.use_chat_template is None else bool(args.use_chat_template)
    targets = _parse_targets(args.targets or ",".join(_default_adapter_targets(source_summary, args.adapter_policy)), source_summary)
    candidates = load_subspace_candidates(source / "candidates.jsonl", _wanted(args), rng_version_override=source_summary.get("rng_version"))
    if args.candidate_id is None and args.candidate_id_file is None:
        candidates = candidates[:1]
    basis_rank = int(candidates[0].basis_rank)
    adapter_rank = int(args.adapter_rank or basis_rank)
    args.adapter_rank = adapter_rank

    exclude_ids = _exclude_ids_from_source(source) if args.exclude_source_splits else set()
    examples = load_examples(args.data, args.prompts, args.seed, exclude_ids=exclude_ids)

    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, local_files_only=bool(args.local_files_only))
    prompt_texts = make_variant_prompts(examples, prompt_variant, tokenizer=tokenizer, use_chat_template=use_chat_template)
    prompt_inputs = make_vllm_prompt_inputs(prompt_texts, tokenizer, prompt_input)
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    adapter_rows: list[dict[str, Any]] = []
    lazy_rows: list[dict[str, Any]] = []
    adapter_build_s: float | None = None
    lazy_timing: dict[str, Any] = {}
    if args.mode in {"both", "adapter"}:
        adapter_rows, adapter_build_s = _run_adapter_signatures(
            args,
            source=source,
            out=out,
            source_summary=source_summary,
            state_summary=state_summary,
            state_payload=state_payload,
            candidates=candidates,
            examples=examples,
            prompt_inputs=prompt_inputs,
            targets=targets,
            adapter_rank=adapter_rank,
            dtype=dtype,
            adapter_dtype=adapter_dtype,
        )
        _write_jsonl(out / "adapter_signatures.jsonl", adapter_rows)
    elif (out / "adapter_signatures.jsonl").exists():
        adapter_rows = _jsonl(out / "adapter_signatures.jsonl")

    if args.mode in {"both", "lazy"}:
        lazy_rows, lazy_timing = _run_lazy_signatures(
            args,
            source_summary=source_summary,
            candidates=candidates,
            examples=examples,
            prompt_inputs=prompt_inputs,
            targets=targets,
            dtype=dtype,
        )
        _write_jsonl(out / "lazy_signatures.jsonl", lazy_rows)
    elif (out / "lazy_signatures.jsonl").exists():
        lazy_rows = _jsonl(out / "lazy_signatures.jsonl")

    comparisons: list[dict[str, Any]] = []
    parity: dict[str, Any] = {}
    target_output_drift: dict[str, Any] = {}
    pass_status = True
    if adapter_rows and lazy_rows:
        comparisons, parity = _compare_rows(adapter_rows, lazy_rows)
        pass_status = bool(parity["generated_token_match_rate"] == 1.0 or not args.strict_token_match)
        max_top = parity.get("max_common_top_logprob_abs_diff")
        if max_top is not None and float(max_top) > float(args.max_logprob_diff):
            pass_status = False
    elif args.mode == "compare":
        raise FileNotFoundError(f"missing signature files under {out}")
    if args.capture_target_outputs:
        adapter_capture_path = _capture_path(out, "adapter")
        lazy_capture_path = _capture_path(out, "lazy")
        if adapter_capture_path.exists() and lazy_capture_path.exists():
            capture_comparisons, target_output_drift = _compare_capture_rows(
                torch.load(adapter_capture_path, map_location="cpu"),
                torch.load(lazy_capture_path, map_location="cpu"),
            )
            _write_capture_comparisons(out, capture_comparisons)
    summary = {
        "kind": "vllm_subspace_lazy_signature_parity_probe",
        "status": "partial" if not (adapter_rows and lazy_rows) else ("pass" if pass_status else "fail"),
        "mode": args.mode,
        "source_run": str(source),
        "model": model,
        "data": args.data,
        "seed": int(args.seed),
        "prompts": len(examples),
        "candidate_ids": [candidate.candidate_id for candidate in candidates],
        "basis_rank": basis_rank,
        "adapter_rank": adapter_rank,
        "targets": targets,
        "adapter_policy": args.adapter_policy,
        "scale_multiplier": float(args.scale_multiplier),
        "dtype": dtype,
        "adapter_dtype": adapter_dtype,
        "lazy_delta_backend": args.lazy_delta_backend,
        "lazy_field_policy": args.lazy_field_policy,
        "lazy_qkv_kernel_policy": args.lazy_qkv_kernel_policy,
        "lazy_compute_dtype": args.lazy_compute_dtype,
        "max_logprob_diff": float(args.max_logprob_diff),
        "strict_token_match": bool(args.strict_token_match),
        "prompt_variant": prompt_variant,
        "prompt_input": prompt_input,
        "use_chat_template": use_chat_template,
        "logprobs": int(args.logprobs),
        "prompt_logprobs": int(args.prompt_logprobs),
        "prompt_tail_tokens": int(args.prompt_tail_tokens),
        "parity": parity,
        "adapter_build_s": adapter_build_s,
        "lazy_timing": lazy_timing,
        "target_output_drift": target_output_drift,
        "capture_target_outputs": bool(args.capture_target_outputs),
        "capture_layers": args.capture_layers,
        "capture_max_rows": int(args.capture_max_rows),
        "capture_max_calls_per_target": int(args.capture_max_calls_per_target),
        "runtime_environment": runtime_environment(),
        "git_commit": git_commit(),
        "git_dirty": git_dirty(),
        "decode_config_hash": config_hash({"max_tokens": 1, "temperature": 0.0, "logprobs": args.logprobs, "prompt_logprobs": args.prompt_logprobs}),
    }
    if comparisons:
        _write_jsonl(out / "comparisons.jsonl", comparisons)
    summary_text = json.dumps(summary, indent=2, sort_keys=True) + "\n"
    write_json(out / "summary.json", summary)
    (out / "summary_hash.txt").write_text(sha256_bytes(summary_text.encode("utf-8")) + "\n")
    print(summary_text)
    return 0 if summary["status"] in {"pass", "partial"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
