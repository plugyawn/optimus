#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch
from transformers import AutoConfig

from optimus.backends.vllm_lazy_hook import (
    LazyHookRuntime,
    _candidate_batch_context,
    _chunked,
    _score_outputs,
    _split_candidate_outputs,
    discover_targets,
    find_vllm_model,
    find_vllm_model_runner,
    install_hooks,
    install_model_runner_routing,
    remove_model_runner_routing,
    remove_hooks,
)
from optimus.serving.prompting import make_vllm_prompt_inputs
from optimus.serving.runtime import (
    configure_vllm_logging,
    make_sampling_params,
    optional_vllm_kwargs,
    runtime_environment,
    write_json,
)
from optimus.subspace import SubspaceCandidate
from optimus.subspace.reference import config_hash, git_commit, git_dirty, sha256_bytes
from optimus.tasks.countdown import load_examples
from optimus.tasks.prompt_variants import make_variant_prompts


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_jsonl_overwrite(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _candidate_map(source: Path, *, rng_version_override: str | None = None) -> dict[str, SubspaceCandidate]:
    out = {}
    for row in _jsonl(source / "candidates.jsonl"):
        candidate = SubspaceCandidate(**row)
        if rng_version_override:
            candidate = replace(candidate, rng_version=str(rng_version_override))
        out[candidate.candidate_id] = candidate
    return out


def _exclude_ids_from_source(source: Path) -> set[int]:
    out = set()
    for name in ("per_prompt.jsonl", "holdout_per_prompt.jsonl"):
        path = source / name
        if not path.exists():
            continue
        for row in _jsonl(path):
            if "example_id" not in row:
                continue
            split = row.get("split")
            candidate_id = row.get("candidate_id")
            mode = str(row.get("mode") or "")
            if candidate_id == "__base__" and split in {"screen", "holdout"}:
                out.add(int(row["example_id"]))
            elif row.get("candidate") == "base" and mode in {"base_screen", "base_holdout"}:
                out.add(int(row["example_id"]))
    return out


def _load_basis(source: Path, *, effective_rank: int | None = None) -> dict[str, torch.Tensor]:
    payload = torch.load(source / "subspace_state.pt", map_location="cpu")
    tensors = payload["basis_tensors"]
    by_site = {}
    for key, tensor in tensors.items():
        site_id = str(key).split("basis/", 1)[-1]
        basis = tensor.detach().cpu().float().contiguous()
        if effective_rank is not None:
            if effective_rank <= 0 or effective_rank > int(basis.shape[0]):
                raise ValueError(f"effective_rank must be in [1, {int(basis.shape[0])}] for {site_id}, got {effective_rank}")
            basis = basis[:effective_rank].contiguous()
        by_site[site_id] = basis
    return by_site


def _expand_fused_qkv_betas(beta_by_target: dict[str, float]) -> dict[str, float]:
    """Expose fused qkv scales under split q/k/v ids for target-split parity."""

    out = dict(beta_by_target)
    for target_id, beta in list(beta_by_target.items()):
        if not target_id.endswith(".self_attn.qkv_proj"):
            continue
        prefix = target_id[: -len("qkv_proj")]
        for suffix in ("q_proj", "k_proj", "v_proj"):
            out.setdefault(prefix + suffix, beta)
    return out


def _load_betas(summary: dict[str, Any], *, radius: float | None = None, scale_multiplier: float = 1.0) -> dict[str, float]:
    out = {}
    for row in summary["resolved_target_scales"]:
        values = row["beta_t_by_radius"]
        if not values:
            continue
        key = f"{radius:g}" if radius is not None else None
        if key is not None and key not in values:
            raise ValueError(f"summary has no beta for radius {key!r} on {row['target_id']}")
        value = values[key] if key is not None else next(iter(values.values()))
        out[row["target_id"]] = float(scale_multiplier) * float(value)
    if not out:
        raise ValueError("source summary is missing resolved_target_scales")
    return _expand_fused_qkv_betas(out)


def _default_targets(source_summary: dict[str, Any]) -> list[str]:
    preset = str(source_summary.get("target_preset") or "")
    if preset == "qv":
        return ["q_proj", "v_proj"]
    if preset == "attn-qkvo":
        return ["q_proj", "k_proj", "v_proj", "o_proj"]
    if preset == "mlp":
        return ["gate_proj", "up_proj", "down_proj"]
    if preset == "transformer-linears":
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    return ["q_proj", "v_proj"]


def _qkv_dims_from_config(model: str, *, local_files_only: bool = False) -> tuple[int, int]:
    config = AutoConfig.from_pretrained(model, trust_remote_code=True, local_files_only=local_files_only)
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    return heads * head_dim, kv_heads * head_dim


def _parse_targets(text: str | None, source_summary: dict[str, Any]) -> list[str]:
    raw = text or ",".join(_default_targets(source_summary))
    targets = [item.strip() for item in raw.split(",") if item.strip()]
    if not targets:
        raise ValueError("at least one target suffix is required")
    return targets


def _filter_targets(targets: list[Any], suffixes: list[str], *, qkv_dims: tuple[int, int] | None = None) -> list[Any]:
    wanted = set(suffixes)
    filtered = [target for target in targets if target.suffix in wanted]
    wanted_qkv_slices = tuple(suffix for suffix in ("q_proj", "k_proj", "v_proj") if suffix in wanted)
    if wanted_qkv_slices:
        direct_attn_layers = {target.layer_index for target in filtered if target.suffix in {"q_proj", "k_proj", "v_proj"}}
        q_out, kv_out = qkv_dims or (None, None)
        for target in targets:
            if target.suffix != "qkv_proj" or target.layer_index in direct_attn_layers:
                continue
            if q_out is None or kv_out is None:
                raise ValueError("fused qkv target filtering requires q/k/v dimensions")
            filtered.append(
                replace(
                    target,
                    fused_qkv_slices=wanted_qkv_slices,
                    fused_q_out=int(q_out),
                    fused_kv_out=int(kv_out),
                )
            )
    if not filtered:
        available = sorted({target.suffix for target in targets})
        raise ValueError(f"no vLLM lazy hook targets matched {sorted(wanted)}; available suffixes: {available}")
    return filtered


def _single_radius(candidates: list[SubspaceCandidate]) -> float:
    values = {f"{float(candidate.rho_or_sigma_w):g}" for candidate in candidates}
    if len(values) != 1:
        raise ValueError(f"mixed candidate radii are not supported in one replay: {sorted(values)}")
    return float(next(iter(values)))


def _candidate_score_record(
    candidate: SubspaceCandidate,
    *,
    score: float,
    tokens: int,
    elapsed: float,
    base_score: float,
    sample_count: int,
    stage: str,
    source: Path,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate.candidate_id,
        "split": "final",
        "selection_stage": stage,
        "aggregate_metrics": {"exact": float(score), "delta_vs_base": float(score) - float(base_score)},
        "elapsed_s": float(elapsed),
        "output_tokens": int(tokens),
        "sample_count": int(sample_count),
        "direction_seed": int(candidate.direction_seed),
        "sign": candidate.sign,
        "source_run": str(source),
    }


def _prompt_slices(count: int, batch_size: int | None) -> list[tuple[int, int]]:
    if batch_size is None or batch_size <= 0 or batch_size >= count:
        return [(0, count)]
    return [(start, min(start + batch_size, count)) for start in range(0, count, batch_size)]


def _generate_in_prompt_batches(
    llm: Any,
    prompt_inputs: list[Any],
    sampling: Any,
    *,
    prompt_batch_size: int | None,
) -> tuple[list[Any], float]:
    outputs: list[Any] = []
    started = time.perf_counter()
    for start, end in _prompt_slices(len(prompt_inputs), prompt_batch_size):
        outputs.extend(llm.generate(prompt_inputs[start:end], sampling, use_tqdm=False))
    return outputs, time.perf_counter() - started


def _evaluate_candidates(
    *,
    llm: Any,
    sampling: Any,
    runtime: LazyHookRuntime,
    examples: list[Any],
    prompt_inputs: list[Any],
    candidates: list[SubspaceCandidate],
    candidate_batch_size: int,
    prompt_batch_size: int | None,
    max_new_tokens: int,
    base_score: float,
    stage: str,
    source: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    score_rows: list[dict[str, Any]] = []
    per_prompt_rows: list[dict[str, Any]] = []
    total_tokens = 0
    total_qx = 0.0
    total_delta = 0.0
    total_stack = 0.0
    total_meta = 0.0
    total_kernel = 0.0
    total_delta_rows = 0
    total_delta_calls = 0
    started_all = time.perf_counter()
    for candidate_chunk in _chunked(candidates, max(1, int(candidate_batch_size))):
        runtime.reset_timing()
        started = time.perf_counter()
        outputs_by_candidate: dict[str, list[Any]] = {candidate.candidate_id: [] for candidate in candidate_chunk}
        if len(candidate_chunk) == 1:
            runtime.set_candidate(candidate_chunk[0])
            for start_idx, end_idx in _prompt_slices(len(prompt_inputs), prompt_batch_size):
                outputs = llm.generate(prompt_inputs[start_idx:end_idx], sampling, use_tqdm=False)
                outputs_by_candidate[candidate_chunk[0].candidate_id].extend(outputs)
        else:
            for start_idx, end_idx in _prompt_slices(len(prompt_inputs), prompt_batch_size):
                batched_prompts = []
                for _candidate in candidate_chunk:
                    batched_prompts.extend(prompt_inputs[start_idx:end_idx])
                with _candidate_batch_context(runtime, llm, candidate_chunk, end_idx - start_idx):
                    outputs = llm.generate(batched_prompts, sampling, use_tqdm=False)
                split = _split_candidate_outputs(outputs, candidate_chunk, examples[start_idx:end_idx])
                for candidate_id, candidate_outputs in split.items():
                    outputs_by_candidate[candidate_id].extend(candidate_outputs)
        elapsed = time.perf_counter() - started
        if runtime.delta_rows <= 0:
            raise RuntimeError("vLLM lazy hook did not apply any perturbation rows; refusing to report true-lazy results")
        total_qx += runtime.qx_time_s
        total_delta += runtime.delta_time_s
        total_stack += runtime.stack_time_s
        total_meta += runtime.meta_time_s
        total_kernel += runtime.kernel_time_s
        total_delta_rows += runtime.delta_rows
        total_delta_calls += runtime.delta_calls
        for candidate in candidate_chunk:
            outputs_for_candidate = outputs_by_candidate[candidate.candidate_id]
            score, tokens, rows = _score_outputs(examples, outputs_for_candidate, max_new_tokens=max_new_tokens)
            total_tokens += tokens
            per_prompt_rows.extend({"split": "final", "candidate_id": candidate.candidate_id, **row} for row in rows)
            score_rows.append(
                _candidate_score_record(
                    candidate,
                    score=score,
                    tokens=tokens,
                    elapsed=elapsed / max(len(candidate_chunk), 1),
                    base_score=base_score,
                    sample_count=len(examples),
                    stage=stage,
                    source=source,
                )
            )
    elapsed_all = time.perf_counter() - started_all
    timing = {
        "elapsed_s": elapsed_all,
        "output_tokens": total_tokens,
        "qx_time_s": total_qx,
        "lazy_delta_time_s": total_delta,
        "lazy_stack_time_s": total_stack,
        "lazy_meta_time_s": total_meta,
        "lazy_kernel_time_s": total_kernel,
        "delta_rows": total_delta_rows,
        "delta_calls": total_delta_calls,
        "candidate_replay_sec": elapsed_all / max(len(candidates), 1),
        "mixed_candidate_sec": len(candidates) / max(elapsed_all, 1e-9),
    }
    return score_rows, per_prompt_rows, timing


def _top_candidates(candidates: list[SubspaceCandidate], score_rows: list[dict[str, Any]], top_k: int) -> list[SubspaceCandidate]:
    if top_k <= 0:
        return []
    by_id = {candidate.candidate_id: candidate for candidate in candidates}
    ordered = sorted(score_rows, key=lambda row: (float(row["aggregate_metrics"]["exact"]), str(row["candidate_id"])), reverse=True)
    selected = []
    seen = set()
    for row in ordered:
        candidate_id = str(row["candidate_id"])
        if candidate_id == "__base__" or candidate_id in seen:
            continue
        selected.append(by_id[candidate_id])
        seen.add(candidate_id)
        if len(selected) >= top_k:
            break
    return selected


def _selected_prompt_variant(summary: dict[str, Any]) -> str:
    prompt_variants = str(summary.get("prompt_contract_hash") or "")
    del prompt_variants
    command = summary.get("command") or []
    for idx, item in enumerate(command):
        if item == "--prompt-variants" and idx + 1 < len(command):
            return str(command[idx + 1]).split(",", 1)[0].strip() or "default"
    return str(summary.get("prompt_variants") or "tight").split(",", 1)[0].strip() or "default"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay specific lazy subspace K=1 candidates on a fresh Countdown split.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--candidate-id", action="append")
    parser.add_argument("--candidate-id-file", action="append")
    parser.add_argument("--effective-rank", type=int)
    parser.add_argument("--adapter-rank", type=int, help="Alias for --effective-rank when comparing with subspace-as-LoRA runs.")
    parser.add_argument("--scale-multiplier", type=float, default=1.0)
    parser.add_argument("--targets")
    parser.add_argument("--candidate-batch-size", type=int, default=1)
    parser.add_argument(
        "--prompt-batch-size",
        type=int,
        default=0,
        help="Split each candidate replay into prompt microbatches; 0 means all prompts in one vLLM call.",
    )
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
    parser.add_argument(
        "--exclude-run",
        action="append",
        default=[],
        help="Additional run directory whose source screen/holdout example IDs should be excluded.",
    )
    parser.add_argument("--dtype")
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
    candidates = _candidate_map(source, rng_version_override=source_summary.get("rng_version"))
    candidate_ids = list(args.candidate_id or [])
    for item in args.candidate_id_file or []:
        candidate_ids.extend(line.strip() for line in Path(item).read_text().splitlines() if line.strip())
    if not candidate_ids:
        raise SystemExit("provide at least one --candidate-id or --candidate-id-file")
    selected = []
    for candidate_id in candidate_ids:
        if candidate_id not in candidates:
            raise SystemExit(f"candidate_id {candidate_id!r} not found in {source / 'candidates.jsonl'}")
        selected.append(candidates[candidate_id])
    source_basis_rank = int(selected[0].basis_rank)
    if any(int(candidate.basis_rank) != source_basis_rank for candidate in selected):
        raise SystemExit("mixed basis ranks are not supported in one lazy replay")
    effective_rank = int(args.effective_rank or args.adapter_rank or source_basis_rank)
    if effective_rank <= 0 or effective_rank > source_basis_rank:
        raise SystemExit(f"--effective-rank must be in [1, {source_basis_rank}], got {effective_rank}")
    if args.scale_multiplier <= 0:
        raise SystemExit("--scale-multiplier must be positive")
    radius = _single_radius(selected)

    exclude_ids = _exclude_ids_from_source(source) if args.exclude_source_splits else set()
    for exclude_run in args.exclude_run:
        exclude_ids.update(_exclude_ids_from_source(Path(exclude_run)))
    examples = load_examples(args.data, args.prompts, args.seed, exclude_ids=exclude_ids)
    prompt_variant = args.prompt_variants or _selected_prompt_variant(source_summary)
    prompt_input = args.prompt_input or source_summary.get("prompt_input") or "text"
    max_new_tokens = int(args.max_new_tokens or source_summary.get("max_new_tokens") or 64)
    use_chat_template = bool(source_summary.get("use_chat_template", False)) if args.use_chat_template is None else bool(args.use_chat_template)
    stop_at_answer = bool(source_summary.get("stop_at_answer", True)) if args.stop_at_answer is None else bool(args.stop_at_answer)
    model_id = args.model or source_summary.get("model") or source_summary.get("model_id_or_path") or "Qwen/Qwen3-4B"
    dtype = args.dtype or source_summary.get("dtype") or "bfloat16"
    target_suffixes = _parse_targets(args.targets, source_summary)

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
    llm_kwargs.setdefault("tensor_parallel_size", args.tensor_parallel_size)
    llm_kwargs.setdefault("enable_prefix_caching", False)
    llm_kwargs.setdefault("enforce_eager", bool(args.enforce_eager))
    llm_kwargs.setdefault("trust_remote_code", True)
    llm_kwargs.setdefault("gpu_memory_utilization", args.gpu_memory_utilization)
    llm = LLM(model=model_id, dtype=dtype, **llm_kwargs)
    _, model = find_vllm_model(llm)
    qkv_dims = _qkv_dims_from_config(model_id, local_files_only=bool(args.local_files_only))
    targets = _filter_targets(
        discover_targets(model, preset=source_summary.get("target_preset") or "qv", layers=None),
        target_suffixes,
        qkv_dims=qkv_dims,
    )
    runtime = LazyHookRuntime(targets)
    runtime.basis_by_site.update(_load_basis(source, effective_rank=effective_rank))
    runtime.beta_by_target.update(_load_betas(source_summary, radius=radius, scale_multiplier=float(args.scale_multiplier)))
    handles = install_hooks(runtime)
    routing_handle = None
    if int(args.candidate_batch_size) > 1:
        _, model_runner = find_vllm_model_runner(llm)
        routing_handle = install_model_runner_routing(runtime, model_runner)
    try:
        tokenizer = llm.get_tokenizer()
        texts = make_variant_prompts(examples, prompt_variant, tokenizer=tokenizer, use_chat_template=use_chat_template)
        prompt_inputs = make_vllm_prompt_inputs(texts, tokenizer, prompt_input)
        sampling = make_sampling_params(SamplingParams, max_new_tokens, stop_at_answer)

        score_rows = []
        per_prompt_rows = []
        total_tokens = 0
        prompt_batch_size = int(args.prompt_batch_size or 0)
        prompt_batch_size_or_none = prompt_batch_size if prompt_batch_size > 0 else None
        runtime.set_candidate(None)
        base_outputs, base_elapsed = _generate_in_prompt_batches(
            llm,
            prompt_inputs,
            sampling,
            prompt_batch_size=prompt_batch_size_or_none,
        )
        base_score, base_tokens, base_prompt_rows = _score_outputs(examples, base_outputs, max_new_tokens=max_new_tokens)
        total_tokens += base_tokens
        per_prompt_rows.extend({"split": "final", "candidate_id": "__base__", **row} for row in base_prompt_rows)
        score_rows.append(
            {
                "candidate_id": "__base__",
                "split": "final",
                "selection_stage": "base_final",
                "aggregate_metrics": {"exact": base_score},
                "elapsed_s": base_elapsed,
                "output_tokens": base_tokens,
                "sample_count": len(examples),
            }
        )

        candidate_rows, candidate_per_prompt_rows, candidate_timing = _evaluate_candidates(
            llm=llm,
            sampling=sampling,
            runtime=runtime,
            examples=examples,
            prompt_inputs=prompt_inputs,
            candidates=selected,
            candidate_batch_size=int(args.candidate_batch_size),
            prompt_batch_size=prompt_batch_size_or_none,
            max_new_tokens=max_new_tokens,
            base_score=base_score,
            stage="k1_final_replay",
            source=source,
        )
        total_tokens += int(candidate_timing["output_tokens"])
        score_rows.extend(candidate_rows)
        per_prompt_rows.extend(candidate_per_prompt_rows)
        confirm_candidates = _top_candidates(selected, candidate_rows, int(args.confirm_top_k))
        confirmed_rows: list[dict[str, Any]] = []
        confirmed_per_prompt_rows: list[dict[str, Any]] = []
        confirmed_timing: dict[str, Any] | None = None
        if confirm_candidates:
            confirmed_rows, confirmed_per_prompt_rows, confirmed_timing = _evaluate_candidates(
                llm=llm,
                sampling=sampling,
                runtime=runtime,
                examples=examples,
                prompt_inputs=prompt_inputs,
                candidates=confirm_candidates,
                candidate_batch_size=1,
                prompt_batch_size=prompt_batch_size_or_none,
                max_new_tokens=max_new_tokens,
                base_score=base_score,
                stage="k1_final_confirmed_chunk1",
                source=source,
            )
            total_tokens += int(confirmed_timing["output_tokens"])

        scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
        confirmed_scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in confirmed_rows)
        candidate_final_scores = {
            row["candidate_id"]: row["aggregate_metrics"]["exact"]
            for row in score_rows
            if row["candidate_id"] != "__base__"
        }
        confirmed_candidate_scores = {row["candidate_id"]: row["aggregate_metrics"]["exact"] for row in confirmed_rows}
        best_candidate_row = max(
            (row for row in score_rows if row["candidate_id"] != "__base__"),
            key=lambda row: (row["aggregate_metrics"]["exact"], row["candidate_id"]),
            default={"candidate_id": None, "aggregate_metrics": {"exact": base_score}},
        )
        confirmed_best_row = max(
            confirmed_rows,
            key=lambda row: (row["aggregate_metrics"]["exact"], row["candidate_id"]),
            default=None,
        )
        summary = {
            "kind": "vllm_lazy_k1_final_replay",
            "source_run": str(source),
            "model": model_id,
            "data": args.data,
            "seed": args.seed,
            "prompts": len(examples),
            "excluded_source_example_ids": len(exclude_ids),
            "population": len(selected),
            "basis_rank": source_basis_rank,
            "effective_rank": effective_rank,
            "scale_multiplier": float(args.scale_multiplier),
            "targets": target_suffixes,
            "dtype": dtype,
            "candidate_batch_size": int(args.candidate_batch_size),
            "prompt_batch_size": prompt_batch_size,
            "routing_patch": routing_handle is not None,
            "lazy_delta_backend": runtime.delta_backend,
            "candidate_ids": [candidate.candidate_id for candidate in selected],
            "base_final_score": base_score,
            "candidate_final_scores": candidate_final_scores,
            "best_candidate_final_score": float(best_candidate_row["aggregate_metrics"]["exact"]),
            "best_candidate_id": best_candidate_row["candidate_id"],
            "candidate_replay_sec": candidate_timing["candidate_replay_sec"],
            "mixed_candidate_sec": candidate_timing["mixed_candidate_sec"],
            "confirm_top_k": int(args.confirm_top_k),
            "confirmed_population": len(confirm_candidates),
            "confirmed_candidate_final_scores": confirmed_candidate_scores,
            "confirmed_best_candidate_final_score": None if confirmed_best_row is None else float(confirmed_best_row["aggregate_metrics"]["exact"]),
            "confirmed_best_candidate_id": None if confirmed_best_row is None else confirmed_best_row["candidate_id"],
            "confirmed_candidate_replay_sec": None if confirmed_timing is None else confirmed_timing["candidate_replay_sec"],
            "confirmed_mixed_candidate_sec": None if confirmed_timing is None else confirmed_timing["mixed_candidate_sec"],
            "lazy_timing": candidate_timing,
            "confirmed_lazy_timing": confirmed_timing,
            "candidate_scores_hash": sha256_bytes(scores_text.encode("utf-8")),
            "confirmed_candidate_scores_hash": sha256_bytes(confirmed_scores_text.encode("utf-8")) if confirmed_rows else None,
            "runtime_environment": runtime_environment(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "prompt_variant": prompt_variant,
            "prompt_input": prompt_input,
            "use_chat_template": use_chat_template,
            "decode_config_hash": config_hash({"max_new_tokens": max_new_tokens, "stop_at_answer": stop_at_answer}),
            "output_tokens": total_tokens,
        }
        write_json(out / "summary.json", summary)
        (out / "candidate_scores.jsonl").write_text(scores_text)
        if confirmed_rows:
            (out / "confirmed_candidate_scores.jsonl").write_text(confirmed_scores_text)
            _write_jsonl_overwrite(out / "confirmed_per_prompt.jsonl", confirmed_per_prompt_rows)
        _write_jsonl_overwrite(out / "per_prompt.jsonl", per_prompt_rows)
        _write_jsonl_overwrite(out / "candidates.jsonl", [asdict(candidate) for candidate in selected])
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        runtime.set_candidate(None)
        remove_model_runner_routing(routing_handle)
        remove_hooks(handles)


if __name__ == "__main__":
    raise SystemExit(main())
