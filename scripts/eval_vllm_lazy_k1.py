#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from optimus.backends.vllm_lazy_hook import (
    LazyHookRuntime,
    _score_outputs,
    discover_targets,
    find_vllm_model,
    install_hooks,
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


def _candidate_map(source: Path) -> dict[str, SubspaceCandidate]:
    out = {}
    for row in _jsonl(source / "candidates.jsonl"):
        candidate = SubspaceCandidate(**row)
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


def _load_basis(source: Path) -> dict[str, torch.Tensor]:
    payload = torch.load(source / "subspace_state.pt", map_location="cpu")
    tensors = payload["basis_tensors"]
    by_site = {}
    for key, tensor in tensors.items():
        site_id = str(key).split("basis/", 1)[-1]
        by_site[site_id] = tensor
    return by_site


def _load_betas(summary: dict[str, Any]) -> dict[str, float]:
    out = {}
    for row in summary["resolved_target_scales"]:
        values = row["beta_t_by_radius"]
        if not values:
            continue
        out[row["target_id"]] = float(next(iter(values.values())))
    return out


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
    parser.add_argument("--model")
    parser.add_argument("--data", required=True)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--prompts", type=int, default=128)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--prompt-input", choices=["text", "token_ids"], default="text")
    parser.add_argument("--prompt-variants")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--stop-at-answer", action="store_true")
    parser.add_argument("--exclude-source-splits", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--exclude-run",
        action="append",
        default=[],
        help="Additional run directory whose source screen/holdout example IDs should be excluded.",
    )
    parser.add_argument("--vllm-kwarg", action="append")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source = Path(args.source_run)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    source_summary = json.loads((source / "summary.json").read_text())
    candidates = _candidate_map(source)
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

    exclude_ids = _exclude_ids_from_source(source) if args.exclude_source_splits else set()
    for exclude_run in args.exclude_run:
        exclude_ids.update(_exclude_ids_from_source(Path(exclude_run)))
    examples = load_examples(args.data, args.prompts, args.seed, exclude_ids=exclude_ids)
    prompt_variant = args.prompt_variants or _selected_prompt_variant(source_summary)
    model_id = args.model or source_summary.get("model") or source_summary.get("model_id_or_path") or "Qwen/Qwen3-4B"

    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/vllm-config")
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    configure_vllm_logging()
    from vllm import LLM, SamplingParams

    vllm_args = SimpleNamespace(
        tensor_parallel_size=1,
        max_num_batched_tokens=0,
        enable_prefix_caching=False,
        enable_chunked_prefill=None,
        kv_cache_dtype="",
        vllm_kwarg=args.vllm_kwarg or [],
    )
    llm_kwargs = optional_vllm_kwargs(vllm_args)
    llm_kwargs.setdefault("tensor_parallel_size", 1)
    llm_kwargs.setdefault("enable_prefix_caching", False)
    llm_kwargs.setdefault("enforce_eager", True)
    llm_kwargs.setdefault("trust_remote_code", True)
    llm_kwargs.setdefault("gpu_memory_utilization", 0.82)
    llm = LLM(model=model_id, **llm_kwargs)
    _, model = find_vllm_model(llm)
    targets = discover_targets(model, preset=source_summary.get("target_preset") or "qv", layers=None)
    runtime = LazyHookRuntime(targets)
    runtime.basis_by_site.update(_load_basis(source))
    runtime.beta_by_target.update(_load_betas(source_summary))
    handles = install_hooks(runtime)
    try:
        tokenizer = llm.get_tokenizer()
        texts = make_variant_prompts(examples, prompt_variant, tokenizer=tokenizer, use_chat_template=bool(args.use_chat_template))
        prompt_inputs = make_vllm_prompt_inputs(texts, tokenizer, args.prompt_input)
        sampling = make_sampling_params(SamplingParams, args.max_new_tokens, bool(args.stop_at_answer))

        score_rows = []
        per_prompt_rows = []
        total_tokens = 0
        runtime.set_candidate(None)
        base_started = time.perf_counter()
        base_outputs = llm.generate(prompt_inputs, sampling, use_tqdm=False)
        base_elapsed = time.perf_counter() - base_started
        base_score, base_tokens, base_prompt_rows = _score_outputs(examples, base_outputs, max_new_tokens=args.max_new_tokens)
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

        started_all = time.perf_counter()
        for candidate in selected:
            runtime.reset_timing()
            runtime.set_candidate(candidate)
            started = time.perf_counter()
            outputs = llm.generate(prompt_inputs, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            if runtime.delta_rows <= 0:
                raise RuntimeError(f"candidate {candidate.candidate_id} did not apply lazy rows")
            score, tokens, rows = _score_outputs(examples, outputs, max_new_tokens=args.max_new_tokens)
            total_tokens += tokens
            per_prompt_rows.extend({"split": "final", "candidate_id": candidate.candidate_id, **row} for row in rows)
            score_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "split": "final",
                    "selection_stage": "k1_final_replay",
                    "aggregate_metrics": {"exact": score, "delta_vs_base": score - base_score},
                    "elapsed_s": elapsed,
                    "output_tokens": tokens,
                    "sample_count": len(examples),
                    "direction_seed": candidate.direction_seed,
                    "sign": candidate.sign,
                    "source_run": str(source),
                }
            )
        elapsed_all = time.perf_counter() - started_all

        scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
        summary = {
            "kind": "vllm_lazy_k1_final_replay",
            "source_run": str(source),
            "model": model_id,
            "data": args.data,
            "seed": args.seed,
            "prompts": len(examples),
            "excluded_source_example_ids": len(exclude_ids),
            "candidate_ids": [candidate.candidate_id for candidate in selected],
            "base_final_score": base_score,
            "candidate_final_scores": {
                row["candidate_id"]: row["aggregate_metrics"]["exact"]
                for row in score_rows
                if row["candidate_id"] != "__base__"
            },
            "best_candidate_final_score": max(
                (row["aggregate_metrics"]["exact"] for row in score_rows if row["candidate_id"] != "__base__"),
                default=base_score,
            ),
            "best_candidate_id": max(
                (row for row in score_rows if row["candidate_id"] != "__base__"),
                key=lambda row: (row["aggregate_metrics"]["exact"], row["candidate_id"]),
                default={"candidate_id": None},
            )["candidate_id"],
            "candidate_replay_sec": elapsed_all / max(len(selected), 1),
            "candidate_scores_hash": sha256_bytes(scores_text.encode("utf-8")),
            "runtime_environment": runtime_environment(),
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "prompt_variant": prompt_variant,
            "decode_config_hash": config_hash({"max_new_tokens": args.max_new_tokens, "stop_at_answer": bool(args.stop_at_answer)}),
            "output_tokens": total_tokens,
        }
        write_json(out / "summary.json", summary)
        (out / "candidate_scores.jsonl").write_text(scores_text)
        _write_jsonl_overwrite(out / "per_prompt.jsonl", per_prompt_rows)
        _write_jsonl_overwrite(out / "candidates.jsonl", [asdict(candidate) for candidate in selected])
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    finally:
        runtime.set_candidate(None)
        remove_hooks(handles)


if __name__ == "__main__":
    raise SystemExit(main())
