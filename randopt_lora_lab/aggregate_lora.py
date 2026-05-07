from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from .countdown import load_examples, unique_example_count, unique_semantic_example_count
from .experiments import evaluate_candidate, make_backend, parse_candidate_key, reset_outputs, tag_rows, write_jsonl
from .lora_space import Candidate, lora_noise_tensors


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def top_candidate_rows(run_dir: Path, top_k: int, score_col: str = "exact_mean") -> list[dict]:
    rows = read_jsonl(run_dir / "candidate_summary.jsonl")
    rows = [row for row in rows if row.get("candidate") and row["candidate"] != "base"]
    rows.sort(key=lambda row: float(row.get(score_col, 0.0)), reverse=True)
    return rows[:top_k]


def normalized_weights(scores: list[float], mode: str) -> list[float]:
    if not scores:
        raise ValueError("cannot build aggregate without scores")
    if mode == "uniform":
        weights = [1.0 for _ in scores]
    elif mode == "centered":
        mean_score = sum(scores) / len(scores)
        weights = [score - mean_score for score in scores]
        if all(abs(weight) < 1e-12 for weight in weights):
            weights = [1.0 for _ in scores]
    elif mode == "score":
        floor = min(scores)
        weights = [score - floor for score in scores]
        if all(abs(weight) < 1e-12 for weight in weights):
            weights = [1.0 for _ in scores]
    else:
        raise ValueError(f"unknown weight mode: {mode}")
    norm = math.sqrt(sum(weight * weight for weight in weights))
    if norm <= 0.0:
        raise ValueError("aggregate weights have zero norm")
    return [weight / norm for weight in weights]


def aggregate_lora_tensors(
    module_name: str,
    in_features: int,
    out_features: int,
    candidates: list[Candidate],
    weights: list[float],
    base_rank: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    if len(candidates) != len(weights):
        raise ValueError("candidates and weights must have the same length")
    a_parts = []
    b_parts = []
    for candidate, weight in zip(candidates, weights):
        a, b = lora_noise_tensors(
            module_name,
            (base_rank, in_features),
            (out_features, base_rank),
            candidate,
            base_rank,
        )
        a_parts.append(a)
        b_parts.append(b * float(weight))
    return torch.cat(a_parts, dim=0).contiguous(), torch.cat(b_parts, dim=1).contiguous()


def build_aggregate_state(model, rows: list[dict], base_rank: int, weight_mode: str) -> tuple[dict, dict]:
    candidates = [parse_candidate_key(row["candidate"]) for row in rows]
    scores = [float(row.get("exact_mean", 0.0)) for row in rows]
    weights = normalized_weights(scores, weight_mode)
    state = {}
    modules = []
    expected_rank = base_rank * len(candidates)
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not module.lora_A:
            continue
        adapter = next(iter(module.lora_A.keys()))
        a_weight = module.lora_A[adapter].weight
        b_weight = module.lora_B[adapter].weight
        if int(a_weight.shape[0]) != expected_rank:
            raise ValueError(f"{name} has rank {a_weight.shape[0]}, expected aggregate rank {expected_rank}")
        a, b = aggregate_lora_tensors(
            name,
            int(a_weight.shape[1]),
            int(b_weight.shape[0]),
            candidates,
            weights,
            base_rank,
        )
        state[name] = {"fixed_a": a, "fixed_b": b}
        modules.append({"module": name, "rank": expected_rank, "in_features": int(a_weight.shape[1]), "out_features": int(b_weight.shape[0])})
    return state, {
        "base_rank": base_rank,
        "aggregate_rank": expected_rank,
        "weight_mode": weight_mode,
        "elites": [
            {"candidate": row["candidate"], "score": float(row.get("exact_mean", 0.0)), "weight": weight}
            for row, weight in zip(rows, weights)
        ],
        "modules": modules,
    }


def run_eval(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out, ["per_prompt.jsonl", "holdout_per_prompt.jsonl"])
    elite_rows = top_candidate_rows(Path(args.source_run), args.top_k)
    if not elite_rows:
        raise ValueError(f"no elite candidates found in {args.source_run}")

    args.rank = args.base_rank * len(elite_rows)
    backend = make_backend(args)
    state, aggregate_summary = build_aggregate_state(backend.model, elite_rows, args.base_rank, args.weight_mode)

    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    base_screen = evaluate_candidate(backend, None, screen)
    base_holdout = evaluate_candidate(backend, None, holdout)
    write_jsonl(out / "per_prompt.jsonl", tag_rows(base_screen["rows"], mode="base_screen"))
    write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(base_holdout["rows"], mode="base_holdout"))

    start = time.time()
    aggregate_candidate = Candidate("elite_aggregate_lora", args.seed, 1.0, 1)
    screen_eval = evaluate_candidate(backend, aggregate_candidate, screen, state)
    holdout_eval = evaluate_candidate(backend, aggregate_candidate, holdout, state)
    write_jsonl(out / "per_prompt.jsonl", tag_rows(screen_eval["rows"], mode="screen"))
    write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(holdout_eval["rows"], mode="holdout"))

    summary = {
        "kind": "elite_aggregate_lora_eval",
        "source_run": args.source_run,
        "top_k": len(elite_rows),
        "base_rank": args.base_rank,
        "rank": args.rank,
        "weight_mode": args.weight_mode,
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "screen_unique_prompts": unique_example_count(screen),
        "holdout_unique_prompts": unique_example_count(holdout),
        "screen_unique_semantic_prompts": unique_semantic_example_count(screen),
        "holdout_unique_semantic_prompts": unique_semantic_example_count(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "base_screen_exact": base_screen["exact_mean"],
        "base_holdout_exact": base_holdout["exact_mean"],
        "aggregate_screen": {key: value for key, value in screen_eval.items() if key != "rows"},
        "aggregate_holdout": {key: value for key, value in holdout_eval.items() if key != "rows"},
        "holdout_lift": holdout_eval["exact_mean"] - base_holdout["exact_mean"],
        "elapsed_s": time.time() - start,
        "aggregate": aggregate_summary,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a serveable aggregate of top LoRA perturbations.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default=None)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--holdout-prompts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--base-rank", type=int, default=8)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--weight-mode", choices=["uniform", "score", "centered"], default="score")
    parser.add_argument("--targets", default="q_proj,v_proj")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--stop-at-answer", action="store_true")
    parser.add_argument("--allow-repeat-data", action="store_true")
    parser.add_argument("--perturbation-backend", choices=["lora"], default="lora")
    args = parser.parse_args(argv)
    run_eval(args)


if __name__ == "__main__":
    main()

