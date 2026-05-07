from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .backends import TransformersDenseGaussianBackend, TransformersLoraBackend
from .countdown import (
    load_examples,
    prompts as make_prompts,
    score_completion,
    unique_example_count,
    unique_semantic_example_count,
)
from .lora_space import Candidate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def reset_outputs(out: Path, names: list[str]) -> None:
    for name in names:
        path = out / name
        if path.exists():
            path.unlink()


def tag_rows(rows: list[dict], **extra) -> list[dict]:
    return [dict(row, **extra) for row in rows]


def evaluate_candidate(backend, candidate: Candidate | None, examples, family_state=None) -> dict:
    ps = make_prompts(examples)
    mutation_start = time.time()
    if candidate is None:
        backend.clear_candidate()
        key = "base"
    else:
        backend.set_candidate(candidate, family_state)
        key = candidate.key
    mutation_s = time.time() - mutation_start
    result = backend.generate(ps)
    rows = []
    exact = []
    malformed = []
    cap_hits = []
    answer_closed = []
    for ex, text, output_tokens in zip(examples, result.texts, result.token_counts):
        score = score_completion(text, ex)
        exact.append(score["exact"])
        malformed.append(float(score["malformed"]))
        cap_hit = float(output_tokens >= backend.max_new_tokens)
        closed = float("</answer>" in text)
        cap_hits.append(cap_hit)
        answer_closed.append(closed)
        rows.append(
            {
                "candidate": key,
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "text": text,
                "output_tokens": output_tokens,
                "cap_hit": cap_hit,
                "answer_closed": closed,
                **score,
            }
        )
    return {
        "candidate": key,
        "exact_mean": float(np.mean(exact)),
        "malformed_mean": float(np.mean(malformed)),
        "cap_hit_mean": float(np.mean(cap_hits)),
        "answer_closed_mean": float(np.mean(answer_closed)),
        "output_tokens": result.output_tokens,
        "elapsed_s": result.elapsed_s,
        "mutation_s": mutation_s,
        "rows": rows,
    }


def make_backend(args):
    if args.perturbation_backend == "dense":
        return TransformersDenseGaussianBackend(
            args.model,
            target_suffixes=tuple(args.targets.split(",")),
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            dtype=args.dtype,
            stop_at_answer=args.stop_at_answer,
            snapshot_device=args.dense_snapshot_device,
        )
    return TransformersLoraBackend(
        args.model,
        rank=args.rank,
        target_suffixes=tuple(args.targets.split(",")),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype,
        stop_at_answer=args.stop_at_answer,
    )


def run_oracle(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out, ["probes.jsonl"])
    backend = make_backend(args)
    examples = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    sig0 = backend.logits_signature(make_prompts(examples[:4]))
    base = evaluate_candidate(backend, None, examples)
    zero = evaluate_candidate(backend, Candidate("isotropic", 0, 0.0), examples)
    random_candidate = Candidate("isotropic", args.seed + 1, args.sigma)
    rand = evaluate_candidate(backend, random_candidate, examples)
    backend.clear_candidate()
    sig1 = backend.logits_signature(make_prompts(examples[:4]))
    restore_max_abs = float((sig0 - sig1).abs().max().item())
    summary = {
        "kind": "oracle",
        "model": args.model,
        "data": args.data,
        "seed": args.seed,
        "rank": args.rank,
        "sigma": args.sigma,
        "targets": args.targets,
        "perturbation_backend": args.perturbation_backend,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "allow_repeat_data": args.allow_repeat_data,
        "base_exact": base["exact_mean"],
        "zero_exact": zero["exact_mean"],
        "random_exact": rand["exact_mean"],
        "restore_logits_max_abs": restore_max_abs,
        "base_elapsed_s": base["elapsed_s"],
        "random_elapsed_s": rand["elapsed_s"],
        "pass_zero_score": base["exact_mean"] == zero["exact_mean"],
        "pass_restore_logits": restore_max_abs == 0.0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "probes.jsonl", base["rows"] + zero["rows"] + rand["rows"])
    print(json.dumps(summary, indent=2, sort_keys=True))


def candidate_panel(family: str, population: int, sigma: float, seed: int, antithetic: bool) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=population if not antithetic else population // 2)]
    out = []
    for s in seeds:
        out.append(Candidate(family, s, sigma, 1))
        if antithetic:
            out.append(Candidate(family, s, sigma, -1))
    return out[:population]


def parse_candidate_key(key: str) -> Candidate:
    parts = key.split(":")
    return Candidate(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


def anzo_anchor_prompts() -> list[str]:
    return [
        "Explain why the sky looks blue in one sentence.",
        "Write a short Python function that reverses a list.",
        "Summarize the benefits of unit tests.",
        "Draft a polite email declining a meeting.",
        "Explain what photosynthesis does.",
        "Give concise debugging advice for an import error.",
        "Compare quicksort and mergesort briefly.",
        "Extract names and dates from a short paragraph.",
    ]


def anzo_random_target_prompts(seed: int, n: int) -> list[str]:
    rng = np.random.default_rng(seed)
    topics = [
        "weather",
        "databases",
        "calendar planning",
        "biology",
        "debugging",
        "sorting algorithms",
        "email",
        "unit tests",
        "maps",
        "music",
        "nutrition",
        "finance",
    ]
    prompts = []
    for idx in rng.integers(0, len(topics), size=n):
        prompts.append(f"Give a concise factual sentence about {topics[int(idx)]}.")
    return prompts


def maybe_build_family_state(args, backend, screen):
    if args.family in {"isotropic", "dense_gaussian", "factor_gaussian_lora", "projected_gaussian_rank_r"}:
        return None
    if args.family.startswith("sparse_low_rank_lora"):
        return None
    if args.family == "random_ortho":
        return backend.build_random_orthonormal_state(args.seed)
    if args.family == "target_svd":
        return backend.build_anzo_state(
            make_prompts(screen[: min(16, len(screen))]),
            anzo_anchor_prompts(),
            subtract_anchor=False,
        )
    if args.family == "anzo_random_target":
        return backend.build_anzo_state(
            anzo_random_target_prompts(args.seed + 17, min(16, len(screen))),
            anzo_anchor_prompts(),
        )
    if args.family == "anzo":
        return backend.build_anzo_state(make_prompts(screen[: min(16, len(screen))]), anzo_anchor_prompts())
    raise ValueError(f"unsupported family: {args.family}")


def run_search(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out, ["per_prompt.jsonl", "candidate_summary.jsonl", "holdout_per_prompt.jsonl"])
    backend = make_backend(args)
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
    family_state = maybe_build_family_state(args, backend, screen)
    candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic)
    summaries = []
    start = time.time()
    for i, cand in enumerate(candidates):
        ev = evaluate_candidate(backend, cand, screen, family_state)
        summaries.append({k: v for k, v in ev.items() if k != "rows"})
        write_jsonl(out / "per_prompt.jsonl", tag_rows(ev["rows"], mode="screen"))
        print(f"{i+1}/{len(candidates)} {cand.key} exact={ev['exact_mean']:.4f} elapsed={ev['elapsed_s']:.2f}", flush=True)
    top = sorted(summaries, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(summaries))]
    holdout_rows = []
    for row in top:
        ev = evaluate_candidate(backend, parse_candidate_key(row["candidate"]), holdout, family_state)
        holdout_rows.append({k: v for k, v in ev.items() if k != "rows"})
        write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(ev["rows"], mode="holdout"))
    total_s = time.time() - start
    summary = {
        "kind": "search",
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "population": len(candidates),
        "rank": args.rank,
        "sigma": args.sigma,
        "seed": args.seed,
        "targets": args.targets,
        "perturbation_backend": args.perturbation_backend,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "allow_repeat_data": args.allow_repeat_data,
        "antithetic": args.antithetic,
        "promote": args.promote,
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "screen_unique_prompts": unique_example_count(screen),
        "holdout_unique_prompts": unique_example_count(holdout),
        "screen_unique_semantic_prompts": unique_semantic_example_count(screen),
        "holdout_unique_semantic_prompts": unique_semantic_example_count(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "base_screen_exact": base_screen["exact_mean"],
        "base_holdout_exact": base_holdout["exact_mean"],
        "candidate_sec": len(candidates) / total_s,
        "pair_sec": len(candidates) * len(screen) / total_s,
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "top_screen": top,
        "top_holdout": holdout_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "candidate_summary.jsonl", summaries)
    print(json.dumps(summary, indent=2, sort_keys=True))


def run_halving(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(
        out,
        [
            "stage_per_prompt.jsonl",
            "full_per_prompt.jsonl",
            "stage_candidate_summary.jsonl",
            "candidate_summary.jsonl",
            "holdout_per_prompt.jsonl",
        ],
    )
    backend = make_backend(args)
    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    stage = screen[: min(args.stage_prompts, len(screen))]
    base_stage = evaluate_candidate(backend, None, stage)
    base_screen = evaluate_candidate(backend, None, screen)
    base_holdout = evaluate_candidate(backend, None, holdout)
    write_jsonl(out / "stage_per_prompt.jsonl", tag_rows(base_stage["rows"], mode="base_stage"))
    write_jsonl(out / "full_per_prompt.jsonl", tag_rows(base_screen["rows"], mode="base_screen"))
    write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(base_holdout["rows"], mode="base_holdout"))
    family_state = maybe_build_family_state(args, backend, screen)
    candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic)
    stage_rows = []
    start = time.time()
    for i, cand in enumerate(candidates):
        ev = evaluate_candidate(backend, cand, stage, family_state)
        row = {k: v for k, v in ev.items() if k != "rows"}
        stage_rows.append(row)
        write_jsonl(out / "stage_per_prompt.jsonl", tag_rows(ev["rows"], mode="stage"))
        print(f"stage {i+1}/{len(candidates)} {cand.key} exact={ev['exact_mean']:.4f}", flush=True)
    survivor_n = min(args.survivors, len(stage_rows))
    survivors = sorted(stage_rows, key=lambda r: r["exact_mean"], reverse=True)[:survivor_n]
    full_rows = []
    for i, row in enumerate(survivors):
        cand = parse_candidate_key(row["candidate"])
        ev = evaluate_candidate(backend, cand, screen, family_state)
        full = {k: v for k, v in ev.items() if k != "rows"}
        full["stage_exact_mean"] = row["exact_mean"]
        full_rows.append(full)
        write_jsonl(out / "full_per_prompt.jsonl", tag_rows(ev["rows"], mode="screen"))
        print(f"full {i+1}/{len(survivors)} {cand.key} exact={ev['exact_mean']:.4f}", flush=True)
    top = sorted(full_rows, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(full_rows))]
    holdout_rows = []
    for row in top:
        ev = evaluate_candidate(backend, parse_candidate_key(row["candidate"]), holdout, family_state)
        holdout_rows.append({k: v for k, v in ev.items() if k != "rows"})
        write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(ev["rows"], mode="holdout"))
    total_s = time.time() - start
    effective_prompt_evals = len(candidates) * len(stage) + len(survivors) * len(screen) + len(top) * len(holdout)
    full_prompt_evals = len(candidates) * len(screen) + len(top) * len(holdout)
    summary = {
        "kind": "halving",
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "population": len(candidates),
        "rank": args.rank,
        "sigma": args.sigma,
        "seed": args.seed,
        "targets": args.targets,
        "perturbation_backend": args.perturbation_backend,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "allow_repeat_data": args.allow_repeat_data,
        "antithetic": args.antithetic,
        "promote": args.promote,
        "stage_prompts": len(stage),
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "stage_unique_prompts": unique_example_count(stage),
        "screen_unique_prompts": unique_example_count(screen),
        "holdout_unique_prompts": unique_example_count(holdout),
        "stage_unique_semantic_prompts": unique_semantic_example_count(stage),
        "screen_unique_semantic_prompts": unique_semantic_example_count(screen),
        "holdout_unique_semantic_prompts": unique_semantic_example_count(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "survivors": len(survivors),
        "base_stage_exact": base_stage["exact_mean"],
        "base_screen_exact": base_screen["exact_mean"],
        "base_holdout_exact": base_holdout["exact_mean"],
        "candidate_sec": len(candidates) / total_s,
        "prompt_eval_sec": effective_prompt_evals / total_s,
        "effective_prompt_evals": effective_prompt_evals,
        "full_prompt_evals": full_prompt_evals,
        "prompt_eval_savings": 1.0 - (effective_prompt_evals / full_prompt_evals),
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "top_stage": sorted(stage_rows, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(stage_rows))],
        "top_screen": top,
        "top_holdout": holdout_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "stage_candidate_summary.jsonl", stage_rows)
    write_jsonl(out / "candidate_summary.jsonl", full_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def parse_int_list(text: str) -> list[int]:
    return [int(x) for x in text.split(",") if x.strip()]


def run_sysbench(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out, ["rows.jsonl"])
    backend = make_backend(args)
    examples = load_examples(
        args.data,
        max(parse_int_list(args.prompt_counts)),
        args.seed,
        allow_repeat=args.allow_repeat_data,
    )
    candidate = Candidate(args.family, args.seed + 17, args.sigma)
    rows = []
    for batch_size in parse_int_list(args.batch_sizes):
        backend.batch_size = batch_size
        for prompt_count in parse_int_list(args.prompt_counts):
            subset = examples[:prompt_count]
            for mode, cand in [("base", None), ("candidate", candidate)]:
                for rep in range(args.repeats):
                    ev = evaluate_candidate(backend, cand, subset)
                    row = {k: v for k, v in ev.items() if k != "rows"}
                    row.update(
                        {
                            "kind": "sysbench_row",
                            "mode": mode,
                            "batch_size": batch_size,
                            "prompt_count": prompt_count,
                            "rep": rep,
                            "tokens_per_sec": ev["output_tokens"] / max(ev["elapsed_s"], 1e-9),
                            "prompts_per_sec": prompt_count / max(ev["elapsed_s"], 1e-9),
                        }
                    )
                    rows.append(row)
                    print(
                        f"sysbench bs={batch_size} prompts={prompt_count} mode={mode} "
                        f"tok/s={row['tokens_per_sec']:.1f} exact={row['exact_mean']:.4f}",
                        flush=True,
                    )
    best = max(rows, key=lambda r: r["tokens_per_sec"]) if rows else {}
    summary = {
        "kind": "sysbench",
        "model": args.model,
        "family": args.family,
        "rank": args.rank,
        "targets": args.targets,
        "batch_sizes": parse_int_list(args.batch_sizes),
        "prompt_counts": parse_int_list(args.prompt_counts),
        "repeats": args.repeats,
        "best_tokens_per_sec": best.get("tokens_per_sec"),
        "best_prompts_per_sec": best.get("prompts_per_sec"),
        "best_batch_size": best.get("batch_size"),
        "best_prompt_count": best.get("prompt_count"),
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "rows": rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "rows.jsonl", rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["oracle", "search", "halving", "sysbench"]:
        sp = sub.add_parser(name)
        sp.add_argument("--out", required=True)
        sp.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
        sp.add_argument("--data", default=None)
        sp.add_argument("--prompts", type=int, default=32)
        sp.add_argument("--holdout-prompts", type=int, default=32)
        sp.add_argument("--seed", type=int, default=1234)
        sp.add_argument("--rank", type=int, default=8)
        sp.add_argument("--sigma", type=float, default=0.02)
        sp.add_argument("--targets", default="q_proj,v_proj")
        sp.add_argument("--max-new-tokens", type=int, default=32)
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
        sp.add_argument("--perturbation-backend", choices=["lora", "dense"], default="lora")
        sp.add_argument("--dense-snapshot-device", choices=["model", "cpu"], default="model")
        sp.add_argument("--stop-at-answer", action="store_true")
        sp.add_argument(
            "--family",
            default="isotropic",
            choices=[
                "isotropic",
                "factor_gaussian_lora",
                "projected_gaussian_rank_r",
                "sparse_low_rank_lora",
                "sparse_low_rank_lora_d0p125",
                "sparse_low_rank_lora_d0p25",
                "sparse_low_rank_lora_d0p5",
                "dense_gaussian",
                "anzo",
                "target_svd",
                "random_ortho",
                "anzo_random_target",
            ],
        )
        sp.add_argument("--population", type=int, default=32)
        sp.add_argument("--promote", type=int, default=4)
        sp.add_argument("--stage-prompts", type=int, default=8)
        sp.add_argument("--survivors", type=int, default=8)
        sp.add_argument("--batch-sizes", default="4,8,16,32")
        sp.add_argument("--prompt-counts", default="8,16,32")
        sp.add_argument("--repeats", type=int, default=2)
        sp.add_argument("--antithetic", action="store_true")
        sp.add_argument("--allow-repeat-data", action="store_true")
    args = p.parse_args()
    if args.perturbation_backend == "dense" and args.cmd != "oracle" and args.family != "dense_gaussian":
        raise ValueError("--perturbation-backend dense requires --family dense_gaussian")
    if args.perturbation_backend == "lora" and args.family == "dense_gaussian":
        raise ValueError("--family dense_gaussian requires --perturbation-backend dense")
    if args.cmd == "oracle":
        run_oracle(args)
    elif args.cmd == "search":
        run_search(args)
    elif args.cmd == "halving":
        run_halving(args)
    elif args.cmd == "sysbench":
        run_sysbench(args)
    else:
        raise ValueError(args.cmd)


if __name__ == "__main__":
    main()
