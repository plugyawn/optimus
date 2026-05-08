from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from .backends import TransformersDenseGaussianBackend, TransformersLoraBackend
from .countdown import (
    extract_numeric_vote,
    load_examples,
    score_completion,
    unique_example_count,
    unique_semantic_example_count,
    voted_answer_exact,
)
from .lora_space import Candidate
from .prompt_variants import make_variant_prompts


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


def make_prompts_for_backend(backend, args, examples) -> list[str]:
    return make_variant_prompts(
        examples,
        args.prompt_variant,
        tokenizer=getattr(backend, "tokenizer", None),
        use_chat_template=args.use_chat_template,
    )


def evaluate_candidate(backend, candidate: Candidate | None, examples, args, family_state=None) -> dict:
    ps = make_prompts_for_backend(backend, args, examples)
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
            dense_noise_mode=args.dense_noise_mode,
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
    sig0 = backend.logits_signature(make_prompts_for_backend(backend, args, examples[:4]))
    base = evaluate_candidate(backend, None, examples, args)
    zero = evaluate_candidate(backend, Candidate("isotropic", 0, 0.0), examples, args)
    random_candidate = Candidate("isotropic", args.seed + 1, args.sigma)
    rand = evaluate_candidate(backend, random_candidate, examples, args)
    backend.clear_candidate()
    sig1 = backend.logits_signature(make_prompts_for_backend(backend, args, examples[:4]))
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
        "dense_snapshot_device": args.dense_snapshot_device,
        "dense_noise_mode": args.dense_noise_mode,
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


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def candidate_panel(
    family: str,
    population: int,
    sigma: float,
    seed: int,
    antithetic: bool,
    sigma_values: list[float] | None = None,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    sigmas = sigma_values or [sigma]
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=population if not antithetic else population // 2)]
    sampled_sigmas = [float(x) for x in rng.choice(sigmas, size=len(seeds), replace=True)]
    out = []
    for s, sampled_sigma in zip(seeds, sampled_sigmas):
        out.append(Candidate(family, s, sampled_sigma, 1))
        if antithetic:
            out.append(Candidate(family, s, sampled_sigma, -1))
    return out[:population]


def parse_candidate_key(key: str) -> Candidate:
    parts = key.split(":")
    return Candidate(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


def read_candidate_file(path: str) -> list[Candidate]:
    candidates = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = line
            key = item.get("candidate") if isinstance(item, dict) else str(item)
            if not key:
                raise ValueError(f"{path}:{line_no} missing candidate")
            candidates.append(parse_candidate_key(str(key)))
    return candidates


def parse_k_list(text: str) -> list[int]:
    return sorted({int(x) for x in text.split(",") if x.strip()})


def parse_ratio_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def ensemble_ks_from_values(population: int, k_text: str = "", ratio_text: str = "") -> list[int]:
    ks = set(parse_k_list(k_text) if k_text else [])
    for ratio in parse_ratio_list(ratio_text) if ratio_text else []:
        ks.add(max(1, int(float(population) * ratio)))
    return sorted(ks)


def rows_by_candidate_and_example(rows: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for row in rows:
        out.setdefault(str(row["candidate"]), {})[int(row["example_id"])] = row
    return out


def majority_vote_evaluation(candidate_order: list[str], rows: list[dict], examples, k_values: list[int]) -> tuple[list[dict], list[dict]]:
    by_candidate = rows_by_candidate_and_example(rows)
    result_rows = []
    per_prompt_rows = []
    for k in k_values:
        active = candidate_order[: min(k, len(candidate_order))]
        exact_values = []
        coverage_values = []
        valid_vote_counts = []
        for ex in examples:
            votes = []
            rejects = Counter()
            for candidate in active:
                row = by_candidate.get(candidate, {}).get(ex.id)
                if not row:
                    continue
                vote = extract_numeric_vote(str(row.get("text", "")), ex)
                if vote["valid_vote"]:
                    votes.append(str(vote["vote"]))
                else:
                    rejects[str(vote["vote_reject"])] += 1
            counter = Counter(votes)
            final_vote = counter.most_common(1)[0][0] if counter else ""
            exact = voted_answer_exact(final_vote, ex)
            exact_values.append(exact)
            coverage_values.append(float(bool(counter)))
            valid_vote_counts.append(len(votes))
            per_prompt_rows.append(
                {
                    "k": k,
                    "example_id": ex.id,
                    "numbers": list(ex.numbers),
                    "target": ex.target,
                    "final_vote": final_vote,
                    "exact": exact,
                    "valid_vote_count": len(votes),
                    "missing_vote_count": max(len(active) - len(votes), 0),
                    "vote_counts": dict(counter),
                    "reject_counts": dict(rejects),
                }
            )
        denom = max(len(examples), 1)
        result_rows.append(
            {
                "k": k,
                "evaluated_candidates": len(active),
                "exact_mean": float(np.mean(exact_values)) if exact_values else 0.0,
                "coverage_mean": float(np.mean(coverage_values)) if coverage_values else 0.0,
                "valid_votes_per_prompt": float(np.mean(valid_vote_counts)) if valid_vote_counts else 0.0,
                "correct": int(sum(exact_values)),
                "total": denom,
            }
        )
    return result_rows, per_prompt_rows


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
    if args.family in {
        "isotropic",
        "dense_gaussian",
        "factor_gaussian_lora",
        "projected_gaussian_rank_r",
        "randomized_projected_gaussian_rank_r",
    }:
        return None
    if args.family.startswith("spectral_projected_gaussian_rank_r"):
        return None
    if args.family.startswith("sparse_low_rank_lora"):
        return None
    if args.family == "random_ortho":
        return backend.build_random_orthonormal_state(args.seed)
    if args.family == "target_svd":
        return backend.build_anzo_state(
            make_prompts_for_backend(backend, args, screen[: min(16, len(screen))]),
            anzo_anchor_prompts(),
            subtract_anchor=False,
        )
    if args.family == "anzo_random_target":
        return backend.build_anzo_state(
            anzo_random_target_prompts(args.seed + 17, min(16, len(screen))),
            anzo_anchor_prompts(),
        )
    if args.family == "anzo":
        return backend.build_anzo_state(make_prompts_for_backend(backend, args, screen[: min(16, len(screen))]), anzo_anchor_prompts())
    raise ValueError(f"unsupported family: {args.family}")


def run_search(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out, ["per_prompt.jsonl", "candidate_summary.jsonl", "holdout_per_prompt.jsonl", "ensemble_per_prompt.jsonl"])
    backend = make_backend(args)
    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    base_screen = evaluate_candidate(backend, None, screen, args)
    base_holdout = evaluate_candidate(backend, None, holdout, args)
    write_jsonl(out / "per_prompt.jsonl", tag_rows(base_screen["rows"], mode="base_screen"))
    write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(base_holdout["rows"], mode="base_holdout"))
    family_state = maybe_build_family_state(args, backend, screen)
    sigma_values = parse_float_list(args.sigma_values) if args.sigma_values else [args.sigma]
    candidates = (
        read_candidate_file(args.candidate_file)
        if args.candidate_file
        else candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic, sigma_values)
    )
    summaries = []
    start = time.time()
    for i, cand in enumerate(candidates):
        ev = evaluate_candidate(backend, cand, screen, args, family_state)
        summaries.append({k: v for k, v in ev.items() if k != "rows"})
        write_jsonl(out / "per_prompt.jsonl", tag_rows(ev["rows"], mode="screen"))
        print(f"{i+1}/{len(candidates)} {cand.key} exact={ev['exact_mean']:.4f} elapsed={ev['elapsed_s']:.2f}", flush=True)
    ensemble_ks = ensemble_ks_from_values(len(candidates), args.ensemble_ks, args.ensemble_ratios)
    promote_n = max(args.promote, max(ensemble_ks, default=0))
    top = sorted(summaries, key=lambda r: r["exact_mean"], reverse=True)[: min(promote_n, len(summaries))]
    holdout_rows = []
    holdout_per_candidate_rows = []
    for row in top:
        ev = evaluate_candidate(backend, parse_candidate_key(row["candidate"]), holdout, args, family_state)
        holdout_rows.append({k: v for k, v in ev.items() if k != "rows"})
        tagged = tag_rows(ev["rows"], mode="holdout")
        holdout_per_candidate_rows.extend(tagged)
        write_jsonl(out / "holdout_per_prompt.jsonl", tagged)
    ensemble_rows = []
    if ensemble_ks:
        ensemble_rows, ensemble_per_prompt = majority_vote_evaluation(
            [str(row["candidate"]) for row in top],
            holdout_per_candidate_rows,
            holdout,
            ensemble_ks,
        )
        write_jsonl(out / "ensemble_per_prompt.jsonl", ensemble_per_prompt)
    total_s = time.time() - start
    summary = {
        "kind": "search",
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "population": len(candidates),
        "rank": args.rank,
        "sigma": args.sigma,
        "sigma_values": sigma_values,
        "prompt_variant": args.prompt_variant,
        "use_chat_template": args.use_chat_template,
        "candidate_score_metric": "exact_answer",
        "ensemble_vote_metric": "valid_numeric_majority_vote",
        "seed": args.seed,
        "targets": args.targets,
        "perturbation_backend": args.perturbation_backend,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "allow_repeat_data": args.allow_repeat_data,
        "candidate_file": args.candidate_file,
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
        "ensemble_ks": ensemble_ks,
        "ensemble_ratios": parse_ratio_list(args.ensemble_ratios) if args.ensemble_ratios else [],
        "ensemble_holdout": ensemble_rows,
        "best_ensemble_holdout_exact": max((row["exact_mean"] for row in ensemble_rows), default=None),
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
    base_stage = evaluate_candidate(backend, None, stage, args)
    base_screen = evaluate_candidate(backend, None, screen, args)
    base_holdout = evaluate_candidate(backend, None, holdout, args)
    write_jsonl(out / "stage_per_prompt.jsonl", tag_rows(base_stage["rows"], mode="base_stage"))
    write_jsonl(out / "full_per_prompt.jsonl", tag_rows(base_screen["rows"], mode="base_screen"))
    write_jsonl(out / "holdout_per_prompt.jsonl", tag_rows(base_holdout["rows"], mode="base_holdout"))
    family_state = maybe_build_family_state(args, backend, screen)
    sigma_values = parse_float_list(args.sigma_values) if args.sigma_values else [args.sigma]
    candidates = (
        read_candidate_file(args.candidate_file)
        if args.candidate_file
        else candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic, sigma_values)
    )
    stage_rows = []
    start = time.time()
    for i, cand in enumerate(candidates):
        ev = evaluate_candidate(backend, cand, stage, args, family_state)
        row = {k: v for k, v in ev.items() if k != "rows"}
        stage_rows.append(row)
        write_jsonl(out / "stage_per_prompt.jsonl", tag_rows(ev["rows"], mode="stage"))
        print(f"stage {i+1}/{len(candidates)} {cand.key} exact={ev['exact_mean']:.4f}", flush=True)
    survivor_n = min(args.survivors, len(stage_rows))
    survivors = sorted(stage_rows, key=lambda r: r["exact_mean"], reverse=True)[:survivor_n]
    full_rows = []
    for i, row in enumerate(survivors):
        cand = parse_candidate_key(row["candidate"])
        ev = evaluate_candidate(backend, cand, screen, args, family_state)
        full = {k: v for k, v in ev.items() if k != "rows"}
        full["stage_exact_mean"] = row["exact_mean"]
        full_rows.append(full)
        write_jsonl(out / "full_per_prompt.jsonl", tag_rows(ev["rows"], mode="screen"))
        print(f"full {i+1}/{len(survivors)} {cand.key} exact={ev['exact_mean']:.4f}", flush=True)
    top = sorted(full_rows, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(full_rows))]
    holdout_rows = []
    for row in top:
        ev = evaluate_candidate(backend, parse_candidate_key(row["candidate"]), holdout, args, family_state)
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
        "sigma_values": sigma_values,
        "seed": args.seed,
        "targets": args.targets,
        "candidate_score_metric": "exact_answer",
        "perturbation_backend": args.perturbation_backend,
        "dtype": args.dtype,
        "batch_size": args.batch_size,
        "allow_repeat_data": args.allow_repeat_data,
        "candidate_file": args.candidate_file,
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
                    ev = evaluate_candidate(backend, cand, subset, args)
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
        sp.add_argument("--sigma-values", default="")
        sp.add_argument("--targets", default="q_proj,v_proj")
        sp.add_argument("--max-new-tokens", type=int, default=32)
        sp.add_argument("--batch-size", type=int, default=16)
        sp.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
        sp.add_argument("--prompt-variant", default="default")
        sp.add_argument("--use-chat-template", action="store_true")
        sp.add_argument("--perturbation-backend", choices=["lora", "dense"], default="lora")
        sp.add_argument("--dense-snapshot-device", choices=["model", "cpu"], default="model")
        sp.add_argument("--dense-noise-mode", choices=["canonical", "paper"], default="canonical")
        sp.add_argument("--stop-at-answer", action="store_true")
        sp.add_argument(
            "--family",
            default="isotropic",
            choices=[
                "isotropic",
                "factor_gaussian_lora",
                "projected_gaussian_rank_r",
                "randomized_projected_gaussian_rank_r",
                "spectral_projected_gaussian_rank_r",
                "spectral_projected_gaussian_rank_r_c0p5",
                "spectral_projected_gaussian_rank_r_c0p75",
                "spectral_projected_gaussian_rank_r_c1p25",
                "spectral_projected_gaussian_rank_r_c1p5",
                "spectral_projected_gaussian_rank_r_c2",
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
        sp.add_argument("--candidate-file", default="", help="Optional JSONL/list of exact candidate keys to evaluate.")
        sp.add_argument("--promote", type=int, default=4)
        sp.add_argument("--ensemble-ks", default="")
        sp.add_argument("--ensemble-ratios", default="")
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
