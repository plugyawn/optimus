from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .backends import TransformersLoraBackend
from .countdown import load_examples, prompts as make_prompts, score_completion
from .lora_space import Candidate


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def evaluate_candidate(backend, candidate: Candidate | None, examples, family_state=None) -> dict:
    ps = make_prompts(examples)
    if candidate is None:
        backend.clear_candidate()
        key = "base"
    else:
        backend.set_candidate(candidate, family_state)
        key = candidate.key
    result = backend.generate(ps)
    rows = []
    exact = []
    malformed = []
    for ex, text in zip(examples, result.texts):
        score = score_completion(text, ex)
        exact.append(score["exact"])
        malformed.append(float(score["malformed"]))
        rows.append(
            {
                "candidate": key,
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "text": text,
                **score,
            }
        )
    return {
        "candidate": key,
        "exact_mean": float(np.mean(exact)),
        "malformed_mean": float(np.mean(malformed)),
        "output_tokens": result.output_tokens,
        "elapsed_s": result.elapsed_s,
        "rows": rows,
    }


def make_backend(args):
    return TransformersLoraBackend(
        args.model,
        rank=args.rank,
        target_suffixes=tuple(args.targets.split(",")),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        dtype=args.dtype,
    )


def run_oracle(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args)
    examples = load_examples(args.data, args.prompts, args.seed)
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
        "rank": args.rank,
        "targets": args.targets,
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


def run_search(args):
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    backend = make_backend(args)
    screen = load_examples(args.data, args.prompts, args.seed)
    holdout = load_examples(args.data, args.holdout_prompts, args.seed + 999)
    base_screen = evaluate_candidate(backend, None, screen)
    base_holdout = evaluate_candidate(backend, None, holdout)
    family_state = None
    if args.family == "anzo":
        anchors = [
            "Explain why the sky looks blue in one sentence.",
            "Write a short Python function that reverses a list.",
            "Summarize the benefits of unit tests.",
            "Draft a polite email declining a meeting.",
            "Explain what photosynthesis does.",
            "Give concise debugging advice for an import error.",
            "Compare quicksort and mergesort briefly.",
            "Extract names and dates from a short paragraph.",
        ]
        family_state = backend.build_anzo_state(make_prompts(screen[: min(16, len(screen))]), anchors)
    candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic)
    summaries = []
    start = time.time()
    for i, cand in enumerate(candidates):
        ev = evaluate_candidate(backend, cand, screen, family_state)
        summaries.append({k: v for k, v in ev.items() if k != "rows"})
        write_jsonl(out / "per_prompt.jsonl", ev["rows"])
        print(f"{i+1}/{len(candidates)} {cand.key} exact={ev['exact_mean']:.4f} elapsed={ev['elapsed_s']:.2f}", flush=True)
    top = sorted(summaries, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(summaries))]
    holdout_rows = []
    for row in top:
        parts = row["candidate"].split(":")
        family = parts[0]
        seed = int(parts[1].removeprefix("seed"))
        sigma = float(parts[2].removeprefix("s"))
        sign = int(parts[3].removeprefix("sign"))
        ev = evaluate_candidate(backend, Candidate(family, seed, sigma, sign), holdout, family_state)
        holdout_rows.append({k: v for k, v in ev.items() if k != "rows"})
        write_jsonl(out / "holdout_per_prompt.jsonl", ev["rows"])
    total_s = time.time() - start
    summary = {
        "kind": "search",
        "family": args.family,
        "population": len(candidates),
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "base_screen_exact": base_screen["exact_mean"],
        "base_holdout_exact": base_holdout["exact_mean"],
        "candidate_sec": len(candidates) / total_s,
        "pair_sec": len(candidates) * len(screen) / total_s,
        "top_screen": top,
        "top_holdout": holdout_rows,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_jsonl(out / "candidate_summary.jsonl", summaries)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ["oracle", "search"]:
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
        sp.add_argument("--family", default="isotropic", choices=["isotropic", "anzo"])
        sp.add_argument("--population", type=int, default=32)
        sp.add_argument("--promote", type=int, default=4)
        sp.add_argument("--antithetic", action="store_true")
    args = p.parse_args()
    if args.cmd == "oracle":
        run_oracle(args)
    else:
        run_search(args)


if __name__ == "__main__":
    main()
