from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .subspace_audit import (
    CandidateKey,
    candidate_sketch,
    expand_paths,
    load_scored_candidates,
    mean_direction_predict,
    power_components,
    standardize_by_train,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def candidate_pool(
    *,
    family: str,
    population: int,
    sigma_values: list[float],
    seed: int,
    antithetic: bool,
) -> list[CandidateKey]:
    rng = np.random.default_rng(seed)
    base_n = population if not antithetic else max(1, (population + 1) // 2)
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=base_n)]
    sigmas = [float(x) for x in rng.choice(sigma_values, size=base_n, replace=True)]
    out = []
    for candidate_seed, sigma in zip(seeds, sigmas):
        out.append(CandidateKey(family, candidate_seed, sigma, 1))
        if antithetic:
            out.append(CandidateKey(family, candidate_seed, sigma, -1))
    return out[:population]


def candidate_matrix(candidates: list[CandidateKey], sketch_dim: int, *, feature_scale: str) -> np.ndarray:
    return np.stack([candidate_sketch(candidate, sketch_dim, feature_scale=feature_scale) for candidate in candidates], axis=0)


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def run(args) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prior_rows = load_scored_candidates(
        expand_paths(args.prior_runs),
        max_cap_hit=args.max_cap_hit,
        max_malformed=args.max_malformed,
    )
    if args.prior_family:
        families = {x.strip() for x in args.prior_family.split(",") if x.strip()}
        prior_rows = [row for row in prior_rows if row.family in families]
    if len(prior_rows) < 8:
        raise ValueError(f"need at least 8 prior rows after filtering, got {len(prior_rows)}")

    x_train_raw = np.stack(
        [
            candidate_sketch(
                CandidateKey(row.family, row.seed, row.sigma, row.sign),
                args.sketch_dim,
                feature_scale=args.feature_scale,
            )
            for row in prior_rows
        ],
        axis=0,
    )
    y_train = np.asarray([row.score for row in prior_rows], dtype=np.float64)
    pool = candidate_pool(
        family=args.family,
        population=args.pool,
        sigma_values=parse_float_list(args.sigma_values) if args.sigma_values else [args.sigma],
        seed=args.seed,
        antithetic=args.antithetic,
    )
    x_pool_raw = candidate_matrix(pool, args.sketch_dim, feature_scale=args.feature_scale)
    x_train, x_pool = standardize_by_train(x_train_raw, x_pool_raw)

    mean_pred = mean_direction_predict(x_train, y_train, x_pool)
    comps = power_components(
        x_train,
        y_train,
        components=args.components,
        n_iter=args.power_iter,
        seed=args.seed + 999,
    )
    projected = x_pool @ comps.T
    energy = np.sum(projected * projected, axis=1)
    if args.score_mode == "mean_direction":
        score = mean_pred
    elif args.score_mode == "power_energy":
        score = energy
    elif args.score_mode == "hybrid":
        mean_z = (mean_pred - np.mean(mean_pred)) / max(float(np.std(mean_pred)), 1e-12)
        energy_z = (energy - np.mean(energy)) / max(float(np.std(energy)), 1e-12)
        score = energy_z + args.mean_weight * mean_z
    else:
        raise ValueError(args.score_mode)

    order = np.argsort(-score)[: max(1, min(args.keep, len(pool)))]
    rows = []
    for rank, idx in enumerate(order, start=1):
        candidate = pool[int(idx)]
        rows.append(
            {
                "rank": rank,
                "candidate": candidate.key,
                "proposal_score": float(score[int(idx)]),
                "power_energy": float(energy[int(idx)]),
                "mean_direction": float(mean_pred[int(idx)]),
                "family": candidate.family,
                "seed": candidate.seed,
                "sigma": candidate.sigma,
                "sign": candidate.sign,
            }
        )
    write_jsonl(out / "candidates.jsonl", rows)
    summary = {
        "kind": "subspace_proposal",
        "prior_runs": [str(path) for path in expand_paths(args.prior_runs)],
        "prior_rows": len(prior_rows),
        "prior_families": sorted({row.family for row in prior_rows}),
        "family": args.family,
        "pool": len(pool),
        "keep": len(rows),
        "sigma_values": parse_float_list(args.sigma_values) if args.sigma_values else [args.sigma],
        "antithetic": args.antithetic,
        "score_mode": args.score_mode,
        "feature_scale": args.feature_scale,
        "sketch_dim": args.sketch_dim,
        "components": args.components,
        "power_iter": args.power_iter,
        "candidate_file": str((out / "candidates.jsonl").resolve()),
        "top": rows[: min(16, len(rows))],
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    report = [
        "# Subspace Proposal",
        "",
        f"- Prior rows: `{len(prior_rows)}`",
        f"- Pool: `{len(pool)}`",
        f"- Kept: `{len(rows)}`",
        f"- Score mode: `{args.score_mode}`",
        f"- Candidate file: `{summary['candidate_file']}`",
        "",
        "Use with:",
        "",
        "```bash",
        f"python -m randopt_lora_lab.vllm_lora_search --candidate-file {summary['candidate_file']} ...",
        "```",
        "",
    ]
    (out / "report.md").write_text("\n".join(report))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate cheap subspace-prefiltered RandOpt candidate lists.")
    p.add_argument("--prior-runs", required=True, help="Comma-separated prior result dirs/files/globs.")
    p.add_argument("--out", required=True)
    p.add_argument("--prior-family", default="", help="Optional family filter for prior rows.")
    p.add_argument("--family", default="isotropic")
    p.add_argument("--pool", type=int, default=10000)
    p.add_argument("--keep", type=int, default=512)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--sigma-values", default="")
    p.add_argument("--antithetic", action="store_true")
    p.add_argument("--score-mode", choices=["power_energy", "mean_direction", "hybrid"], default="power_energy")
    p.add_argument("--feature-scale", choices=["sigma", "unit"], default="unit")
    p.add_argument("--mean-weight", type=float, default=0.25)
    p.add_argument("--sketch-dim", type=int, default=512)
    p.add_argument("--components", type=int, default=16)
    p.add_argument("--power-iter", type=int, default=2)
    p.add_argument("--max-cap-hit", type=float, default=0.05)
    p.add_argument("--max-malformed", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=1234)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    summary = run(args)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
