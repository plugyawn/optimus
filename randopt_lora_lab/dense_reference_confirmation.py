from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .compare_backends import read_jsonl
from .confirmation_economics import (
    estimate_full_screen_s,
    proposal_eval_s,
    read_json,
    score,
)
from .parity_report import candidate_spec_key


def parse_ks(text: str) -> list[int]:
    return sorted({int(part) for part in text.split(",") if part.strip()})


def sorted_rows(rows: list[dict], score_col: str) -> list[dict]:
    return sorted(rows, key=lambda row: (score(row, score_col), str(row["candidate"])), reverse=True)


def row_by_spec(rows: list[dict]) -> dict[str, dict]:
    return {candidate_spec_key(str(row["candidate"])): row for row in rows}


def elapsed(row: dict) -> float:
    return float(row.get("elapsed_s", 0.0) or 0.0)


def analyze(
    dense_dir: Path,
    trusted_spectral_dir: Path,
    proposal_dir: Path,
    *,
    ks: list[int],
    dense_score_col: str = "exact_mean",
    spectral_score_col: str = "exact_mean",
    proposal_score_col: str = "exact_mean",
) -> tuple[list[dict], dict]:
    dense_rows = read_jsonl(dense_dir / "candidate_summary.jsonl")
    spectral_rows = read_jsonl(trusted_spectral_dir / "candidate_summary.jsonl")
    proposal_rows = read_jsonl(proposal_dir / "candidate_summary.jsonl")
    if not dense_rows:
        raise FileNotFoundError(f"missing dense candidate rows in {dense_dir}")
    if not spectral_rows:
        raise FileNotFoundError(f"missing spectral candidate rows in {trusted_spectral_dir}")
    if not proposal_rows:
        raise FileNotFoundError(f"missing proposal candidate rows in {proposal_dir}")

    dense_summary = read_json(dense_dir / "summary.json")
    spectral_summary = read_json(trusted_spectral_dir / "summary.json")
    proposal_summary = read_json(proposal_dir / "summary.json")
    dense_by_key = row_by_spec(dense_rows)
    spectral_by_key = row_by_spec(spectral_rows)
    shared_keys = sorted(set(dense_by_key) & set(spectral_by_key))
    proposal_ranked = [
        row
        for row in sorted_rows(proposal_rows, proposal_score_col)
        if candidate_spec_key(str(row["candidate"])) in shared_keys
    ]
    if not shared_keys:
        raise ValueError("dense and spectral trusted runs share no candidate specs")
    if not proposal_ranked:
        raise ValueError("proposal run shares no candidate specs with dense/spectral trusted runs")

    dense_ranked_keys = sorted(shared_keys, key=lambda key: (score(dense_by_key[key], dense_score_col), key), reverse=True)
    spectral_ranked_keys = sorted(shared_keys, key=lambda key: (score(spectral_by_key[key], spectral_score_col), key), reverse=True)
    dense_best_key = dense_ranked_keys[0]
    spectral_best_key = spectral_ranked_keys[0]
    dense_best_score = score(dense_by_key[dense_best_key], dense_score_col)
    spectral_best_score = score(spectral_by_key[spectral_best_key], spectral_score_col)
    dense_top_sets = {k: set(dense_ranked_keys[: min(k, len(dense_ranked_keys))]) for k in ks}

    dense_full_s = estimate_full_screen_s(dense_summary, dense_rows)
    proposal_s = proposal_eval_s(proposal_summary, proposal_rows)
    proposal_load_build_s = float(proposal_summary.get("load_s", 0.0) or 0.0) + float(
        proposal_summary.get("adapter_build_s", 0.0) or 0.0
    )

    rows = []
    for k in ks:
        selected = proposal_ranked[: min(k, len(proposal_ranked))]
        selected_keys = [candidate_spec_key(str(row["candidate"])) for row in selected]
        confirmed_key = max(selected_keys, key=lambda key: (score(spectral_by_key[key], spectral_score_col), key)) if selected_keys else ""
        confirm_s = sum(elapsed(spectral_by_key[key]) for key in selected_keys)
        dense_confirmed_score = score(dense_by_key[confirmed_key], dense_score_col) if confirmed_key else 0.0
        spectral_confirmed_score = score(spectral_by_key[confirmed_key], spectral_score_col) if confirmed_key else 0.0
        full_total_s = proposal_s + proposal_load_build_s + confirm_s
        eval_total_s = proposal_s + confirm_s
        rows.append(
            {
                "k": k,
                "proposal_candidates": len(selected_keys),
                "confirmed_spec": confirmed_key,
                "confirmed_spectral_candidate": spectral_by_key[confirmed_key]["candidate"] if confirmed_key else "",
                "contains_dense_best": dense_best_key in selected_keys,
                "contains_spectral_best": spectral_best_key in selected_keys,
                "dense_topk_overlap": len(set(selected_keys) & dense_top_sets[k]),
                "dense_topk_possible": min(k, len(dense_ranked_keys)),
                "dense_best_spec": dense_best_key,
                "dense_best_score": dense_best_score,
                "dense_score_at_confirmed": dense_confirmed_score,
                "dense_regret_vs_best": dense_best_score - dense_confirmed_score,
                "spectral_best_spec": spectral_best_key,
                "spectral_best_score": spectral_best_score,
                "spectral_confirmed_score": spectral_confirmed_score,
                "spectral_regret_vs_best": spectral_best_score - spectral_confirmed_score,
                "dense_full_screen_s": dense_full_s,
                "proposal_screen_s": proposal_s,
                "spectral_confirm_s": confirm_s,
                "proposal_plus_confirm_s": eval_total_s,
                "proposal_full_plus_confirm_s": full_total_s,
                "eval_only_speedup_vs_dense_full": dense_full_s / max(eval_total_s, 1e-9),
                "full_without_dense_load_speedup_vs_dense_full": dense_full_s / max(full_total_s, 1e-9),
            }
        )

    zero_dense_regret_k = min((row["k"] for row in rows if row["dense_regret_vs_best"] <= 0.0), default=None)
    dense_best_recovered_k = min((row["k"] for row in rows if row["contains_dense_best"]), default=None)
    summary = {
        "kind": "dense_reference_confirmation",
        "dense_dir": str(dense_dir),
        "trusted_spectral_dir": str(trusted_spectral_dir),
        "proposal_dir": str(proposal_dir),
        "dense_candidates": len(dense_rows),
        "trusted_spectral_candidates": len(spectral_rows),
        "proposal_candidates": len(proposal_rows),
        "shared_candidates": len(shared_keys),
        "dense_best_spec": dense_best_key,
        "dense_best_candidate": dense_by_key[dense_best_key]["candidate"],
        "dense_best_score": dense_best_score,
        "spectral_best_spec": spectral_best_key,
        "spectral_best_candidate": spectral_by_key[spectral_best_key]["candidate"],
        "spectral_best_score": spectral_best_score,
        "dense_full_screen_s": dense_full_s,
        "proposal_screen_s": proposal_s,
        "proposal_load_build_s": proposal_load_build_s,
        "zero_dense_regret_k": zero_dense_regret_k,
        "dense_best_recovered_k": dense_best_recovered_k,
        "rows": rows,
    }
    return rows, summary


def first_row_at_or_below(rows: list[dict], k: int | None) -> dict | None:
    if k is None:
        return None
    eligible = [row for row in rows if int(row["k"]) >= int(k)]
    return min(eligible, key=lambda row: int(row["k"])) if eligible else None


def gate(
    rows: list[dict],
    summary: dict,
    *,
    max_confirm_k: int,
    max_dense_regret: float = 0.0,
    min_full_without_dense_load_speedup: float = 1.0,
) -> dict:
    zero_k = summary.get("zero_dense_regret_k")
    zero_row = first_row_at_or_below(rows, zero_k)
    checks = [
        {
            "check": "zero_dense_regret_within_k",
            "passed": zero_k is not None and int(zero_k) <= max_confirm_k,
            "detail": {"zero_dense_regret_k": zero_k, "max_confirm_k": max_confirm_k},
        },
        {
            "check": "dense_regret_threshold",
            "passed": zero_row is not None and float(zero_row["dense_regret_vs_best"]) <= max_dense_regret,
            "detail": {
                "k": None if zero_row is None else zero_row["k"],
                "dense_regret": None if zero_row is None else zero_row["dense_regret_vs_best"],
                "max_dense_regret": max_dense_regret,
            },
        },
        {
            "check": "positive_full_speedup_vs_dense",
            "passed": zero_row is not None
            and float(zero_row["full_without_dense_load_speedup_vs_dense_full"]) >= min_full_without_dense_load_speedup,
            "detail": {
                "k": None if zero_row is None else zero_row["k"],
                "speedup": None if zero_row is None else zero_row["full_without_dense_load_speedup_vs_dense_full"],
                "min_speedup": min_full_without_dense_load_speedup,
            },
        },
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    return {
        "pass": not failed,
        "failed": failed,
        "checks": checks,
        "thresholds": {
            "max_confirm_k": max_confirm_k,
            "max_dense_regret": max_dense_regret,
            "min_full_without_dense_load_speedup": min_full_without_dense_load_speedup,
        },
    }


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def render_report(summary: dict) -> str:
    gate_payload = summary.get("gate", {})
    lines = [
        "# Dense-Referenced Confirmation",
        "",
        f"Gate: **{'PASS' if gate_payload.get('pass') else 'FAIL'}**",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| shared candidates | {summary['shared_candidates']} |",
        f"| dense best | `{summary['dense_best_candidate']}` |",
        f"| dense best score | {summary['dense_best_score']:.6g} |",
        f"| spectral best | `{summary['spectral_best_candidate']}` |",
        f"| spectral best score | {summary['spectral_best_score']:.6g} |",
        f"| zero dense-regret k | {summary['zero_dense_regret_k']} |",
        f"| dense best recovered k | {summary['dense_best_recovered_k']} |",
        "",
        "## Gate",
        "",
        "| check | pass | detail |",
        "| --- | ---: | --- |",
    ]
    for check in gate_payload.get("checks", []):
        lines.append(f"| {check['check']} | {str(check['passed']).lower()} | `{json.dumps(check['detail'], sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Top-K",
            "",
            "| k | confirmed spec | dense score | dense regret | spectral score | spectral regret | dense top-k overlap | full speedup |",
            "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            f"| {row['k']} | `{row['confirmed_spec']}` | {row['dense_score_at_confirmed']:.6g} | "
            f"{row['dense_regret_vs_best']:.6g} | {row['spectral_confirmed_score']:.6g} | "
            f"{row['spectral_regret_vs_best']:.6g} | {row['dense_topk_overlap']}/{row['dense_topk_possible']} | "
            f"{row['full_without_dense_load_speedup_vs_dense_full']:.6g} |"
        )
    lines.extend(
        [
            "",
            "This is a dense-reference screen-score gate. It does not replace matched holdout or drift parity.",
            "",
        ]
    )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check whether vLLM spectral shortlist plus PEFT confirmation recovers dense Gaussian quality.")
    parser.add_argument("--dense", type=Path, required=True)
    parser.add_argument("--trusted-spectral", type=Path, required=True)
    parser.add_argument("--proposal", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--ks", default="1,2,4,8,16,32")
    parser.add_argument("--dense-score-col", default="exact_mean")
    parser.add_argument("--spectral-score-col", default="exact_mean")
    parser.add_argument("--proposal-score-col", default="exact_mean")
    parser.add_argument("--max-confirm-k", type=int, default=8)
    parser.add_argument("--max-dense-regret", type=float, default=0.0)
    parser.add_argument("--min-full-without-dense-load-speedup", type=float, default=1.0)
    args = parser.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, summary = analyze(
        args.dense,
        args.trusted_spectral,
        args.proposal,
        ks=parse_ks(args.ks),
        dense_score_col=args.dense_score_col,
        spectral_score_col=args.spectral_score_col,
        proposal_score_col=args.proposal_score_col,
    )
    summary["gate"] = gate(
        rows,
        summary,
        max_confirm_k=args.max_confirm_k,
        max_dense_regret=args.max_dense_regret,
        min_full_without_dense_load_speedup=args.min_full_without_dense_load_speedup,
    )
    write_csv(args.out / "rows.csv", rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
