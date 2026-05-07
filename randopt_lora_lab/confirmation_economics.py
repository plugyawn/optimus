from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean

from .compare_backends import read_jsonl


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def parse_ks(text: str) -> list[int]:
    return sorted({int(part) for part in text.split(",") if part.strip()})


def candidate_key(row: dict) -> str:
    return str(row["candidate"])


def score(row: dict, score_col: str) -> float:
    if score_col in row:
        return float(row[score_col])
    if "exact_mean" in row:
        return float(row["exact_mean"])
    raise ValueError(f"row for {row.get('candidate')} missing score column {score_col!r} and exact_mean")


def elapsed(row: dict) -> float:
    return float(row.get("elapsed_s", 0.0))


def sorted_candidates(rows: list[dict], score_col: str) -> list[dict]:
    return sorted(rows, key=lambda row: (score(row, score_col), candidate_key(row)), reverse=True)


def estimate_full_screen_s(summary: dict, rows: list[dict]) -> float:
    candidate_sec = summary.get("candidate_sec")
    population = summary.get("population") or len(rows)
    if candidate_sec:
        return float(population) / max(float(candidate_sec), 1e-9)
    elapsed_values = [elapsed(row) for row in rows]
    return sum(elapsed_values)


def proposal_eval_s(summary: dict, rows: list[dict]) -> float:
    value = summary.get("eval_elapsed_s")
    if value is not None:
        return float(value)
    screen_candidate_sec = summary.get("screen_candidate_sec") or summary.get("candidate_sec")
    population = summary.get("population") or len(rows)
    if screen_candidate_sec:
        return float(population) / max(float(screen_candidate_sec), 1e-9)
    return sum(elapsed(row) for row in rows)


def analyze(
    trusted_dir: Path,
    proposal_dir: Path,
    *,
    ks: list[int],
    trusted_score_col: str = "exact_mean",
    proposal_score_col: str = "exact_mean",
) -> tuple[list[dict], dict]:
    trusted_rows = read_jsonl(trusted_dir / "candidate_summary.jsonl")
    proposal_rows = read_jsonl(proposal_dir / "candidate_summary.jsonl")
    if not trusted_rows:
        raise FileNotFoundError(f"missing trusted candidate rows in {trusted_dir}")
    if not proposal_rows:
        raise FileNotFoundError(f"missing proposal candidate rows in {proposal_dir}")

    trusted_summary = read_json(trusted_dir / "summary.json")
    proposal_summary = read_json(proposal_dir / "summary.json")
    trusted_by_candidate = {candidate_key(row): row for row in trusted_rows}
    proposal_ranked = [row for row in sorted_candidates(proposal_rows, proposal_score_col) if candidate_key(row) in trusted_by_candidate]
    trusted_ranked = sorted_candidates(trusted_rows, trusted_score_col)
    trusted_best = trusted_ranked[0]
    trusted_best_key = candidate_key(trusted_best)
    trusted_best_score = score(trusted_best, trusted_score_col)
    trusted_top_sets = {
        k: {candidate_key(row) for row in trusted_ranked[: min(k, len(trusted_ranked))]}
        for k in ks
    }

    full_s = estimate_full_screen_s(trusted_summary, trusted_rows)
    proposal_s = proposal_eval_s(proposal_summary, proposal_rows)
    proposal_load_build_s = float(proposal_summary.get("load_s", 0.0) or 0.0) + float(proposal_summary.get("adapter_build_s", 0.0) or 0.0)

    rows = []
    for k in ks:
        selected = proposal_ranked[: min(k, len(proposal_ranked))]
        selected_keys = [candidate_key(row) for row in selected]
        selected_trusted = [trusted_by_candidate[key] for key in selected_keys]
        confirmed = max(selected_trusted, key=lambda row: (score(row, trusted_score_col), candidate_key(row))) if selected_trusted else {}
        confirm_s = sum(elapsed(row) for row in selected_trusted)
        eval_total_s = proposal_s + confirm_s
        full_total_s = proposal_s + proposal_load_build_s + confirm_s
        confirmed_score = score(confirmed, trusted_score_col) if confirmed else 0.0
        row = {
            "k": k,
            "proposal_candidates": len(selected_keys),
            "contains_trusted_best": trusted_best_key in selected_keys,
            "trusted_topk_overlap": len(set(selected_keys) & trusted_top_sets[k]),
            "trusted_topk_possible": min(k, len(trusted_ranked)),
            "confirmed_candidate": candidate_key(confirmed) if confirmed else "",
            "confirmed_trusted_score": confirmed_score,
            "trusted_best_candidate": trusted_best_key,
            "trusted_best_score": trusted_best_score,
            "regret_vs_trusted_best": trusted_best_score - confirmed_score,
            "trusted_full_screen_s": full_s,
            "proposal_screen_s": proposal_s,
            "peft_confirm_s": confirm_s,
            "proposal_plus_confirm_s": eval_total_s,
            "proposal_full_plus_confirm_s": full_total_s,
            "eval_only_speedup_vs_trusted_full": full_s / max(eval_total_s, 1e-9),
            "full_without_peft_load_speedup_vs_trusted_full": full_s / max(full_total_s, 1e-9),
        }
        rows.append(row)

    summary = {
        "kind": "confirmation_economics",
        "trusted_dir": str(trusted_dir),
        "proposal_dir": str(proposal_dir),
        "trusted_candidates": len(trusted_rows),
        "proposal_candidates": len(proposal_rows),
        "trusted_best_candidate": trusted_best_key,
        "trusted_best_score": trusted_best_score,
        "trusted_full_screen_s": full_s,
        "proposal_screen_s": proposal_s,
        "proposal_load_build_s": proposal_load_build_s,
        "best_recovered_k": min((row["k"] for row in rows if row["contains_trusted_best"]), default=None),
        "zero_regret_k": min((row["k"] for row in rows if row["regret_vs_trusted_best"] == 0.0), default=None),
        "rows": rows,
    }
    return rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        values = []
        for col in columns:
            value = row.get(col, "")
            if isinstance(value, float):
                value = f"{value:.6g}"
            values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_report(path: Path, rows: list[dict], summary: dict) -> None:
    lines = [
        "# Confirmation Economics",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| trusted candidates | {summary['trusted_candidates']} |",
        f"| proposal candidates | {summary['proposal_candidates']} |",
        f"| trusted full screen seconds | {summary['trusted_full_screen_s']:.6g} |",
        f"| proposal screen seconds | {summary['proposal_screen_s']:.6g} |",
        f"| proposal load/build seconds | {summary['proposal_load_build_s']:.6g} |",
        f"| trusted best candidate | `{summary['trusted_best_candidate']}` |",
        f"| trusted best score | {summary['trusted_best_score']:.6g} |",
        f"| best recovered at k | {summary['best_recovered_k']} |",
        f"| zero-regret k | {summary['zero_regret_k']} |",
        "",
        "## Top-K Confirmation",
        "",
        markdown_table(
            rows,
            [
                "k",
                "contains_trusted_best",
                "trusted_topk_overlap",
                "trusted_topk_possible",
                "confirmed_trusted_score",
                "regret_vs_trusted_best",
                "peft_confirm_s",
                "proposal_plus_confirm_s",
                "eval_only_speedup_vs_trusted_full",
                "full_without_peft_load_speedup_vs_trusted_full",
            ],
        ),
        "",
        "The full-without-PEFT-load estimate includes vLLM load and adapter build time but not a separate PEFT model load.",
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Estimate vLLM proposal plus PEFT confirmation speed/quality tradeoffs.")
    p.add_argument("--trusted", required=True, type=Path)
    p.add_argument("--proposal", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ks", default="1,2,4,8,16,32")
    p.add_argument("--trusted-score-col", default="exact_mean")
    p.add_argument("--proposal-score-col", default="exact_mean")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    rows, summary = analyze(
        args.trusted,
        args.proposal,
        ks=parse_ks(args.ks),
        trusted_score_col=args.trusted_score_col,
        proposal_score_col=args.proposal_score_col,
    )
    write_csv(args.out / "rows.csv", rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.out / "report.md", rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
