from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean

from .compare_backends import read_jsonl


def row_key(row: dict) -> tuple[str, str, str]:
    return (
        str(row.get("candidate")),
        str(row.get("example_id")),
        str(row.get("prompt_variant", "default")),
    )


def selected_rows(run_dir: Path, *, mode: str) -> list[dict]:
    rows = read_jsonl(run_dir / "per_prompt.jsonl")
    return [row for row in rows if row.get("mode") == mode]


def bool_float(value) -> float:
    return float(bool(value))


def compare_rows(trusted_dir: Path, candidate_dir: Path, *, mode: str, trusted_name: str, candidate_name: str) -> tuple[list[dict], list[dict], dict]:
    trusted = {row_key(row): row for row in selected_rows(trusted_dir, mode=mode)}
    candidate = {row_key(row): row for row in selected_rows(candidate_dir, mode=mode)}
    common_keys = sorted(set(trusted) & set(candidate))
    detail_rows = []
    by_candidate: dict[str, list[dict]] = defaultdict(list)
    for key in common_keys:
        left = trusted[key]
        right = candidate[key]
        candidate_key, example_id, prompt_variant = key
        row = {
            "candidate": candidate_key,
            "example_id": example_id,
            "prompt_variant": prompt_variant,
            f"{trusted_name}_exact": float(left.get("exact", 0.0)),
            f"{candidate_name}_exact": float(right.get("exact", 0.0)),
            f"{trusted_name}_malformed": bool_float(left.get("malformed")),
            f"{candidate_name}_malformed": bool_float(right.get("malformed")),
            f"{trusted_name}_cap_hit": float(left.get("cap_hit", 0.0)),
            f"{candidate_name}_cap_hit": float(right.get("cap_hit", 0.0)),
            f"{trusted_name}_answer": left.get("answer", ""),
            f"{candidate_name}_answer": right.get("answer", ""),
            "answer_equal": left.get("answer", "") == right.get("answer", ""),
            "text_equal": left.get("text", "") == right.get("text", ""),
        }
        row["exact_delta"] = row[f"{candidate_name}_exact"] - row[f"{trusted_name}_exact"]
        row["malformed_delta"] = row[f"{candidate_name}_malformed"] - row[f"{trusted_name}_malformed"]
        row["cap_hit_delta"] = row[f"{candidate_name}_cap_hit"] - row[f"{trusted_name}_cap_hit"]
        detail_rows.append(row)
        by_candidate[candidate_key].append(row)

    candidate_rows = []
    for candidate_key, rows in by_candidate.items():
        trusted_exact = [row[f"{trusted_name}_exact"] for row in rows]
        candidate_exact = [row[f"{candidate_name}_exact"] for row in rows]
        candidate_rows.append(
            {
                "candidate": candidate_key,
                "n": len(rows),
                f"{trusted_name}_exact_mean": mean(trusted_exact),
                f"{candidate_name}_exact_mean": mean(candidate_exact),
                "exact_delta": mean([row["exact_delta"] for row in rows]),
                "exact_disagreement_rate": mean([row[f"{trusted_name}_exact"] != row[f"{candidate_name}_exact"] for row in rows]),
                "answer_equal_rate": mean([row["answer_equal"] for row in rows]),
                "text_equal_rate": mean([row["text_equal"] for row in rows]),
                "malformed_delta": mean([row["malformed_delta"] for row in rows]),
                "cap_hit_delta": mean([row["cap_hit_delta"] for row in rows]),
            }
        )
    candidate_rows.sort(key=lambda row: (abs(row["exact_delta"]), row["exact_disagreement_rate"]), reverse=True)
    summary = {
        "kind": "backend_output_diff",
        "trusted_dir": str(trusted_dir),
        "candidate_dir": str(candidate_dir),
        "trusted_name": trusted_name,
        "candidate_name": candidate_name,
        "mode": mode,
        "trusted_rows": len(trusted),
        "candidate_rows": len(candidate),
        "common_rows": len(detail_rows),
        "common_candidates": len(candidate_rows),
        "exact_disagreement_rate": mean([row[f"{trusted_name}_exact"] != row[f"{candidate_name}_exact"] for row in detail_rows]) if detail_rows else None,
        "answer_equal_rate": mean([row["answer_equal"] for row in detail_rows]) if detail_rows else None,
        "text_equal_rate": mean([row["text_equal"] for row in detail_rows]) if detail_rows else None,
        "mean_abs_exact_delta_by_candidate": mean([abs(row["exact_delta"]) for row in candidate_rows]) if candidate_rows else None,
        "max_abs_exact_delta_by_candidate": max([abs(row["exact_delta"]) for row in candidate_rows]) if candidate_rows else None,
        "worst_candidates": candidate_rows[:8],
    }
    return detail_rows, candidate_rows, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: list[dict], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    return "\n".join(lines)


def write_report(path: Path, candidate_rows: list[dict], summary: dict) -> None:
    lines = [
        "# Backend Output Diff",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| common rows | {summary['common_rows']} |",
        f"| common candidates | {summary['common_candidates']} |",
        f"| exact disagreement rate | {summary['exact_disagreement_rate']} |",
        f"| answer equal rate | {summary['answer_equal_rate']} |",
        f"| text equal rate | {summary['text_equal_rate']} |",
        f"| max abs exact delta by candidate | {summary['max_abs_exact_delta_by_candidate']} |",
        "",
        "## Worst Candidate Deltas",
        "",
        markdown_table(
            candidate_rows[:16],
            [
                "candidate",
                "n",
                f"{summary['trusted_name']}_exact_mean",
                f"{summary['candidate_name']}_exact_mean",
                "exact_delta",
                "exact_disagreement_rate",
                "answer_equal_rate",
                "text_equal_rate",
            ],
        ),
        "",
    ]
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare saved per-prompt outputs from two backend runs.")
    p.add_argument("--trusted", required=True, type=Path)
    p.add_argument("--candidate", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--trusted-name", default="peft")
    p.add_argument("--candidate-name", default="vllm")
    p.add_argument("--mode", default="screen")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    detail_rows, candidate_rows, summary = compare_rows(
        args.trusted,
        args.candidate,
        mode=args.mode,
        trusted_name=args.trusted_name,
        candidate_name=args.candidate_name,
    )
    write_csv(args.out / "per_prompt_diff.csv", detail_rows)
    write_csv(args.out / "candidate_diff.csv", candidate_rows)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.out / "report.md", candidate_rows, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
