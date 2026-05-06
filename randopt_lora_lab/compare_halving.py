from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .compare_backends import parse_ks, read_jsonl


def rows_by_candidate(path: Path) -> dict[str, dict]:
    rows = read_jsonl(path)
    if not rows:
        raise FileNotFoundError(f"no rows found in {path}")
    by_candidate = {}
    for idx, row in enumerate(rows):
        if "candidate" not in row or "exact_mean" not in row:
            raise ValueError(f"{path} row {idx} missing candidate/exact_mean")
        by_candidate[str(row["candidate"])] = row
    return by_candidate


def top_ids(rows: dict[str, dict], k: int) -> list[str]:
    return [
        key
        for key, _ in sorted(
            rows.items(),
            key=lambda item: float(item[1]["exact_mean"]),
            reverse=True,
        )[:k]
    ]


def compare_halving(full_dir: Path, halving_dir: Path, *, ks: list[int]) -> tuple[list[dict], dict]:
    full_rows = rows_by_candidate(full_dir / "candidate_summary.jsonl")
    stage_rows = rows_by_candidate(halving_dir / "stage_candidate_summary.jsonl")
    survivor_rows = rows_by_candidate(halving_dir / "candidate_summary.jsonl")
    full_best_id = top_ids(full_rows, 1)[0]
    survivor_best_id = top_ids(survivor_rows, 1)[0]
    stage_best_id = top_ids(stage_rows, 1)[0]
    full_best_score = float(full_rows[full_best_id]["exact_mean"])
    survivor_best_full_score = float(full_rows[survivor_best_id]["exact_mean"])
    stage_best_full_score = float(full_rows[stage_best_id]["exact_mean"])

    survivor_set = set(survivor_rows)
    details = []
    for candidate in top_ids(full_rows, max(ks) if ks else 1):
        details.append(
            {
                "candidate": candidate,
                "full_exact_mean": float(full_rows[candidate]["exact_mean"]),
                "stage_exact_mean": float(stage_rows.get(candidate, {}).get("exact_mean", 0.0)),
                "survived": candidate in survivor_set,
                "survivor_screen_exact_mean": (
                    float(survivor_rows[candidate]["exact_mean"]) if candidate in survivor_rows else None
                ),
            }
        )

    summary = {
        "kind": "halving_recall",
        "full_dir": str(full_dir),
        "halving_dir": str(halving_dir),
        "n_full": len(full_rows),
        "n_stage": len(stage_rows),
        "n_survivors": len(survivor_rows),
        "full_best_candidate": full_best_id,
        "full_best_score": full_best_score,
        "halving_best_candidate": survivor_best_id,
        "halving_best_stage_score": float(stage_rows[survivor_best_id]["exact_mean"]),
        "halving_best_screen_score": float(survivor_rows[survivor_best_id]["exact_mean"]),
        "halving_best_full_score": survivor_best_full_score,
        "halving_selected_regret_vs_full": full_best_score - survivor_best_full_score,
        "stage_best_candidate": stage_best_id,
        "stage_best_stage_score": float(stage_rows[stage_best_id]["exact_mean"]),
        "stage_best_full_score": stage_best_full_score,
        "stage_selected_regret_vs_full": full_best_score - stage_best_full_score,
        "full_best_survived": full_best_id in survivor_set,
    }
    halving_summary_path = halving_dir / "summary.json"
    if halving_summary_path.exists():
        halving_summary = json.loads(halving_summary_path.read_text())
        for key in [
            "stage_prompts",
            "screen_prompts",
            "survivors",
            "prompt_eval_savings",
            "candidate_sec",
            "stage_candidate_sec",
            "screen_candidate_sec",
            "eval_elapsed_s",
        ]:
            if key in halving_summary:
                summary[key] = halving_summary[key]
    for k in ks:
        full_top = set(top_ids(full_rows, min(k, len(full_rows))))
        summary[f"top{k}_survivor_recall"] = len(full_top & survivor_set)
        summary[f"top{k}_possible"] = len(full_top)
    return details, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    columns = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary: dict) -> None:
    lines = [
        "# Halving Recall Report",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| full candidates | {summary['n_full']} |",
        f"| survivors | {summary['n_survivors']} |",
        f"| full best survived | {summary['full_best_survived']} |",
        f"| halving selected regret | {summary['halving_selected_regret_vs_full']} |",
        f"| stage selected regret | {summary['stage_selected_regret_vs_full']} |",
    ]
    if "prompt_eval_savings" in summary:
        lines.append(f"| prompt eval savings | {summary['prompt_eval_savings']} |")
    for key, value in summary.items():
        if key.startswith("top") and key.endswith("_survivor_recall"):
            possible = summary.get(key.replace("_survivor_recall", "_possible"))
            lines.append(f"| {key} | {value}/{possible} |")
    lines.extend(
        [
            "",
            "## Winners",
            "",
            f"- Full best: `{summary['full_best_candidate']}` at `{summary['full_best_score']}`.",
            f"- Halving best: `{summary['halving_best_candidate']}`; full score `{summary['halving_best_full_score']}`, halving score `{summary['halving_best_screen_score']}`.",
            f"- Stage best: `{summary['stage_best_candidate']}`; full score `{summary['stage_best_full_score']}`, stage score `{summary['stage_best_stage_score']}`.",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Compare staged halving against a full-screen run.")
    p.add_argument("--full", required=True, type=Path)
    p.add_argument("--halving", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ks", default="4,8,16,32")
    args = p.parse_args(argv)

    args.out.mkdir(parents=True, exist_ok=True)
    details, summary = compare_halving(args.full, args.halving, ks=parse_ks(args.ks))
    write_csv(args.out / "details.csv", details)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    write_report(args.out / "report.md", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
