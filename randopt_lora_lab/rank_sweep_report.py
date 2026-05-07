from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any


RANK_RE = re.compile(r"rank(\d+)$")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def collect_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rank_dir in sorted(root.glob("rank*"), key=lambda p: int(RANK_RE.match(p.name).group(1)) if RANK_RE.match(p.name) else 10**9):
        match = RANK_RE.match(rank_dir.name)
        if not match:
            continue
        report_path = rank_dir / "report" / "summary.json"
        if not report_path.exists():
            continue
        report = _load_json(report_path)
        for arm, comparison in sorted(report.get("comparisons", {}).items()):
            rows.append(
                {
                    "rank": int(match.group(1)),
                    "arm": arm,
                    "pass": comparison.get("pass"),
                    "spearman": comparison.get("spearman"),
                    "topk_overlap": comparison.get("topk_overlap"),
                    "selected_regret": comparison.get("selected_regret"),
                    "speed_ratio_over_dense": comparison.get("speed_ratio_lora_over_dense"),
                    "dense_ensemble": comparison.get("dense_best_ensemble_holdout_exact"),
                    "arm_ensemble": comparison.get("lora_best_ensemble_holdout_exact"),
                    "ensemble_delta": comparison.get("ensemble_holdout_delta_lora_minus_dense"),
                    "dense_best_score": comparison.get("dense_best_score"),
                    "arm_pick_score": comparison.get("lora_pick_score"),
                    "dense_score_at_arm_pick": comparison.get("dense_score_at_lora_pick"),
                    "arm_pick_cap_hit": comparison.get("lora_pick_cap_hit_mean"),
                    "arm_pick_malformed": comparison.get("lora_pick_malformed_mean"),
                }
            )
    return rows


def render_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# Rank Sweep Summary",
        "",
        "| rank | arm | pass | Spearman | top-k overlap | regret | speed/dense | dense ensemble | arm ensemble | ensemble delta | cap-hit | malformed |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    _fmt(row["rank"]),
                    _fmt(row["arm"]),
                    _fmt(row["pass"]),
                    _fmt(row["spearman"]),
                    _fmt(row["topk_overlap"]),
                    _fmt(row["selected_regret"]),
                    _fmt(row["speed_ratio_over_dense"]),
                    _fmt(row["dense_ensemble"]),
                    _fmt(row["arm_ensemble"]),
                    _fmt(row["ensemble_delta"]),
                    _fmt(row["arm_pick_cap_hit"]),
                    _fmt(row["arm_pick_malformed"]),
                ]
            )
            + " |"
        )
    lines.append("")
    if rows:
        passed = [row for row in rows if row.get("pass")]
        lines.append(f"Overall pass: `{str(bool(passed) and len(passed) == len(rows)).lower()}`")
    else:
        lines.append("Overall pass: `false`")
    lines.append("")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else [
        "rank",
        "arm",
        "pass",
        "spearman",
        "topk_overlap",
        "selected_regret",
        "speed_ratio_over_dense",
        "dense_ensemble",
        "arm_ensemble",
        "ensemble_delta",
        "dense_best_score",
        "arm_pick_score",
        "dense_score_at_arm_pick",
        "arm_pick_cap_hit",
        "arm_pick_malformed",
    ]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize dense-vs-candidate rank sweep parity reports.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    rows = collect_rows(args.root)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps({"rows": rows, "pass": bool(rows) and all(row.get("pass") for row in rows)}, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(rows))
    write_csv(args.out / "summary.csv", rows)
    print(json.dumps({"rows": rows, "pass": bool(rows) and all(row.get("pass") for row in rows)}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
