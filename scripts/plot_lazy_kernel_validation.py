#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _labelled_run(text: str) -> tuple[str, Path]:
    if "=" not in text:
        path = Path(text)
        return path.name, path
    label, raw = text.split("=", 1)
    label = label.strip()
    if not label:
        raise ValueError(f"empty run label in {text!r}")
    return label, Path(raw)


def _candidate_scores(run: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in _jsonl(run / "candidate_scores.jsonl"):
        candidate_id = str(row.get("candidate_id") or row.get("candidate") or "")
        if candidate_id in {"", "base", "__base__"}:
            continue
        metrics = row.get("aggregate_metrics") or {}
        if "exact" in metrics:
            out[candidate_id] = float(metrics["exact"])
        elif "exact_mean" in row:
            out[candidate_id] = float(row["exact_mean"])
    return out


def _summary_row(label: str, run: Path) -> dict[str, Any]:
    summary = _read_json(run / "summary.json")
    base = float(summary.get("base_final_score", summary.get("base_holdout_score", 0.0)))
    best = summary.get("confirmed_best_candidate_final_score")
    if best is None:
        best = summary.get("best_candidate_final_score", summary.get("promoted_best_holdout_score", base))
    return {
        "label": label,
        "run_dir": str(run),
        "kind": summary.get("kind"),
        "population": summary.get("population"),
        "basis_rank": summary.get("basis_rank"),
        "effective_rank": summary.get("effective_rank", summary.get("adapter_rank")),
        "scale_multiplier": summary.get("scale_multiplier"),
        "targets": ",".join(summary.get("targets") or []),
        "base_score": base,
        "best_score": float(best),
        "delta_vs_base": float(best) - base,
        "best_candidate_id": summary.get("confirmed_best_candidate_id") or summary.get("best_candidate_id"),
        "candidate_sec": summary.get("mixed_candidate_sec") or summary.get("candidates_per_sec"),
        "confirmed_candidate_sec": summary.get("confirmed_mixed_candidate_sec"),
        "candidate_replay_sec": summary.get("candidate_replay_sec"),
        "confirmed_candidate_replay_sec": summary.get("confirmed_candidate_replay_sec"),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _md_table(rows: list[dict[str, Any]], columns: list[str]) -> str:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows:
        vals = []
        for col in columns:
            val = row.get(col, "")
            if isinstance(val, float):
                val = f"{val:.4g}"
            vals.append(str(val))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _plot_quality(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    base = [float(row["base_score"]) for row in rows]
    best = [float(row["best_score"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(6, 1.4 * len(rows)), 4.6))
    ax.bar([idx - 0.18 for idx in x], base, width=0.35, label="base", color="#6b7280")
    ax.bar([idx + 0.18 for idx in x], best, width=0.35, label="best K=1", color="#256f5c")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=25, ha="right")
    ax.set_ylabel("exact accuracy")
    ax.set_title("Lazy-kernel K=1 quality")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_throughput(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    selected = [row for row in rows if row.get("candidate_sec") is not None]
    labels = [str(row["label"]) for row in selected]
    values = [float(row["candidate_sec"]) for row in selected]
    fig, ax = plt.subplots(figsize=(max(6, 1.3 * len(selected)), 4.4))
    ax.bar(labels, values, color="#2f6f73")
    ax.set_ylabel("candidates/sec")
    ax.set_title("Search/replay throughput")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_parity(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [float(row["trusted_exact"]) for row in rows]
    ys = [float(row["candidate_exact"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.scatter(xs, ys, s=28, color="#256f5c", alpha=0.75, edgecolor="#111827", linewidth=0.25)
    lo = min(xs + ys) if rows else 0.0
    hi = max(xs + ys) if rows else 1.0
    pad = max(0.02, 0.05 * (hi - lo))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("trusted adapter exact")
    ax.set_ylabel("true lazy exact")
    ax.set_title("Candidate-score parity")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot true-lazy GQ validation against subspace-as-LoRA artifacts.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, help="label=RUN_DIR")
    parser.add_argument("--trusted-run", type=Path, help="Adapter/materialized run used as candidate-score parity reference.")
    parser.add_argument("--candidate-run", type=Path, help="True-lazy run compared against --trusted-run.")
    args = parser.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    runs = [_labelled_run(item) for item in args.run]
    summary_rows = [_summary_row(label, run) for label, run in runs]
    summary_cols = [
        "label",
        "kind",
        "population",
        "basis_rank",
        "effective_rank",
        "scale_multiplier",
        "targets",
        "base_score",
        "best_score",
        "delta_vs_base",
        "best_candidate_id",
        "candidate_sec",
        "confirmed_candidate_sec",
    ]
    _write_csv(args.out / "summary.csv", summary_rows, summary_cols + ["run_dir", "candidate_replay_sec", "confirmed_candidate_replay_sec"])
    (args.out / "summary.md").write_text("# Lazy Kernel Validation\n\n" + _md_table(summary_rows, summary_cols) + "\n")
    _plot_quality(args.out / "quality.png", summary_rows)
    _plot_throughput(args.out / "throughput.png", summary_rows)

    if args.trusted_run and args.candidate_run:
        trusted = _candidate_scores(args.trusted_run)
        candidate = _candidate_scores(args.candidate_run)
        common = sorted(set(trusted) & set(candidate))
        parity_rows = [
            {
                "candidate_id": candidate_id,
                "trusted_exact": trusted[candidate_id],
                "candidate_exact": candidate[candidate_id],
                "diff": candidate[candidate_id] - trusted[candidate_id],
            }
            for candidate_id in common
        ]
        _write_csv(args.out / "candidate_parity.csv", parity_rows, ["candidate_id", "trusted_exact", "candidate_exact", "diff"])
        if parity_rows:
            _plot_parity(args.out / "candidate_score_parity.png", parity_rows)
        max_abs = max((abs(row["diff"]) for row in parity_rows), default=None)
        parity_summary = {
            "trusted_run": str(args.trusted_run),
            "candidate_run": str(args.candidate_run),
            "common_candidates": len(common),
            "max_abs_score_diff": max_abs,
            "exact_score_match": max_abs == 0.0 if max_abs is not None else False,
        }
        (args.out / "parity_summary.json").write_text(json.dumps(parity_summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
