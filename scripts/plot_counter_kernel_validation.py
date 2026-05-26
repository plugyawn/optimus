#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def _json(path: Path) -> dict[str, Any]:
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


def _slug(text: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", text).strip("_") or "run"


def _mean(values: list[float]) -> float | None:
    return None if not values else sum(values) / len(values)


def _first_not_none(*values: Any) -> Any:
    for value in values:
        if value is not None:
            return value
    return None


def _score_map(run: Path, *, split: str, stage: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for row in _jsonl(run / "candidate_scores.jsonl"):
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in {"", "__base__", "base"}:
            continue
        if row.get("split") != split or row.get("selection_stage") != stage:
            continue
        metrics = row.get("aggregate_metrics") or {}
        if "exact" in metrics:
            out[candidate_id] = float(metrics["exact"])
    return out


def _prompt_map(run: Path, *, split: str) -> dict[tuple[str, str], dict[str, Any]]:
    out: dict[tuple[str, str], dict[str, Any]] = {}
    for row in _jsonl(run / "per_prompt.jsonl"):
        candidate_id = str(row.get("candidate_id") or "")
        if candidate_id in {"", "__base__", "base"}:
            continue
        if row.get("split") != split:
            continue
        out[(candidate_id, str(row.get("example_id")))] = row
    return out


def _summary_row(label: str, run: Path) -> dict[str, Any]:
    summary = _json(run / "summary.json")
    systems = _json(run / "systems_report.json") if (run / "systems_report.json").exists() else {}
    merged = {**summary, **systems}
    lazy = summary.get("lazy_timing") or systems.get("lazy_timing") or {}
    elapsed_s = _first_not_none(lazy.get("elapsed_s"), merged.get("elapsed_s"))
    output_tokens = _first_not_none(lazy.get("output_tokens"), merged.get("output_tokens"))
    output_tokens_per_sec = merged.get("output_tokens_per_sec")
    if output_tokens_per_sec is None and elapsed_s and output_tokens is not None:
        output_tokens_per_sec = float(output_tokens) / float(elapsed_s)
    return {
        "label": label,
        "run_dir": str(run),
        "kind": merged.get("kind"),
        "rng_version": merged.get("rng_version") or summary.get("rng_version"),
        "population": merged.get("population"),
        "candidate_batch_size": merged.get("candidate_batch_size"),
        "basis_rank": merged.get("basis_rank"),
        "target_preset": merged.get("target_preset") or summary.get("target_preset"),
        "candidates_per_sec": _first_not_none(merged.get("candidates_per_sec"), summary.get("mixed_candidate_sec"), lazy.get("mixed_candidate_sec")),
        "prompts_per_sec": merged.get("prompts_per_sec"),
        "output_tokens_per_sec": output_tokens_per_sec,
        "lazy_timing_mode": _first_not_none(merged.get("lazy_timing_mode"), lazy.get("lazy_timing_mode")),
        "lazy_delta_time_s": _first_not_none(merged.get("lazy_delta_time_s"), lazy.get("lazy_delta_time_s")),
        "lazy_delta_dispatch_time_s": _first_not_none(merged.get("lazy_delta_dispatch_time_s"), lazy.get("lazy_delta_dispatch_time_s")),
        "lazy_kernel_time_s": _first_not_none(merged.get("lazy_kernel_time_s"), lazy.get("lazy_kernel_time_s")),
        "lazy_stack_time_s": _first_not_none(merged.get("lazy_stack_time_s"), lazy.get("lazy_stack_time_s")),
        "lazy_meta_time_s": _first_not_none(merged.get("lazy_meta_time_s"), lazy.get("lazy_meta_time_s")),
        "qx_time_s": _first_not_none(merged.get("qx_time_s"), lazy.get("qx_time_s")),
        "delta_calls": _first_not_none(merged.get("delta_calls"), lazy.get("delta_calls")),
        "delta_rows": _first_not_none(merged.get("delta_rows"), lazy.get("delta_rows")),
        "scoring_time_s": merged.get("scoring_time_s"),
        "base_screen_score": merged.get("base_screen_score"),
        "base_holdout_score": merged.get("base_holdout_score"),
        "screen_score": _first_not_none(merged.get("screen_score"), summary.get("best_screen_score"), summary.get("best_candidate_final_score")),
        "holdout_score": _first_not_none(merged.get("holdout_score"), summary.get("selected_holdout_score"), summary.get("confirmed_best_candidate_final_score")),
        "best_ensemble_holdout_exact": merged.get("best_ensemble_holdout_exact"),
        "best_strict_ensemble_holdout_exact": merged.get("best_strict_ensemble_holdout_exact"),
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


def _score_parity(reference: dict[str, float], candidate: dict[str, float]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    common = sorted(set(reference) & set(candidate))
    rows = []
    for candidate_id in common:
        ref = reference[candidate_id]
        got = candidate[candidate_id]
        rows.append(
            {
                "candidate_id": candidate_id,
                "reference_exact": ref,
                "candidate_exact": got,
                "diff": got - ref,
                "abs_diff": abs(got - ref),
                "exact_match": ref == got,
            }
        )
    diffs = [float(row["diff"]) for row in rows]
    abs_diffs = [abs(diff) for diff in diffs]
    return (
        {
            "reference_candidates": len(reference),
            "candidate_candidates": len(candidate),
            "common_candidates": len(rows),
            "score_mismatch_count": sum(1 for row in rows if not row["exact_match"]),
            "max_abs_score_diff": max(abs_diffs, default=None),
            "mean_abs_score_diff": _mean(abs_diffs),
            "rmse_score_diff": None if not diffs else math.sqrt(sum(diff * diff for diff in diffs) / len(diffs)),
        },
        rows,
    )


def _prompt_parity(reference: dict[tuple[str, str], dict[str, Any]], candidate: dict[tuple[str, str], dict[str, Any]]) -> dict[str, Any]:
    common = sorted(set(reference) & set(candidate))
    text_matches = 0
    exact_matches = 0
    examples = []
    for key in common:
        ref = reference[key]
        got = candidate[key]
        ref_text = str(ref.get("text") or "")
        got_text = str(got.get("text") or "")
        ref_exact = float(ref.get("exact") or 0.0)
        got_exact = float(got.get("exact") or 0.0)
        if ref_text == got_text:
            text_matches += 1
        if ref_exact == got_exact:
            exact_matches += 1
        if (ref_text != got_text or ref_exact != got_exact) and len(examples) < 8:
            examples.append(
                {
                    "candidate_id": key[0],
                    "example_id": key[1],
                    "reference_exact": ref_exact,
                    "candidate_exact": got_exact,
                    "reference_tokens": ref.get("output_tokens"),
                    "candidate_tokens": got.get("output_tokens"),
                    "reference_text": ref_text[:240],
                    "candidate_text": got_text[:240],
                }
            )
    return {
        "reference_prompt_rows": len(reference),
        "candidate_prompt_rows": len(candidate),
        "common_prompt_rows": len(common),
        "text_match_count": text_matches,
        "text_match_rate": None if not common else text_matches / len(common),
        "exact_match_count": exact_matches,
        "exact_match_rate": None if not common else exact_matches / len(common),
        "mismatch_examples": examples,
    }


def _plot_throughput(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    values = [float(row.get("candidates_per_sec") or 0.0) for row in rows]
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(rows)), 4.6))
    ax.bar(labels, values, color="#2f6f73")
    ax.set_ylabel("candidates/sec")
    ax.set_title("Counter-kernel end-to-end throughput")
    ax.tick_params(axis="x", rotation=25)
    ax.grid(axis="y", alpha=0.25)
    for idx, value in enumerate(values):
        ax.text(idx, value, f"{value:.2f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_timing(path: Path, rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["label"]) for row in rows]
    x = list(range(len(rows)))
    parts = [
        ("kernel", [float(row.get("lazy_kernel_time_s") or 0.0) for row in rows], "#2563eb"),
        ("stack", [float(row.get("lazy_stack_time_s") or 0.0) for row in rows], "#d97706"),
        ("Qx", [float(row.get("qx_time_s") or 0.0) for row in rows], "#7c3aed"),
        ("other lazy", [max(0.0, float(row.get("lazy_delta_time_s") or 0.0) - float(row.get("lazy_kernel_time_s") or 0.0) - float(row.get("lazy_stack_time_s") or 0.0) - float(row.get("qx_time_s") or 0.0)) for row in rows], "#9ca3af"),
    ]
    fig, ax = plt.subplots(figsize=(max(7, 1.35 * len(rows)), 4.8))
    bottoms = [0.0 for _ in rows]
    for label, values, color in parts:
        ax.bar(x, values, bottom=bottoms, label=label, color=color)
        bottoms = [a + b for a, b in zip(bottoms, values)]
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=25)
    ax.set_ylabel("seconds")
    ax.set_title("Lazy-path timing breakdown")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_score_parity(path: Path, rows: list[dict[str, Any]], *, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs = [float(row["reference_exact"]) for row in rows]
    ys = [float(row["candidate_exact"]) for row in rows]
    fig, ax = plt.subplots(figsize=(5.0, 5.0))
    ax.scatter(xs, ys, s=26, color="#256f5c", alpha=0.75, edgecolor="#111827", linewidth=0.25)
    lo = min(xs + ys) if rows else 0.0
    hi = max(xs + ys) if rows else 1.0
    pad = max(0.02, 0.05 * (hi - lo))
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#6b7280", linestyle="--", linewidth=1.0)
    ax.set_xlabel("reference exact")
    ax.set_ylabel("candidate exact")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Build validation plots for the stateless counter lazy subspace kernel.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--run", action="append", required=True, help="label=RUN_DIR")
    parser.add_argument("--reference-label", help="Run label used as score/output parity reference. Defaults to the first run.")
    parser.add_argument("--score-split", default="screen")
    parser.add_argument("--score-stage", default="screen")
    parser.add_argument("--prompt-split", default="screen")
    args = parser.parse_args()

    runs = [_labelled_run(item) for item in args.run]
    run_by_label = {label: path for label, path in runs}
    reference_label = args.reference_label or runs[0][0]
    if reference_label not in run_by_label:
        raise ValueError(f"unknown --reference-label {reference_label!r}")
    args.out.mkdir(parents=True, exist_ok=True)

    summary_rows = [_summary_row(label, run) for label, run in runs]
    summary_cols = [
        "label",
        "population",
        "candidate_batch_size",
        "rng_version",
        "candidates_per_sec",
        "output_tokens_per_sec",
        "lazy_timing_mode",
        "lazy_delta_time_s",
        "lazy_delta_dispatch_time_s",
        "lazy_kernel_time_s",
        "lazy_stack_time_s",
        "qx_time_s",
        "screen_score",
        "holdout_score",
        "best_ensemble_holdout_exact",
    ]
    _write_csv(args.out / "run_summary.csv", summary_rows, summary_cols + ["run_dir", "prompts_per_sec", "delta_calls", "delta_rows", "scoring_time_s"])
    _plot_throughput(args.out / "throughput.png", summary_rows)
    _plot_timing(args.out / "lazy_timing_breakdown.png", summary_rows)

    ref_run = run_by_label[reference_label]
    ref_scores = _score_map(ref_run, split=args.score_split, stage=args.score_stage)
    ref_prompts = _prompt_map(ref_run, split=args.prompt_split)
    parity: dict[str, Any] = {
        "reference_label": reference_label,
        "score_split": args.score_split,
        "score_stage": args.score_stage,
        "prompt_split": args.prompt_split,
        "pairs": {},
    }
    for label, run in runs:
        if label == reference_label:
            continue
        score_summary, score_rows = _score_parity(ref_scores, _score_map(run, split=args.score_split, stage=args.score_stage))
        prompt_summary = _prompt_parity(ref_prompts, _prompt_map(run, split=args.prompt_split))
        pair_key = f"{reference_label}__vs__{label}"
        score_csv = f"score_parity_{_slug(pair_key)}.csv"
        score_png = f"score_parity_{_slug(pair_key)}.png"
        _write_csv(args.out / score_csv, score_rows, ["candidate_id", "reference_exact", "candidate_exact", "diff", "abs_diff", "exact_match"])
        _plot_score_parity(args.out / score_png, score_rows, title=f"{reference_label} vs {label}")
        parity["pairs"][pair_key] = {
            "label": label,
            "run_dir": str(run),
            "score": {**score_summary, "csv": score_csv, "plot": score_png},
            "prompt": prompt_summary,
        }

    report = "# Counter Kernel Validation\n\n"
    report += "## Runs\n\n" + _md_table(summary_rows, summary_cols) + "\n\n"
    parity_rows = []
    for pair, payload in parity["pairs"].items():
        score = payload["score"]
        prompt = payload["prompt"]
        parity_rows.append(
            {
                "pair": pair,
                "common_candidates": score["common_candidates"],
                "score_mismatch_count": score["score_mismatch_count"],
                "max_abs_score_diff": score["max_abs_score_diff"],
                "common_prompt_rows": prompt["common_prompt_rows"],
                "text_match_rate": prompt["text_match_rate"],
                "exact_match_rate": prompt["exact_match_rate"],
            }
        )
    report += "## Parity\n\n" + _md_table(
        parity_rows,
        ["pair", "common_candidates", "score_mismatch_count", "max_abs_score_diff", "common_prompt_rows", "text_match_rate", "exact_match_rate"],
    ) + "\n"
    (args.out / "validation_summary.md").write_text(report)
    (args.out / "validation_summary.json").write_text(json.dumps({"runs": summary_rows, "parity": parity}, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
