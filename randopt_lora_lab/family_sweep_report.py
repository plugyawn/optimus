from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return read_json(path)


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def holdout_total(run_dir: Path) -> int | None:
    summary = optional_json(run_dir / "summary.json")
    if not summary:
        return None
    value = summary.get("holdout_prompts")
    return int(value) if value is not None else None


def validity_pass(run_dir: Path) -> bool:
    summary = optional_json(run_dir / "validity" / "summary.json")
    return bool(summary and summary.get("pass"))


def arm_run_dir(root: Path, arm: str, *, baseline_arm: str, baseline_dir: str) -> Path:
    if arm == baseline_arm:
        return root / baseline_dir
    return root / arm


def collect_variant(
    variant: str,
    root: Path,
    *,
    baseline_arm: str,
    baseline_dir: str,
    min_improvement_examples: int,
    max_cap_hit_delta: float,
    max_malformed_delta: float,
    min_speed_ratio_over_dense: float,
) -> dict[str, Any]:
    parity = read_json(root / "parity" / "summary.json")
    comparisons = parity.get("comparisons", {})
    if baseline_arm not in comparisons:
        raise ValueError(f"baseline arm {baseline_arm!r} missing from {root / 'parity' / 'summary.json'}")
    baseline = comparisons[baseline_arm]
    dense_valid = validity_pass(root / "dense")
    baseline_valid = validity_pass(root / baseline_dir)
    total = holdout_total(root / "dense")
    baseline_ensemble = baseline.get("lora_best_ensemble_holdout_exact")
    baseline_cap = baseline.get("lora_pick_cap_hit_mean")
    baseline_malformed = baseline.get("lora_pick_malformed_mean")

    rows = []
    for arm, comparison in sorted(comparisons.items()):
        arm_valid = validity_pass(arm_run_dir(root, arm, baseline_arm=baseline_arm, baseline_dir=baseline_dir))
        arm_ensemble = comparison.get("lora_best_ensemble_holdout_exact")
        delta = None
        delta_examples = None
        if baseline_ensemble is not None and arm_ensemble is not None:
            delta = float(arm_ensemble) - float(baseline_ensemble)
            if total is not None:
                delta_examples = delta * total
        arm_cap = comparison.get("lora_pick_cap_hit_mean")
        arm_malformed = comparison.get("lora_pick_malformed_mean")
        cap_delta = None if baseline_cap is None or arm_cap is None else float(arm_cap) - float(baseline_cap)
        malformed_delta = (
            None if baseline_malformed is None or arm_malformed is None else float(arm_malformed) - float(baseline_malformed)
        )
        speed = comparison.get("speed_ratio_lora_over_dense")
        quality_pass = arm == baseline_arm or (
            delta_examples is not None and delta_examples >= float(min_improvement_examples)
        )
        cap_pass = cap_delta is not None and cap_delta <= max_cap_hit_delta
        malformed_pass = malformed_delta is not None and malformed_delta <= max_malformed_delta
        speed_pass = speed is not None and float(speed) >= min_speed_ratio_over_dense
        row_pass = bool(
            arm != baseline_arm
            and dense_valid
            and baseline_valid
            and arm_valid
            and quality_pass
            and cap_pass
            and malformed_pass
            and speed_pass
        )
        rows.append(
            {
                "variant": variant,
                "arm": arm,
                "baseline_arm": baseline_arm,
                "dense_validity_pass": dense_valid,
                "baseline_validity_pass": baseline_valid,
                "arm_validity_pass": arm_valid,
                "dense_best_ensemble_holdout_exact": comparison.get("dense_best_ensemble_holdout_exact"),
                "baseline_best_ensemble_holdout_exact": baseline_ensemble,
                "arm_best_ensemble_holdout_exact": arm_ensemble,
                "ensemble_delta_vs_baseline": delta,
                "ensemble_delta_examples_vs_baseline": delta_examples,
                "selected_regret_vs_dense": comparison.get("selected_regret"),
                "spearman_vs_dense": comparison.get("spearman"),
                "topk_overlap_vs_dense": comparison.get("topk_overlap"),
                "speed_ratio_over_dense": speed,
                "baseline_pick_cap_hit_mean": baseline_cap,
                "arm_pick_cap_hit_mean": arm_cap,
                "cap_hit_delta_vs_baseline": cap_delta,
                "baseline_pick_malformed_mean": baseline_malformed,
                "arm_pick_malformed_mean": arm_malformed,
                "malformed_delta_vs_baseline": malformed_delta,
                "gates": {
                    "quality_improves_baseline": quality_pass,
                    "dense_validity": dense_valid,
                    "baseline_validity": baseline_valid,
                    "arm_validity": arm_valid,
                    "cap_hit_not_worse": cap_pass,
                    "malformed_not_worse": malformed_pass,
                    "speed_over_dense": speed_pass,
                },
                "pass": row_pass,
            }
        )
    return {"variant": variant, "root": str(root), "holdout_prompts": total, "rows": rows}


def aggregate(
    variant_roots: list[tuple[str, Path]],
    *,
    baseline_arm: str = "factor",
    baseline_dir: str = "factor",
    min_variants: int = 2,
    min_improvement_examples: int = 2,
    max_cap_hit_delta: float = 0.02,
    max_malformed_delta: float = 0.02,
    min_speed_ratio_over_dense: float = 1.0,
) -> dict[str, Any]:
    variants = [
        collect_variant(
            name,
            path,
            baseline_arm=baseline_arm,
            baseline_dir=baseline_dir,
            min_improvement_examples=min_improvement_examples,
            max_cap_hit_delta=max_cap_hit_delta,
            max_malformed_delta=max_malformed_delta,
            min_speed_ratio_over_dense=min_speed_ratio_over_dense,
        )
        for name, path in variant_roots
    ]
    rows = [row for variant in variants for row in variant["rows"]]
    by_arm: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["arm"] != baseline_arm:
            by_arm[row["arm"]].append(row)
    arm_pass = {
        arm: len(arm_rows) >= min_variants and all(row["pass"] for row in arm_rows)
        for arm, arm_rows in sorted(by_arm.items())
    }
    failed = []
    if len(variants) < min_variants:
        failed.append("min_variants")
    if not any(arm_pass.values()):
        failed.append("no_family_beats_baseline_across_variants")
    return {
        "kind": "family_sweep_report",
        "pass": not failed,
        "failed": failed,
        "thresholds": {
            "baseline_arm": baseline_arm,
            "baseline_dir": baseline_dir,
            "min_variants": min_variants,
            "min_improvement_examples": min_improvement_examples,
            "max_cap_hit_delta": max_cap_hit_delta,
            "max_malformed_delta": max_malformed_delta,
            "min_speed_ratio_over_dense": min_speed_ratio_over_dense,
        },
        "arm_pass": arm_pass,
        "variants": variants,
        "rows": rows,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Family Sweep Report",
        "",
        f"Overall pass: `{str(summary['pass']).lower()}`",
        "",
        "## Arm Pass",
        "",
        "| arm | pass |",
        "| --- | ---: |",
    ]
    for arm, passed in summary["arm_pass"].items():
        lines.append(f"| {arm} | {str(passed).lower()} |")
    lines.extend(
        [
            "",
            "## Rows",
            "",
            "| variant | arm | pass | ensemble delta examples vs baseline | arm ensemble | baseline ensemble | selected regret vs dense | Spearman vs dense | speed/dense | cap delta | malformed delta |",
            "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    fmt(row["variant"]),
                    fmt(row["arm"]),
                    fmt(row["pass"]),
                    fmt(row["ensemble_delta_examples_vs_baseline"]),
                    fmt(row["arm_best_ensemble_holdout_exact"]),
                    fmt(row["baseline_best_ensemble_holdout_exact"]),
                    fmt(row["selected_regret_vs_dense"]),
                    fmt(row["spearman_vs_dense"]),
                    fmt(row["speed_ratio_over_dense"]),
                    fmt(row["cap_hit_delta_vs_baseline"]),
                    fmt(row["malformed_delta_vs_baseline"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


def parse_variant_root(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--variant-root must be NAME=PATH")
    name, path = value.split("=", 1)
    name = name.strip()
    if not name:
        raise argparse.ArgumentTypeError("variant name cannot be empty")
    return name, Path(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarize matched LoRA-family sweeps against a baseline LoRA arm.")
    parser.add_argument("--variant-root", action="append", type=parse_variant_root, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--baseline-arm", default="factor")
    parser.add_argument("--baseline-dir", default="factor")
    parser.add_argument("--min-variants", type=int, default=2)
    parser.add_argument("--min-improvement-examples", type=int, default=2)
    parser.add_argument("--max-cap-hit-delta", type=float, default=0.02)
    parser.add_argument("--max-malformed-delta", type=float, default=0.02)
    parser.add_argument("--min-speed-ratio-over-dense", type=float, default=1.0)
    args = parser.parse_args(argv)

    summary = aggregate(
        args.variant_root,
        baseline_arm=args.baseline_arm,
        baseline_dir=args.baseline_dir,
        min_variants=args.min_variants,
        min_improvement_examples=args.min_improvement_examples,
        max_cap_hit_delta=args.max_cap_hit_delta,
        max_malformed_delta=args.max_malformed_delta,
        min_speed_ratio_over_dense=args.min_speed_ratio_over_dense,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
