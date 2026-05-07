from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from .compare_backends import spearman


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def f(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    return default if value is None else float(value)


def candidate_key(row: dict[str, Any]) -> str:
    return str(row["candidate"])


def pair_key(row: dict[str, Any]) -> tuple[str, int, float]:
    family = str(row.get("family") or str(row.get("candidate", "")).split(":", 1)[0])
    return family, int(row["seed"]), float(row["sigma"])


def top_rows(rows: list[dict[str, Any]], key: str, k: int) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: f(row, key), reverse=True)[: min(k, len(rows))]


def maybe_mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def top_mean(rows: list[dict[str, Any]], key: str, k: int) -> float | None:
    selected = top_rows(rows, key, k)
    return maybe_mean([f(row, key) for row in selected])


def top_set(rows: list[dict[str, Any]], key: str, k: int) -> set[str]:
    return {candidate_key(row) for row in top_rows(rows, key, k)}


def validity_fraction(rows: list[dict[str, Any]], max_cap_hit: float, max_malformed: float) -> float | None:
    if not rows:
        return None
    valid = [
        f(row, "cap_hit_mean") <= max_cap_hit and f(row, "malformed_mean") <= max_malformed
        for row in rows
    ]
    return sum(valid) / len(valid)


def prompt_variant_audit(condition_rows: list[dict[str, Any]], *, top_k: int) -> dict[str, Any]:
    by_variant: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in condition_rows:
        by_variant[str(row.get("prompt_variant", "default"))][candidate_key(row)] = row
    variants = sorted(by_variant)
    out: dict[str, Any] = {
        "variants": variants,
        "condition_rows": len(condition_rows),
    }
    if len(variants) < 2:
        out["available"] = False
        return out

    left, right = variants[:2]
    common = sorted(set(by_variant[left]) & set(by_variant[right]))
    left_rows = [by_variant[left][candidate] for candidate in common]
    right_rows = [by_variant[right][candidate] for candidate in common]
    out.update(
        {
            "available": True,
            "left_variant": left,
            "right_variant": right,
            "common_candidates": len(common),
            "selection_spearman": spearman(
                [f(row, "condition_selection_score") for row in left_rows],
                [f(row, "condition_selection_score") for row in right_rows],
            ),
            "exact_spearman": spearman(
                [f(row, "exact_mean") for row in left_rows],
                [f(row, "exact_mean") for row in right_rows],
            ),
            "selection_mean_abs_delta": maybe_mean(
                [
                    abs(f(left_row, "condition_selection_score") - f(right_row, "condition_selection_score"))
                    for left_row, right_row in zip(left_rows, right_rows)
                ]
            ),
            "exact_mean_abs_delta": maybe_mean(
                [
                    abs(f(left_row, "exact_mean") - f(right_row, "exact_mean"))
                    for left_row, right_row in zip(left_rows, right_rows)
                ]
            ),
        }
    )
    left_top = top_set(left_rows, "condition_selection_score", top_k)
    right_top = top_set(right_rows, "condition_selection_score", top_k)
    out[f"top{top_k}_selection_overlap"] = len(left_top & right_top)
    out[f"top{top_k}_selection_possible"] = min(top_k, len(common))
    return out


def holdout_transfer_audit(screen_rows: list[dict[str, Any]], holdout_rows: list[dict[str, Any]]) -> dict[str, Any]:
    screen = {candidate_key(row): row for row in screen_rows}
    holdout = {candidate_key(row): row for row in holdout_rows}
    common_keys = sorted(set(screen) & set(holdout))
    out: dict[str, Any] = {
        "screen_candidates": len(screen_rows),
        "holdout_candidates": len(holdout_rows),
        "common_candidates": len(common_keys),
    }
    if not common_keys:
        out["available"] = False
        return out
    screen_common = [screen[key] for key in common_keys]
    holdout_common = [holdout[key] for key in common_keys]
    best_holdout = max(holdout_common, key=lambda row: f(row, "exact_mean"))
    selected_by_screen = max(screen_common, key=lambda row: f(row, "selection_score"))
    selected_holdout = holdout[candidate_key(selected_by_screen)]
    out.update(
        {
            "available": True,
            "screen_selection_vs_holdout_exact_spearman": spearman(
                [f(row, "selection_score") for row in screen_common],
                [f(row, "exact_mean") for row in holdout_common],
            ),
            "screen_exact_vs_holdout_exact_spearman": spearman(
                [f(row, "exact_mean") for row in screen_common],
                [f(row, "exact_mean") for row in holdout_common],
            ),
            "best_holdout_candidate": candidate_key(best_holdout),
            "best_holdout_exact": f(best_holdout, "exact_mean"),
            "screen_selected_candidate": candidate_key(selected_by_screen),
            "screen_selected_holdout_exact": f(selected_holdout, "exact_mean"),
            "screen_selected_regret": f(best_holdout, "exact_mean") - f(selected_holdout, "exact_mean"),
            "holdout_exact_mean": maybe_mean([f(row, "exact_mean") for row in holdout_common]),
        }
    )
    return out


def antithetic_pair_audit(
    screen_rows: list[dict[str, Any]],
    *,
    top_k: int,
    max_cap_hit: float,
    max_malformed: float,
) -> dict[str, Any]:
    groups: dict[tuple[str, int, float], list[dict[str, Any]]] = defaultdict(list)
    for row in screen_rows:
        if "seed" in row and "sigma" in row and "sign" in row:
            groups[pair_key(row)].append(row)
    pairs = [rows for rows in groups.values() if {int(row["sign"]) for row in rows} >= {-1, 1}]
    out: dict[str, Any] = {
        "pairs": len(pairs),
        "rows_in_pairs": sum(len(rows) for rows in pairs),
    }
    if not pairs:
        out["available"] = False
        return out

    pair_rows = []
    for rows in pairs:
        best = max(rows, key=lambda row: f(row, "selection_score"))
        worst = min(rows, key=lambda row: f(row, "selection_score"))
        valid_flags = [
            f(row, "cap_hit_mean") <= max_cap_hit and f(row, "malformed_mean") <= max_malformed
            for row in rows
        ]
        pair_rows.append(
            {
                "candidate": candidate_key(best),
                "best_selection_score": f(best, "selection_score"),
                "best_exact_mean": f(best, "exact_mean"),
                "worst_selection_score": f(worst, "selection_score"),
                "score_gap": f(best, "selection_score") - f(worst, "selection_score"),
                "any_valid": any(valid_flags),
                "all_valid": all(valid_flags),
                "one_valid_one_invalid": any(valid_flags) and not all(valid_flags),
            }
        )
    top_pair_rows = top_rows(pair_rows, "best_selection_score", top_k)
    out.update(
        {
            "available": True,
            "pair_best_selection_top_mean": top_mean(pair_rows, "best_selection_score", top_k),
            "pair_best_exact_top_mean": top_mean(pair_rows, "best_exact_mean", top_k),
            "pair_score_gap_mean": maybe_mean([f(row, "score_gap") for row in pair_rows]),
            "pair_score_gap_top_mean": maybe_mean([f(row, "score_gap") for row in top_pair_rows]),
            "pair_any_valid_fraction": sum(1 for row in pair_rows if row["any_valid"]) / len(pair_rows),
            "pair_all_valid_fraction": sum(1 for row in pair_rows if row["all_valid"]) / len(pair_rows),
            "pair_one_valid_one_invalid_fraction": sum(1 for row in pair_rows if row["one_valid_one_invalid"])
            / len(pair_rows),
        }
    )
    return out


def summarize_run(
    run_dir: Path,
    *,
    top_k: int = 16,
    max_cap_hit: float = 0.05,
    max_malformed: float = 0.05,
) -> dict[str, Any]:
    summary = read_json(run_dir / "summary.json")
    screen_rows = read_jsonl(run_dir / "candidate_summary.jsonl")
    holdout_rows = read_jsonl(run_dir / "holdout_candidate_summary.jsonl")
    condition_rows = read_jsonl(run_dir / "candidate_condition_summary.jsonl")
    best_ensemble = max((f(row, "exact_mean") for row in summary.get("ensemble_holdout", [])), default=None)
    best_ensemble_row = max(summary.get("ensemble_holdout", []), key=lambda row: f(row, "exact_mean"), default={})
    return {
        "run_dir": str(run_dir),
        "population": summary.get("population"),
        "base_screen_exact": summary.get("base_screen_exact"),
        "base_holdout_exact": summary.get("base_holdout_exact"),
        "candidate_sec": summary.get("candidate_sec"),
        "screen_candidate_sec": summary.get("screen_candidate_sec"),
        "best_ensemble_holdout_exact": best_ensemble,
        "best_ensemble_k": best_ensemble_row.get("k"),
        "screen_candidates": len(screen_rows),
        "holdout_candidates": len(holdout_rows),
        f"screen_top{top_k}_selection_mean": top_mean(screen_rows, "selection_score", top_k),
        f"screen_top{top_k}_exact_mean": top_mean(screen_rows, "exact_mean", top_k),
        "screen_valid_fraction": validity_fraction(screen_rows, max_cap_hit, max_malformed),
        "prompt_variants": prompt_variant_audit(condition_rows, top_k=top_k),
        "holdout_transfer": holdout_transfer_audit(screen_rows, holdout_rows),
        "antithetic_pairs": antithetic_pair_audit(
            screen_rows,
            top_k=top_k,
            max_cap_hit=max_cap_hit,
            max_malformed=max_malformed,
        ),
    }


def gate_check(check: str, passed: bool, detail: dict[str, Any]) -> dict[str, Any]:
    return {"check": check, "passed": bool(passed), "detail": detail}


def ratio(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    right_f = float(right)
    if right_f == 0.0:
        return None
    return float(left) / right_f


def delta(left: Any, right: Any) -> float | None:
    if left is None or right is None:
        return None
    return float(left) - float(right)


def proposal_gate(
    comparison: dict[str, Any],
    *,
    top_k: int,
    min_candidate_sec_ratio: float = 1.0,
    min_screen_candidate_sec_ratio: float = 1.0,
    min_ensemble_delta: float = 0.0,
    min_screen_top_exact_delta: float = 0.0,
    min_valid_fraction_delta: float = 0.0,
    min_prompt_selection_spearman: float = 0.5,
    min_prompt_selection_spearman_delta: float = 0.0,
    min_transfer_spearman_delta: float = 0.0,
    max_selected_regret_increase: float = 0.0,
) -> dict[str, Any]:
    left = comparison["left"]
    right = comparison["right"]
    screen_top_key = f"screen_top{top_k}_exact_mean"
    left_prompt = left["prompt_variants"].get("selection_spearman")
    right_prompt = right["prompt_variants"].get("selection_spearman")
    left_transfer = left["holdout_transfer"].get("screen_selection_vs_holdout_exact_spearman")
    right_transfer = right["holdout_transfer"].get("screen_selection_vs_holdout_exact_spearman")
    left_regret = left["holdout_transfer"].get("screen_selected_regret")
    right_regret = right["holdout_transfer"].get("screen_selected_regret")
    checks = [
        gate_check(
            "candidate_throughput_not_slower",
            (speed_ratio := ratio(left.get("candidate_sec"), right.get("candidate_sec"))) is not None
            and speed_ratio >= min_candidate_sec_ratio,
            {
                "left": left.get("candidate_sec"),
                "right": right.get("candidate_sec"),
                "ratio": speed_ratio,
                "min_ratio": min_candidate_sec_ratio,
            },
        ),
        gate_check(
            "screen_throughput_not_slower",
            (screen_speed_ratio := ratio(left.get("screen_candidate_sec"), right.get("screen_candidate_sec"))) is not None
            and screen_speed_ratio >= min_screen_candidate_sec_ratio,
            {
                "left": left.get("screen_candidate_sec"),
                "right": right.get("screen_candidate_sec"),
                "ratio": screen_speed_ratio,
                "min_ratio": min_screen_candidate_sec_ratio,
            },
        ),
        gate_check(
            "ensemble_quality_not_worse",
            (ensemble_delta := delta(left.get("best_ensemble_holdout_exact"), right.get("best_ensemble_holdout_exact")))
            is not None
            and ensemble_delta >= min_ensemble_delta,
            {
                "left": left.get("best_ensemble_holdout_exact"),
                "right": right.get("best_ensemble_holdout_exact"),
                "delta": ensemble_delta,
                "min_delta": min_ensemble_delta,
            },
        ),
        gate_check(
            f"screen_top{top_k}_quality_not_worse",
            (screen_top_delta := delta(left.get(screen_top_key), right.get(screen_top_key))) is not None
            and screen_top_delta >= min_screen_top_exact_delta,
            {
                "left": left.get(screen_top_key),
                "right": right.get(screen_top_key),
                "delta": screen_top_delta,
                "min_delta": min_screen_top_exact_delta,
            },
        ),
        gate_check(
            "screen_valid_fraction_not_worse",
            (valid_delta := delta(left.get("screen_valid_fraction"), right.get("screen_valid_fraction"))) is not None
            and valid_delta >= min_valid_fraction_delta,
            {
                "left": left.get("screen_valid_fraction"),
                "right": right.get("screen_valid_fraction"),
                "delta": valid_delta,
                "min_delta": min_valid_fraction_delta,
            },
        ),
        gate_check(
            "prompt_selection_rank_stable",
            left_prompt is not None and float(left_prompt) >= min_prompt_selection_spearman,
            {
                "left": left_prompt,
                "min_spearman": min_prompt_selection_spearman,
            },
        ),
        gate_check(
            "prompt_selection_rank_not_worse_than_control",
            (prompt_delta := delta(left_prompt, right_prompt)) is not None
            and prompt_delta >= min_prompt_selection_spearman_delta,
            {
                "left": left_prompt,
                "right": right_prompt,
                "delta": prompt_delta,
                "min_delta": min_prompt_selection_spearman_delta,
            },
        ),
        gate_check(
            "screen_to_holdout_transfer_not_worse_than_control",
            (transfer_delta := delta(left_transfer, right_transfer)) is not None
            and transfer_delta >= min_transfer_spearman_delta,
            {
                "left": left_transfer,
                "right": right_transfer,
                "delta": transfer_delta,
                "min_delta": min_transfer_spearman_delta,
            },
        ),
        gate_check(
            "screen_selected_regret_not_worse_than_control",
            (regret_delta := delta(left_regret, right_regret)) is not None
            and regret_delta <= max_selected_regret_increase,
            {
                "left": left_regret,
                "right": right_regret,
                "delta": regret_delta,
                "max_increase": max_selected_regret_increase,
            },
        ),
    ]
    failed = [row["check"] for row in checks if not row["passed"]]
    return {
        "pass": not failed,
        "failed": failed,
        "checks": checks,
        "thresholds": {
            "top_k": top_k,
            "min_candidate_sec_ratio": min_candidate_sec_ratio,
            "min_screen_candidate_sec_ratio": min_screen_candidate_sec_ratio,
            "min_ensemble_delta": min_ensemble_delta,
            "min_screen_top_exact_delta": min_screen_top_exact_delta,
            "min_valid_fraction_delta": min_valid_fraction_delta,
            "min_prompt_selection_spearman": min_prompt_selection_spearman,
            "min_prompt_selection_spearman_delta": min_prompt_selection_spearman_delta,
            "min_transfer_spearman_delta": min_transfer_spearman_delta,
            "max_selected_regret_increase": max_selected_regret_increase,
        },
    }


def compare_summaries(
    left: dict[str, Any],
    right: dict[str, Any],
    *,
    left_name: str,
    right_name: str,
    top_k: int = 16,
    gate_config: dict[str, float] | None = None,
) -> dict[str, Any]:
    screen_selection_key = f"screen_top{top_k}_selection_mean"
    screen_exact_key = f"screen_top{top_k}_exact_mean"
    keys = [
        "candidate_sec",
        "screen_candidate_sec",
        "best_ensemble_holdout_exact",
        "screen_valid_fraction",
        screen_selection_key,
        screen_exact_key,
    ]
    deltas = {}
    for key in keys:
        if left.get(key) is not None and right.get(key) is not None:
            deltas[f"{key}_{left_name}_minus_{right_name}"] = float(left[key]) - float(right[key])
    comparison = {
        "kind": "proposal_audit_comparison",
        "top_k": top_k,
        "left_name": left_name,
        "right_name": right_name,
        "left": left,
        "right": right,
        "delta": deltas,
    }
    comparison["gate"] = proposal_gate(comparison, top_k=top_k, **(gate_config or {}))
    return comparison


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def write_report(path: Path, comparison: dict[str, Any]) -> None:
    left_name = comparison["left_name"]
    right_name = comparison["right_name"]
    top_k = int(comparison.get("top_k", 16))
    left = comparison["left"]
    right = comparison["right"]
    delta = comparison["delta"]
    gate = comparison["gate"]
    screen_selection_key = f"screen_top{top_k}_selection_mean"
    screen_exact_key = f"screen_top{top_k}_exact_mean"

    def row(label: str, key: str) -> str:
        return f"| {label} | {fmt(left.get(key))} | {fmt(right.get(key))} | {fmt(delta.get(f'{key}_{left_name}_minus_{right_name}'))} |"

    lines = [
        "# Proposal Audit",
        "",
        f"Scale gate: **{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        f"Compared `{left_name}` against `{right_name}` using saved run artifacts.",
        "",
        "| Metric | " + left_name + " | " + right_name + " | delta |",
        "| --- | ---: | ---: | ---: |",
        row("candidate/sec", "candidate_sec"),
        row("screen candidate/sec", "screen_candidate_sec"),
        row("best ensemble holdout exact", "best_ensemble_holdout_exact"),
        row("screen valid fraction", "screen_valid_fraction"),
        row(f"screen top{top_k} selection mean", screen_selection_key),
        row(f"screen top{top_k} exact mean", screen_exact_key),
        "",
        "## Scale Gate",
        "",
        "| Check | pass | detail |",
        "| --- | ---: | --- |",
    ]
    for check in gate["checks"]:
        detail = json.dumps(check["detail"], sort_keys=True)
        lines.append(f"| {check['check']} | {check['passed']} | `{detail}` |")
    lines.extend(
        [
            "",
            "Failed checks: " + (", ".join(gate["failed"]) if gate["failed"] else "none"),
        "",
        "## Prompt Variant Stability",
        "",
        "| Metric | " + left_name + " | " + right_name + " |",
        "| --- | ---: | ---: |",
        ]
    )
    for key in [
        "selection_spearman",
        "exact_spearman",
        "selection_mean_abs_delta",
        "exact_mean_abs_delta",
        f"top{top_k}_selection_overlap",
        "common_candidates",
    ]:
        lines.append(f"| {key} | {fmt(left['prompt_variants'].get(key))} | {fmt(right['prompt_variants'].get(key))} |")

    lines.extend(
        [
            "",
            "## Screen To Holdout Transfer",
            "",
            "| Metric | " + left_name + " | " + right_name + " |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in [
        "common_candidates",
        "screen_selection_vs_holdout_exact_spearman",
        "screen_exact_vs_holdout_exact_spearman",
        "screen_selected_regret",
        "screen_selected_holdout_exact",
        "best_holdout_exact",
    ]:
        lines.append(f"| {key} | {fmt(left['holdout_transfer'].get(key))} | {fmt(right['holdout_transfer'].get(key))} |")

    lines.extend(
        [
            "",
            "## Antithetic Pair Structure",
            "",
            "| Metric | " + left_name + " | " + right_name + " |",
            "| --- | ---: | ---: |",
        ]
    )
    for key in [
        "pairs",
        "pair_best_selection_top_mean",
        "pair_best_exact_top_mean",
        "pair_score_gap_mean",
        "pair_one_valid_one_invalid_fraction",
        "pair_all_valid_fraction",
    ]:
        lines.append(f"| {key} | {fmt(left['antithetic_pairs'].get(key))} | {fmt(right['antithetic_pairs'].get(key))} |")

    path.write_text("\n".join(lines) + "\n")


def run(args: argparse.Namespace) -> dict[str, Any]:
    left = summarize_run(Path(args.left), top_k=args.top_k, max_cap_hit=args.max_cap_hit, max_malformed=args.max_malformed)
    right = summarize_run(
        Path(args.right),
        top_k=args.top_k,
        max_cap_hit=args.max_cap_hit,
        max_malformed=args.max_malformed,
    )
    gate_config = {
        "min_candidate_sec_ratio": args.min_candidate_sec_ratio,
        "min_screen_candidate_sec_ratio": args.min_screen_candidate_sec_ratio,
        "min_ensemble_delta": args.min_ensemble_delta,
        "min_screen_top_exact_delta": args.min_screen_top_exact_delta,
        "min_valid_fraction_delta": args.min_valid_fraction_delta,
        "min_prompt_selection_spearman": args.min_prompt_selection_spearman,
        "min_prompt_selection_spearman_delta": args.min_prompt_selection_spearman_delta,
        "min_transfer_spearman_delta": args.min_transfer_spearman_delta,
        "max_selected_regret_increase": args.max_selected_regret_increase,
    }
    comparison = compare_summaries(
        left,
        right,
        left_name=args.left_name,
        right_name=args.right_name,
        top_k=args.top_k,
        gate_config=gate_config,
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.json").write_text(json.dumps(comparison, indent=2, sort_keys=True) + "\n")
    write_report(out / "report.md", comparison)
    return comparison


def main() -> None:
    p = argparse.ArgumentParser(description="Audit proposal search runs for transfer, prompt stability, and validity.")
    p.add_argument("--left", required=True)
    p.add_argument("--right", required=True)
    p.add_argument("--left-name", default="left")
    p.add_argument("--right-name", default="right")
    p.add_argument("--out", required=True)
    p.add_argument("--top-k", type=int, default=16)
    p.add_argument("--max-cap-hit", type=float, default=0.05)
    p.add_argument("--max-malformed", type=float, default=0.05)
    p.add_argument("--min-candidate-sec-ratio", type=float, default=1.0)
    p.add_argument("--min-screen-candidate-sec-ratio", type=float, default=1.0)
    p.add_argument("--min-ensemble-delta", type=float, default=0.0)
    p.add_argument("--min-screen-top-exact-delta", type=float, default=0.0)
    p.add_argument("--min-valid-fraction-delta", type=float, default=0.0)
    p.add_argument("--min-prompt-selection-spearman", type=float, default=0.5)
    p.add_argument("--min-prompt-selection-spearman-delta", type=float, default=0.0)
    p.add_argument("--min-transfer-spearman-delta", type=float, default=0.0)
    p.add_argument("--max-selected-regret-increase", type=float, default=0.0)
    run(p.parse_args())


if __name__ == "__main__":
    main()
