from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def best_exact(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return max(float(row.get("exact_mean", 0.0) or 0.0) for row in rows)


def row_by_k(rows: list[dict[str, Any]]) -> dict[int, dict[str, Any]]:
    return {int(row["k"]): row for row in rows if "k" in row}


def validity_pass(run_dir: Path) -> bool | None:
    path = run_dir / "validity" / "summary.json"
    if not path.exists():
        return None
    return bool(read_json(path).get("pass"))


def analyze(root: Path, *, max_confirm_k: int = 8, min_full_speedup: float = 1.0, min_holdout_delta: float = 0.0) -> dict[str, Any]:
    dense_dir = root / "dense"
    confirmed_dir = root / "confirmed"
    speed_dir = root / "shortlist_dense_confirmation"
    dense = read_json(dense_dir / "summary.json")
    confirmed = read_json(confirmed_dir / "summary.json")
    speed = read_json(speed_dir / "summary.json")

    dense_strict = row_by_k(dense.get("strict_ensemble_holdout", []))
    confirmed_strict = row_by_k(confirmed.get("strict_ensemble_holdout", []))
    speed_rows = {int(row["k"]): row for row in speed.get("rows", [])}

    dense_best_strict = best_exact(list(dense_strict.values()))
    dense_best_screen = best_exact(dense.get("top_screen", []))
    confirmed_best_screen = best_exact(confirmed.get("top_screen", []))

    rows = []
    for k in sorted(set(confirmed_strict) & set(speed_rows)):
        if k > max_confirm_k:
            continue
        strict_row = confirmed_strict[k]
        speed_row = speed_rows[k]
        confirmed_strict_exact = float(strict_row.get("exact_mean", 0.0) or 0.0)
        dense_at_k = float(dense_strict.get(k, {}).get("exact_mean", 0.0) or 0.0) if k in dense_strict else None
        delta_vs_dense_best = None if dense_best_strict is None else confirmed_strict_exact - dense_best_strict
        full_speedup = float(speed_row.get("full_without_dense_load_speedup_vs_dense_full", 0.0) or 0.0)
        rows.append(
            {
                "k": k,
                "confirmed_strict_exact": confirmed_strict_exact,
                "dense_strict_exact_at_k": dense_at_k,
                "delta_vs_dense_best_strict": delta_vs_dense_best,
                "delta_vs_dense_same_k": None if dense_at_k is None else confirmed_strict_exact - dense_at_k,
                "full_speedup": full_speedup,
                "eval_only_speedup": float(speed_row.get("eval_only_speedup_vs_dense_full", 0.0) or 0.0),
                "passes_quality": delta_vs_dense_best is not None and delta_vs_dense_best >= min_holdout_delta,
                "passes_speed": full_speedup >= min_full_speedup,
            }
        )

    dense_valid = validity_pass(dense_dir)
    confirmed_valid = validity_pass(confirmed_dir)
    passing_rows = [row for row in rows if row["passes_quality"] and row["passes_speed"]]
    gate_checks = [
        {
            "check": "dense_validity_pass",
            "passed": dense_valid is True,
            "detail": {"pass": dense_valid},
        },
        {
            "check": "confirmed_validity_pass",
            "passed": confirmed_valid is True,
            "detail": {"pass": confirmed_valid},
        },
        {
            "check": "strict_holdout_quality_at_speed",
            "passed": bool(passing_rows),
            "detail": {
                "max_confirm_k": max_confirm_k,
                "min_holdout_delta": min_holdout_delta,
                "min_full_speedup": min_full_speedup,
                "passing_k": None if not passing_rows else passing_rows[0]["k"],
            },
        },
    ]
    failed = [row["check"] for row in gate_checks if not row["passed"]]
    return {
        "kind": "search_quality_confirmation",
        "run_root": str(root),
        "dense_best_screen_exact": dense_best_screen,
        "confirmed_best_screen_exact": confirmed_best_screen,
        "screen_delta_vs_dense": None
        if dense_best_screen is None or confirmed_best_screen is None
        else confirmed_best_screen - dense_best_screen,
        "dense_best_strict_holdout_exact": dense_best_strict,
        "confirmed_best_strict_holdout_exact": best_exact(list(confirmed_strict.values())),
        "rows": rows,
        "gate": {
            "pass": not failed,
            "failed": failed,
            "checks": gate_checks,
            "thresholds": {
                "max_confirm_k": max_confirm_k,
                "min_full_speedup": min_full_speedup,
                "min_holdout_delta": min_holdout_delta,
            },
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    gate = summary["gate"]
    lines = [
        "# Search Quality Confirmation",
        "",
        f"Gate: **{'PASS' if gate['pass'] else 'FAIL'}**",
        "",
        "| metric | value |",
        "| --- | ---: |",
        f"| dense best screen exact | {fmt(summary['dense_best_screen_exact'])} |",
        f"| confirmed best screen exact | {fmt(summary['confirmed_best_screen_exact'])} |",
        f"| screen delta vs dense | {fmt(summary['screen_delta_vs_dense'])} |",
        f"| dense best strict holdout | {fmt(summary['dense_best_strict_holdout_exact'])} |",
        f"| confirmed best strict holdout | {fmt(summary['confirmed_best_strict_holdout_exact'])} |",
        "",
        "## Rows",
        "",
        "| k | confirmed strict | dense strict at k | delta vs dense best | delta vs same k | full speedup | pass quality | pass speed |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for row in summary["rows"]:
        lines.append(
            f"| {row['k']} | {fmt(row['confirmed_strict_exact'])} | {fmt(row['dense_strict_exact_at_k'])} | "
            f"{fmt(row['delta_vs_dense_best_strict'])} | {fmt(row['delta_vs_dense_same_k'])} | "
            f"{fmt(row['full_speedup'])} | {str(row['passes_quality']).lower()} | {str(row['passes_speed']).lower()} |"
        )
    lines.extend(
        [
            "",
            "## Gate Checks",
            "",
            "| check | pass | detail |",
            "| --- | --- | --- |",
        ]
    )
    for check in gate["checks"]:
        lines.append(f"| {check['check']} | {str(check['passed']).lower()} | `{json.dumps(check['detail'], sort_keys=True)}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Confirm accelerated search quality against dense strict holdout and speed.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-confirm-k", type=int, default=8)
    parser.add_argument("--min-full-speedup", type=float, default=1.0)
    parser.add_argument("--min-holdout-delta", type=float, default=0.0)
    args = parser.parse_args(argv)

    summary = analyze(
        args.root,
        max_confirm_k=args.max_confirm_k,
        min_full_speedup=args.min_full_speedup,
        min_holdout_delta=args.min_holdout_delta,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
