#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def _is_png(path: Path) -> bool:
    return path.exists() and path.read_bytes().startswith(PNG_HEADER)


def _as_float(row: dict[str, str], key: str) -> float | None:
    raw = row.get(key)
    if raw in {None, "", "None"}:
        return None
    return float(raw)


def _check_file(path: Path, failures: list[str]) -> None:
    if not path.exists():
        failures.append(f"missing required file: {path}")


def _check_png(path: Path, failures: list[str]) -> None:
    if not _is_png(path):
        failures.append(f"missing or malformed PNG: {path}")


def _parity_section(report: dict[str, Any], key: str) -> dict[str, Any] | None:
    section = report.get(key)
    return section if isinstance(section, dict) else None


def validate(args: argparse.Namespace) -> dict[str, Any]:
    plot_dir = Path(args.plot_dir)
    failures: list[str] = []
    warnings: list[str] = []

    summary_csv = plot_dir / "summary.csv"
    parity_json = plot_dir / "parity_summary.json"
    _check_file(summary_csv, failures)
    _check_file(parity_json, failures)
    _check_png(plot_dir / "quality.png", failures)
    _check_png(plot_dir / "throughput.png", failures)
    _check_png(plot_dir / "lazy_timing_breakdown.png", failures)

    rows: list[dict[str, str]] = []
    if summary_csv.exists():
        rows = _read_csv(summary_csv)
        if not rows:
            failures.append("summary.csv has no runs")
        for idx, row in enumerate(rows):
            label = row.get("label") or f"row_{idx}"
            base = _as_float(row, "base_score")
            best = _as_float(row, "best_score")
            cand_sec = _as_float(row, "candidate_sec")
            if base is None or best is None:
                failures.append(f"{label}: base_score/best_score is missing or nonnumeric")
            if cand_sec is None or cand_sec <= 0:
                failures.append(f"{label}: candidate_sec is missing or nonpositive")
            run_dir = row.get("run_dir")
            if run_dir and not (Path(run_dir) / "summary.json").exists():
                failures.append(f"{label}: run_dir does not contain summary.json: {run_dir}")

    lazy_rows = [row for row in rows if "lazy" in (row.get("kind") or "")]
    if not lazy_rows:
        failures.append("no lazy backend runs found in summary.csv")
    if args.require_p1024:
        p1024 = [row for row in lazy_rows if str(row.get("population")) == "1024"]
        if not p1024:
            failures.append("no lazy p1024 run found in summary.csv")
        elif args.require_positive_p1024:
            for row in p1024:
                base = _as_float(row, "base_score")
                best = _as_float(row, "best_score")
                if base is None or best is None or best <= base:
                    failures.append(f"{row.get('label', 'p1024')}: p1024 best_score is not above base_score")

    parity: dict[str, Any] = {}
    if parity_json.exists():
        parity = _read_json(parity_json)
        candidate = _parity_section(parity, "candidate_scores")
        confirmed = _parity_section(parity, "confirmed_candidate_scores")
        if candidate is None:
            failures.append("parity_summary.json missing candidate_scores section")
        else:
            common = int(candidate.get("common_candidates") or 0)
            if common < int(args.min_common_candidates):
                failures.append(f"candidate parity has only {common} common candidates")
            if candidate.get("plot"):
                _check_png(plot_dir / str(candidate["plot"]), failures)
            max_abs = candidate.get("max_abs_score_diff")
            if args.max_candidate_score_diff is not None and (max_abs is None or float(max_abs) > args.max_candidate_score_diff):
                failures.append(
                    "candidate parity max_abs_score_diff "
                    f"{max_abs} exceeds {args.max_candidate_score_diff}"
                )
        if args.require_confirmed_parity and confirmed is None:
            failures.append("parity_summary.json missing confirmed_candidate_scores section")
        if confirmed is not None:
            common = int(confirmed.get("common_candidates") or 0)
            if common == 0:
                warnings.append("confirmed parity has no common candidates")
            if confirmed.get("plot"):
                _check_png(plot_dir / str(confirmed["plot"]), failures)
            max_abs = confirmed.get("max_abs_score_diff")
            if args.max_confirmed_score_diff is not None and (max_abs is None or float(max_abs) > args.max_confirmed_score_diff):
                failures.append(
                    "confirmed parity max_abs_score_diff "
                    f"{max_abs} exceeds {args.max_confirmed_score_diff}"
                )

    return {
        "schema_version": "lazy_kernel_validation_report_v1",
        "plot_dir": str(plot_dir),
        "status": "fail" if failures else "pass",
        "failures": failures,
        "warnings": warnings,
        "summary_rows": len(rows),
        "lazy_rows": len(lazy_rows),
        "parity": parity,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate lazy-kernel parity/ablation plot artifacts.")
    parser.add_argument("--plot-dir", required=True, type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--min-common-candidates", type=int, default=1)
    parser.add_argument("--max-candidate-score-diff", type=float)
    parser.add_argument("--max-confirmed-score-diff", type=float)
    parser.add_argument("--require-p1024", action="store_true")
    parser.add_argument("--require-positive-p1024", action="store_true")
    parser.add_argument("--require-confirmed-parity", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    report = validate(args)
    out = args.out or (args.plot_dir / "validation_report.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
