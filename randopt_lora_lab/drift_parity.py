from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


DRIFT_KIND = "logit_drift"


@dataclass
class DriftCheck:
    check: str
    passed: bool
    detail: Any


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


def finite_nonnegative(values: list[float], *, atol: float = 1e-9) -> bool:
    return all(math.isfinite(value) and value >= -atol for value in values)


def row_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if key in row and row[key] is not None]


def summary_value(summary: dict[str, Any], key: str) -> float | None:
    value = summary.get(key)
    return None if value is None else float(value)


def ratio(candidate: float | None, reference: float | None) -> float | None:
    if candidate is None or reference is None or reference == 0.0:
        return None
    return candidate / reference


def check(check: str, passed: bool, detail: Any) -> DriftCheck:
    return DriftCheck(check, bool(passed), detail)


def load_run(path: Path) -> dict[str, Any]:
    summary = read_json(path / "summary.json")
    rows = read_jsonl(path / "candidate_drift.jsonl")
    return {"path": str(path), "summary": summary, "rows": rows}


def drift_summary(run: dict[str, Any]) -> dict[str, Any]:
    summary = run["summary"]
    rows = run["rows"]
    return {
        "path": run["path"],
        "kind": summary.get("kind"),
        "family": summary.get("family"),
        "population": summary.get("population"),
        "rank": summary.get("rank"),
        "sigma": summary.get("sigma"),
        "sigma_values": summary.get("sigma_values", []),
        "prompts": summary.get("prompts"),
        "rows": len(rows),
        "kl_base_to_candidate_mean_mean": summary_value(summary, "kl_base_to_candidate_mean_mean"),
        "kl_base_to_candidate_mean_max": summary_value(summary, "kl_base_to_candidate_mean_max"),
        "kl_candidate_to_base_mean_mean": summary_value(summary, "kl_candidate_to_base_mean_mean"),
        "logit_l2_mean_mean": summary_value(summary, "logit_l2_mean_mean"),
        "top1_equal_rate_mean": summary_value(summary, "top1_equal_rate_mean"),
        "top1_equal_rate_min": summary_value(summary, "top1_equal_rate_min"),
    }


def run_drift_parity(
    reference_dir: Path,
    candidate_dir: Path,
    *,
    max_kl_ratio: float = 1.1,
    max_logit_l2_ratio: float = 1.1,
    min_top1_delta: float = -0.01,
    min_rows: int = 1,
    require_same_prompts: bool = True,
) -> dict[str, Any]:
    reference = load_run(reference_dir)
    candidate = load_run(candidate_dir)
    ref_summary = drift_summary(reference)
    cand_summary = drift_summary(candidate)
    ref_rows = reference["rows"]
    cand_rows = candidate["rows"]
    ref_kl = row_values(ref_rows, "kl_base_to_candidate_mean") + row_values(ref_rows, "kl_candidate_to_base_mean")
    cand_kl = row_values(cand_rows, "kl_base_to_candidate_mean") + row_values(cand_rows, "kl_candidate_to_base_mean")
    ref_mean_kl = ref_summary["kl_base_to_candidate_mean_mean"]
    cand_mean_kl = cand_summary["kl_base_to_candidate_mean_mean"]
    ref_l2 = ref_summary["logit_l2_mean_mean"]
    cand_l2 = cand_summary["logit_l2_mean_mean"]
    ref_top1 = ref_summary["top1_equal_rate_mean"]
    cand_top1 = cand_summary["top1_equal_rate_mean"]
    kl_ratio = ratio(cand_mean_kl, ref_mean_kl)
    l2_ratio = ratio(cand_l2, ref_l2)
    top1_delta = None if cand_top1 is None or ref_top1 is None else cand_top1 - ref_top1
    checks = [
        check(
            "reference_is_logit_drift",
            ref_summary["kind"] == DRIFT_KIND,
            {"kind": ref_summary["kind"], "path": ref_summary["path"]},
        ),
        check(
            "candidate_is_logit_drift",
            cand_summary["kind"] == DRIFT_KIND,
            {"kind": cand_summary["kind"], "path": cand_summary["path"]},
        ),
        check(
            "rows_present",
            len(ref_rows) >= min_rows and len(cand_rows) >= min_rows,
            {"reference_rows": len(ref_rows), "candidate_rows": len(cand_rows), "min_rows": min_rows},
        ),
        check(
            "same_prompt_count",
            (not require_same_prompts) or ref_summary["prompts"] == cand_summary["prompts"],
            {"reference_prompts": ref_summary["prompts"], "candidate_prompts": cand_summary["prompts"]},
        ),
        check(
            "kl_nonnegative",
            finite_nonnegative(ref_kl) and finite_nonnegative(cand_kl),
            {
                "reference_min_kl": min(ref_kl) if ref_kl else None,
                "candidate_min_kl": min(cand_kl) if cand_kl else None,
            },
        ),
        check(
            "candidate_mean_kl_not_higher",
            kl_ratio is not None and kl_ratio <= max_kl_ratio,
            {
                "reference": ref_mean_kl,
                "candidate": cand_mean_kl,
                "ratio": kl_ratio,
                "max_ratio": max_kl_ratio,
            },
        ),
        check(
            "candidate_logit_l2_not_higher",
            l2_ratio is not None and l2_ratio <= max_logit_l2_ratio,
            {
                "reference": ref_l2,
                "candidate": cand_l2,
                "ratio": l2_ratio,
                "max_ratio": max_logit_l2_ratio,
            },
        ),
        check(
            "candidate_top1_not_worse",
            top1_delta is not None and top1_delta >= min_top1_delta,
            {
                "reference": ref_top1,
                "candidate": cand_top1,
                "delta": top1_delta,
                "min_delta": min_top1_delta,
            },
        ),
    ]
    rows = [asdict(item) for item in checks]
    failed = [row["check"] for row in rows if not row["passed"]]
    return {
        "kind": "drift_parity",
        "pass": not failed,
        "failed": failed,
        "reference": ref_summary,
        "candidate": cand_summary,
        "comparison": {
            "kl_base_to_candidate_mean_ratio": kl_ratio,
            "logit_l2_mean_ratio": l2_ratio,
            "top1_equal_rate_delta": top1_delta,
        },
        "thresholds": {
            "max_kl_ratio": max_kl_ratio,
            "max_logit_l2_ratio": max_logit_l2_ratio,
            "min_top1_delta": min_top1_delta,
            "min_rows": min_rows,
            "require_same_prompts": require_same_prompts,
        },
        "checks": rows,
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_markdown(summary: dict[str, Any]) -> str:
    ref = summary["reference"]
    cand = summary["candidate"]
    comp = summary["comparison"]
    lines = [
        "# Drift Parity Report",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
        "",
        "| metric | reference | candidate | comparison |",
        "| --- | ---: | ---: | ---: |",
        f"| family | {ref['family']} | {cand['family']} |  |",
        f"| rows | {ref['rows']} | {cand['rows']} |  |",
        f"| prompts | {ref['prompts']} | {cand['prompts']} |  |",
        f"| KL(base||candidate) mean | {fmt(ref['kl_base_to_candidate_mean_mean'])} | {fmt(cand['kl_base_to_candidate_mean_mean'])} | ratio {fmt(comp['kl_base_to_candidate_mean_ratio'])} |",
        f"| logit L2 mean | {fmt(ref['logit_l2_mean_mean'])} | {fmt(cand['logit_l2_mean_mean'])} | ratio {fmt(comp['logit_l2_mean_ratio'])} |",
        f"| top-1 equal mean | {fmt(ref['top1_equal_rate_mean'])} | {fmt(cand['top1_equal_rate_mean'])} | delta {fmt(comp['top1_equal_rate_delta'])} |",
        "",
        "## Checks",
        "",
        "| check | pass | detail |",
        "| --- | ---: | --- |",
    ]
    for row in summary["checks"]:
        lines.append(f"| {row['check']} | {row['passed']} | `{json.dumps(row['detail'], sort_keys=True)}` |")
    lines.append("")
    lines.append("Failed checks: " + (", ".join(summary["failed"]) if summary["failed"] else "none"))
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare two logit-drift runs as a dense-vs-candidate drift parity gate.")
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--max-kl-ratio", type=float, default=1.1)
    parser.add_argument("--max-logit-l2-ratio", type=float, default=1.1)
    parser.add_argument("--min-top1-delta", type=float, default=-0.01)
    parser.add_argument("--min-rows", type=int, default=1)
    parser.add_argument("--allow-different-prompts", action="store_true")
    args = parser.parse_args(argv)

    summary = run_drift_parity(
        args.reference,
        args.candidate,
        max_kl_ratio=args.max_kl_ratio,
        max_logit_l2_ratio=args.max_logit_l2_ratio,
        min_top1_delta=args.min_top1_delta,
        min_rows=args.min_rows,
        require_same_prompts=not args.allow_different_prompts,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
