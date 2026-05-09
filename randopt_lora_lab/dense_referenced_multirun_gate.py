from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class GateCheck:
    check: str
    passed: bool
    detail: Any


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def read_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def gate_pass(payload: dict[str, Any] | None) -> bool:
    if payload is None:
        return False
    gate = payload.get("gate", payload)
    return bool(gate.get("pass"))


def validity_pass(run_dir: Path, arm: str) -> bool | None:
    payload = read_optional_json(run_dir / arm / "validity" / "summary.json")
    if payload is None:
        return None
    return bool(payload.get("pass"))


def passing_quality_rows(payload: dict[str, Any], *, max_confirm_k: int) -> list[dict[str, Any]]:
    rows = []
    for row in payload.get("rows", []):
        if int(row.get("k", 0)) > max_confirm_k:
            continue
        if bool(row.get("passes_quality")) and bool(row.get("passes_speed")):
            rows.append(row)
    return sorted(rows, key=lambda row: int(row["k"]))


def prompt_variants(vllm_summary: dict[str, Any]) -> list[str]:
    variants = vllm_summary.get("screen_selection_prompt_variants")
    if variants is None:
        variants = vllm_summary.get("prompt_variants") or []
    return [str(variant) for variant in variants]


def _prompt_health_rows(vllm_summary: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for split_key in ["base_screen_by_prompt", "base_holdout_by_prompt"]:
        for variant, row in (vllm_summary.get(split_key) or {}).items():
            rows.append({"split": split_key, "prompt_variant": str(variant), **dict(row)})
    return rows


def prompt_health_pass(
    vllm_summary: dict[str, Any],
    *,
    max_base_cap_hit: float,
    max_base_malformed: float,
    min_base_answer_closed: float,
) -> tuple[bool, dict[str, Any]]:
    rows = _prompt_health_rows(vllm_summary)
    bad_rows = [
        row
        for row in rows
        if float(row.get("cap_hit_mean", 0.0) or 0.0) > max_base_cap_hit
        or float(row.get("malformed_mean", 0.0) or 0.0) > max_base_malformed
        or float(row.get("answer_closed_mean", 0.0) or 0.0) < min_base_answer_closed
    ]
    return (
        not bad_rows and bool(rows),
        {
            "rows": len(rows),
            "bad_rows": bad_rows,
            "max_base_cap_hit": max_base_cap_hit,
            "max_base_malformed": max_base_malformed,
            "min_base_answer_closed": min_base_answer_closed,
        },
    )


def load_run(
    run_dir: Path,
    *,
    max_confirm_k: int,
    max_base_cap_hit: float,
    max_base_malformed: float,
    min_base_answer_closed: float,
) -> dict[str, Any]:
    shortlist = read_optional_json(run_dir / "shortlist_dense_confirmation" / "summary.json")
    search_quality = read_optional_json(run_dir / "search_quality_confirmation" / "summary.json")
    score_sanity = read_optional_json(run_dir / "score_sanity" / "summary.json")
    provenance = read_optional_json(run_dir / "family_state_provenance_audit" / "summary.json")
    replay = read_optional_json(run_dir / "replay_manifest" / "summary.json")
    vllm = read_json(run_dir / "vllm" / "summary.json")
    dense_validity = validity_pass(run_dir, "dense")
    confirmed_validity = validity_pass(run_dir, "confirmed")

    quality_rows = [] if search_quality is None else passing_quality_rows(search_quality, max_confirm_k=max_confirm_k)
    best_quality_row = quality_rows[0] if quality_rows else None
    shortlist_gate = (shortlist or {}).get("gate", {}) if shortlist else {}
    search_quality_gate = (search_quality or {}).get("gate", {}) if search_quality else {}
    prompt_ok, prompt_detail = prompt_health_pass(
        vllm,
        max_base_cap_hit=max_base_cap_hit,
        max_base_malformed=max_base_malformed,
        min_base_answer_closed=min_base_answer_closed,
    )
    variants = prompt_variants(vllm)
    return {
        "run_dir": str(run_dir),
        "dense_validity_pass": dense_validity,
        "confirmed_validity_pass": confirmed_validity,
        "shortlist_dense_pass": gate_pass(shortlist),
        "shortlist_dense_failed": shortlist_gate.get("failed", ["missing"] if shortlist is None else []),
        "zero_dense_regret_k": None if shortlist is None else shortlist.get("zero_dense_regret_k"),
        "dense_best_recovered_k": None if shortlist is None else shortlist.get("dense_best_recovered_k"),
        "dense_best_score": None if shortlist is None else shortlist.get("dense_best_score"),
        "search_quality_pass": gate_pass(search_quality),
        "search_quality_failed": search_quality_gate.get("failed", ["missing"] if search_quality is None else []),
        "quality_k": None if best_quality_row is None else int(best_quality_row["k"]),
        "confirmed_strict_exact": None if best_quality_row is None else best_quality_row.get("confirmed_strict_exact"),
        "dense_strict_exact_at_k": None if best_quality_row is None else best_quality_row.get("dense_strict_exact_at_k"),
        "delta_vs_dense_best_strict": None if best_quality_row is None else best_quality_row.get("delta_vs_dense_best_strict"),
        "full_speedup": None if best_quality_row is None else best_quality_row.get("full_speedup"),
        "eval_only_speedup": None if best_quality_row is None else best_quality_row.get("eval_only_speedup"),
        "score_sanity_pass": bool(score_sanity and score_sanity.get("pass")),
        "score_sanity_failed": [] if score_sanity is None else score_sanity.get("failed", []),
        "family_state_provenance_pass": bool(provenance and provenance.get("pass")),
        "family_state_provenance_failed": [] if provenance is None else provenance.get("failed", []),
        "artifact_complete": bool(replay and replay.get("artifact_complete")),
        "missing_required": [] if replay is None else replay.get("missing_required", []),
        "prompt_variants": variants,
        "prompt_variant_count": len(variants),
        "require_all_prompt_variants_valid": bool(vllm.get("require_all_prompt_variants_valid")),
        "prompt_health_pass": prompt_ok,
        "prompt_health_detail": prompt_detail,
        "candidate_sec": vllm.get("candidate_sec"),
    }


def numeric_values(rows: list[dict[str, Any]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def aggregate(
    run_dirs: list[Path],
    *,
    min_runs: int = 2,
    min_prompt_variants: int = 2,
    max_confirm_k: int = 4,
    min_full_speedup: float = 1.0,
    max_base_cap_hit: float = 0.05,
    max_base_malformed: float = 0.05,
    min_base_answer_closed: float = 1.0,
) -> dict[str, Any]:
    runs = [
        load_run(
            path,
            max_confirm_k=max_confirm_k,
            max_base_cap_hit=max_base_cap_hit,
            max_base_malformed=max_base_malformed,
            min_base_answer_closed=min_base_answer_closed,
        )
        for path in run_dirs
    ]
    full_speedups = numeric_values(runs, "full_speedup")
    strict_deltas = numeric_values(runs, "delta_vs_dense_best_strict")
    checks = [
        GateCheck("min_runs", len(runs) >= min_runs, {"runs": len(runs), "min_runs": min_runs}),
        GateCheck(
            "all_validity_pass",
            all(row["dense_validity_pass"] is True and row["confirmed_validity_pass"] is True for row in runs),
            {
                row["run_dir"]: {
                    "dense": row["dense_validity_pass"],
                    "confirmed": row["confirmed_validity_pass"],
                }
                for row in runs
            },
        ),
        GateCheck(
            "all_shortlist_dense_pass",
            all(row["shortlist_dense_pass"] for row in runs),
            {
                row["run_dir"]: {
                    "pass": row["shortlist_dense_pass"],
                    "failed": row["shortlist_dense_failed"],
                    "zero_dense_regret_k": row["zero_dense_regret_k"],
                    "dense_best_recovered_k": row["dense_best_recovered_k"],
                }
                for row in runs
            },
        ),
        GateCheck(
            "all_search_quality_pass",
            all(row["search_quality_pass"] and row["quality_k"] is not None for row in runs),
            {
                row["run_dir"]: {
                    "pass": row["search_quality_pass"],
                    "failed": row["search_quality_failed"],
                    "quality_k": row["quality_k"],
                    "delta_vs_dense_best_strict": row["delta_vs_dense_best_strict"],
                }
                for row in runs
            },
        ),
        GateCheck(
            "positive_full_speedup",
            len(full_speedups) == len(runs) and bool(full_speedups) and min(full_speedups) >= min_full_speedup,
            {"min_full_speedup": min(full_speedups) if full_speedups else None, "threshold": min_full_speedup},
        ),
        GateCheck(
            "all_score_sanity_pass",
            all(row["score_sanity_pass"] for row in runs),
            {row["run_dir"]: row["score_sanity_failed"] for row in runs},
        ),
        GateCheck(
            "all_family_state_provenance_pass",
            all(row["family_state_provenance_pass"] for row in runs),
            {row["run_dir"]: row["family_state_provenance_failed"] for row in runs},
        ),
        GateCheck(
            "all_artifacts_complete",
            all(row["artifact_complete"] and not row["missing_required"] for row in runs),
            {row["run_dir"]: {"artifact_complete": row["artifact_complete"], "missing_required": row["missing_required"]} for row in runs},
        ),
        GateCheck(
            "prompt_robust_selection",
            all(
                row["prompt_variant_count"] >= min_prompt_variants
                and row["require_all_prompt_variants_valid"]
                and row["prompt_health_pass"]
                for row in runs
            ),
            {
                row["run_dir"]: {
                    "prompt_variant_count": row["prompt_variant_count"],
                    "prompt_variants": row["prompt_variants"],
                    "require_all_prompt_variants_valid": row["require_all_prompt_variants_valid"],
                    "prompt_health_pass": row["prompt_health_pass"],
                    "prompt_health_detail": row["prompt_health_detail"],
                    "min_prompt_variants": min_prompt_variants,
                }
                for row in runs
            },
        ),
    ]
    failed = [check.check for check in checks if not check.passed]
    return {
        "kind": "dense_referenced_multirun_gate",
        "pass": not failed,
        "failed": failed,
        "thresholds": {
            "min_runs": min_runs,
            "min_prompt_variants": min_prompt_variants,
            "max_confirm_k": max_confirm_k,
            "min_full_speedup": min_full_speedup,
            "max_base_cap_hit": max_base_cap_hit,
            "max_base_malformed": max_base_malformed,
            "min_base_answer_closed": min_base_answer_closed,
        },
        "aggregate": {
            "runs": len(runs),
            "shortlist_dense_pass_count": sum(1 for row in runs if row["shortlist_dense_pass"]),
            "search_quality_pass_count": sum(1 for row in runs if row["search_quality_pass"]),
            "validity_pass_count": sum(
                1 for row in runs if row["dense_validity_pass"] is True and row["confirmed_validity_pass"] is True
            ),
            "score_sanity_pass_count": sum(1 for row in runs if row["score_sanity_pass"]),
            "family_state_provenance_pass_count": sum(1 for row in runs if row["family_state_provenance_pass"]),
            "artifact_complete_count": sum(1 for row in runs if row["artifact_complete"] and not row["missing_required"]),
            "prompt_robust_count": sum(
                1
                for row in runs
                if row["prompt_variant_count"] >= min_prompt_variants
                and row["require_all_prompt_variants_valid"]
                and row["prompt_health_pass"]
            ),
            "min_full_speedup": min(full_speedups) if full_speedups else None,
            "mean_full_speedup": mean(full_speedups) if full_speedups else None,
            "min_delta_vs_dense_best_strict": min(strict_deltas) if strict_deltas else None,
            "mean_delta_vs_dense_best_strict": mean(strict_deltas) if strict_deltas else None,
        },
        "checks": [asdict(check) for check in checks],
        "runs": runs,
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Dense-Referenced Multi-Run Gate",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
        "",
        "This gate aggregates repeated corrected q-proj shortlist confirmations. It is not a dense ranking-correlation claim.",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary["aggregate"].items():
        lines.append(f"| {key} | {value if value is not None else 'null'} |")
    lines.extend(["", "## Gates", "", "| gate | pass | detail |", "| --- | ---: | --- |"])
    for check in summary["checks"]:
        lines.append(f"| {check['check']} | {str(check['passed']).lower()} | `{json.dumps(check['detail'], sort_keys=True)}` |")
    lines.extend(
        [
            "",
            "## Runs",
            "",
            "| run | quality k | zero dense regret k | strict delta | full speedup | prompts |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["runs"]:
        lines.append(
            f"| `{row['run_dir']}` | {row['quality_k']} | {row['zero_dense_regret_k']} | "
            f"{row['delta_vs_dense_best_strict']} | {row['full_speedup']} | {','.join(row['prompt_variants'])} |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate corrected dense-referenced q-proj confirmation runs.")
    parser.add_argument("--run", action="append", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--min-prompt-variants", type=int, default=2)
    parser.add_argument("--max-confirm-k", type=int, default=4)
    parser.add_argument("--min-full-speedup", type=float, default=1.0)
    parser.add_argument("--max-base-cap-hit", type=float, default=0.05)
    parser.add_argument("--max-base-malformed", type=float, default=0.05)
    parser.add_argument("--min-base-answer-closed", type=float, default=1.0)
    args = parser.parse_args(argv)

    summary = aggregate(
        args.run,
        min_runs=args.min_runs,
        min_prompt_variants=args.min_prompt_variants,
        max_confirm_k=args.max_confirm_k,
        min_full_speedup=args.min_full_speedup,
        max_base_cap_hit=args.max_base_cap_hit,
        max_base_malformed=args.max_base_malformed,
        min_base_answer_closed=args.min_base_answer_closed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
