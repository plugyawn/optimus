from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from .countdown import CountdownExample, score_completion, semantic_example_key


@dataclass
class ValidityCheck:
    check: str
    passed: bool
    detail: Any


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def bool_check(check: str, passed: bool, detail: Any) -> ValidityCheck:
    return ValidityCheck(check, bool(passed), detail)


def row_example(row: dict) -> CountdownExample | None:
    if not {"example_id", "numbers", "target"}.issubset(row):
        return None
    return CountdownExample(int(row["example_id"]), tuple(int(x) for x in row["numbers"]), int(row["target"]))


def mode_rows(rows: list[dict], mode: str) -> list[dict]:
    return [row for row in rows if row.get("mode") == mode]


def prompt_signature(row: dict) -> tuple[tuple[int, ...], int] | None:
    ex = row_example(row)
    if ex is None:
        return None
    return semantic_example_key(ex)


def strict_rescore_mismatches(rows: list[dict]) -> list[dict]:
    mismatches = []
    fields = ["exact", "malformed", "missing_answer", "multiple_answers", "trailing_text", "answer_count"]
    for idx, row in enumerate(rows):
        if "text" not in row:
            continue
        ex = row_example(row)
        if ex is None:
            continue
        rescored = score_completion(str(row["text"]), ex, strict=True)
        changed = {}
        for field in fields:
            if field not in row:
                continue
            old = row[field]
            new = rescored[field]
            if isinstance(old, float) or isinstance(new, float):
                same = float(old) == float(new)
            else:
                same = old == new
            if not same:
                changed[field] = {"stored": old, "strict": new}
        if changed:
            mismatches.append(
                {
                    "row_index": idx,
                    "mode": row.get("mode"),
                    "candidate": row.get("candidate"),
                    "example_id": row.get("example_id"),
                    "changed": changed,
                }
            )
    return mismatches


def unique_count(values: list[Any]) -> int:
    return len(set(values))


def base_prompt_checks(per_prompt: list[dict], holdout: list[dict]) -> list[ValidityCheck]:
    checks = []
    base_screen = mode_rows(per_prompt, "base_screen")
    base_holdout = mode_rows(holdout, "base_holdout")
    checks.append(bool_check("base_screen_rows_present", bool(base_screen), {"rows": len(base_screen)}))
    checks.append(bool_check("base_holdout_rows_present", bool(base_holdout), {"rows": len(base_holdout)}))
    for label, rows in [("screen", base_screen), ("holdout", base_holdout)]:
        ids = [int(row["example_id"]) for row in rows if "example_id" in row]
        semantics = [sig for row in rows if (sig := prompt_signature(row)) is not None]
        checks.append(
            bool_check(
                f"{label}_base_ids_unique",
                len(ids) == unique_count(ids),
                {"rows": len(ids), "unique": unique_count(ids)},
            )
        )
        checks.append(
            bool_check(
                f"{label}_base_semantics_unique",
                len(semantics) == unique_count(semantics),
                {"rows": len(semantics), "unique": unique_count(semantics)},
            )
        )
    if base_screen and base_holdout:
        screen_ids = {int(row["example_id"]) for row in base_screen if "example_id" in row}
        holdout_ids = {int(row["example_id"]) for row in base_holdout if "example_id" in row}
        screen_sem = {sig for row in base_screen if (sig := prompt_signature(row)) is not None}
        holdout_sem = {sig for row in base_holdout if (sig := prompt_signature(row)) is not None}
        checks.append(
            bool_check(
                "screen_holdout_ids_disjoint",
                not (screen_ids & holdout_ids),
                {"overlap": len(screen_ids & holdout_ids)},
            )
        )
        checks.append(
            bool_check(
                "screen_holdout_semantics_disjoint",
                not (screen_sem & holdout_sem),
                {"overlap": len(screen_sem & holdout_sem)},
            )
        )
    return checks


def summary_prompt_checks(summary: dict) -> list[ValidityCheck]:
    checks = []
    pairs = [
        ("screen_prompts", "screen_unique_prompts"),
        ("holdout_prompts", "holdout_unique_prompts"),
        ("screen_prompts", "screen_unique_semantic_prompts"),
        ("holdout_prompts", "holdout_unique_semantic_prompts"),
    ]
    for total_key, unique_key in pairs:
        if total_key not in summary or unique_key not in summary:
            checks.append(bool_check(f"summary_{unique_key}_present", False, {"missing": [total_key, unique_key]}))
            continue
        checks.append(
            bool_check(
                f"summary_{unique_key}_matches_total",
                int(summary[total_key]) == int(summary[unique_key]),
                {total_key: summary[total_key], unique_key: summary[unique_key]},
            )
        )
    if "screen_holdout_overlap" in summary:
        checks.append(
            bool_check(
                "summary_screen_holdout_overlap_zero",
                int(summary["screen_holdout_overlap"]) == 0,
                {"screen_holdout_overlap": summary["screen_holdout_overlap"]},
            )
        )
    else:
        checks.append(bool_check("summary_screen_holdout_overlap_present", False, "missing"))
    return checks


def protocol_checks(summary: dict, holdout: list[dict], ensemble_per_prompt: list[dict]) -> list[ValidityCheck]:
    checks = []
    checks.append(
        bool_check(
            "candidate_score_metric_exact_answer",
            summary.get("candidate_score_metric") == "exact_answer",
            {"candidate_score_metric": summary.get("candidate_score_metric")},
        )
    )
    ensemble_ks = summary.get("ensemble_ks") or []
    if ensemble_ks:
        checks.append(
            bool_check(
                "ensemble_vote_metric_numeric",
                summary.get("ensemble_vote_metric") == "valid_numeric_majority_vote",
                {"ensemble_vote_metric": summary.get("ensemble_vote_metric")},
            )
        )
        present_ks = {int(row["k"]) for row in ensemble_per_prompt if "k" in row}
        checks.append(
            bool_check(
                "ensemble_per_prompt_rows_present",
                set(int(k) for k in ensemble_ks).issubset(present_ks),
                {"expected_ks": ensemble_ks, "present_ks": sorted(present_ks)},
            )
        )
    holdout_modes = Counter(str(row.get("mode", "")) for row in holdout)
    checks.append(
        bool_check(
            "candidate_holdout_rows_present",
            holdout_modes.get("holdout", 0) > 0,
            {"holdout_mode_counts": dict(holdout_modes)},
        )
    )
    return checks


def selected_candidate_checks(
    summary: dict,
    holdout: list[dict],
    *,
    max_selected_cap_hit: float,
    max_selected_malformed: float,
) -> list[ValidityCheck]:
    candidate_rows = [row for row in holdout if row.get("mode") == "holdout"]
    by_candidate: dict[str, list[dict]] = {}
    for row in candidate_rows:
        by_candidate.setdefault(str(row.get("candidate")), []).append(row)
    cap_hit = {
        candidate: mean(float(row.get("cap_hit", 0.0)) for row in rows)
        for candidate, rows in by_candidate.items()
    }
    malformed = {
        candidate: mean(float(row.get("malformed", 0.0)) for row in rows)
        for candidate, rows in by_candidate.items()
    }
    max_cap = max(cap_hit.values(), default=0.0)
    max_bad = max(malformed.values(), default=0.0)
    checks = [
        bool_check(
            "selected_candidate_cap_hit_below_threshold",
            max_cap <= max_selected_cap_hit,
            {"max": max_cap, "threshold": max_selected_cap_hit, "by_candidate": cap_hit},
        ),
        bool_check(
            "selected_candidate_malformed_below_threshold",
            max_bad <= max_selected_malformed,
            {"max": max_bad, "threshold": max_selected_malformed, "by_candidate": malformed},
        ),
    ]
    top_holdout = summary.get("top_holdout") or []
    if top_holdout:
        top_cap = max(float(row.get("cap_hit_mean", 0.0)) for row in top_holdout)
        top_bad = max(float(row.get("malformed_mean", 0.0)) for row in top_holdout)
        checks.extend(
            [
                bool_check(
                    "summary_top_holdout_cap_hit_below_threshold",
                    top_cap <= max_selected_cap_hit,
                    {"max": top_cap, "threshold": max_selected_cap_hit},
                ),
                bool_check(
                    "summary_top_holdout_malformed_below_threshold",
                    top_bad <= max_selected_malformed,
                    {"max": top_bad, "threshold": max_selected_malformed},
                ),
            ]
        )
    return checks


def run_validity_audit(
    run_dir: Path,
    *,
    max_selected_cap_hit: float = 0.10,
    max_selected_malformed: float = 0.10,
    max_rescore_mismatches: int = 0,
) -> dict:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(summary_path)
    summary = read_json(summary_path)
    per_prompt = read_jsonl(run_dir / "per_prompt.jsonl")
    holdout = read_jsonl(run_dir / "holdout_per_prompt.jsonl")
    ensemble_per_prompt = read_jsonl(run_dir / "ensemble_per_prompt.jsonl")
    checks: list[ValidityCheck] = []
    checks.extend(summary_prompt_checks(summary))
    checks.extend(base_prompt_checks(per_prompt, holdout))
    checks.extend(protocol_checks(summary, holdout, ensemble_per_prompt))
    mismatches = strict_rescore_mismatches(per_prompt + holdout)
    checks.append(
        bool_check(
            "stored_rows_match_current_strict_parser",
            len(mismatches) <= max_rescore_mismatches,
            {"mismatches": len(mismatches), "sample": mismatches[:16]},
        )
    )
    checks.extend(
        selected_candidate_checks(
            summary,
            holdout,
            max_selected_cap_hit=max_selected_cap_hit,
            max_selected_malformed=max_selected_malformed,
        )
    )
    rows = [asdict(check) for check in checks]
    failed = [row["check"] for row in rows if not row["passed"]]
    return {
        "kind": "result_validity_audit",
        "run_dir": str(run_dir),
        "pass": not failed,
        "failed": failed,
        "checks": rows,
        "thresholds": {
            "max_selected_cap_hit": max_selected_cap_hit,
            "max_selected_malformed": max_selected_malformed,
            "max_rescore_mismatches": max_rescore_mismatches,
        },
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Result Validity Audit",
        "",
        f"Run: `{summary['run_dir']}`",
        f"Pass: `{str(summary['pass']).lower()}`",
        "",
        "| check | pass | detail |",
        "| --- | ---: | --- |",
    ]
    for row in summary["checks"]:
        lines.append(f"| {row['check']} | {str(row['passed']).lower()} | `{json.dumps(row['detail'], sort_keys=True)}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit a search run before using it for quality claims.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--max-selected-cap-hit", type=float, default=0.10)
    parser.add_argument("--max-selected-malformed", type=float, default=0.10)
    parser.add_argument("--max-rescore-mismatches", type=int, default=0)
    args = parser.parse_args(argv)

    summary = run_validity_audit(
        args.run,
        max_selected_cap_hit=args.max_selected_cap_hit,
        max_selected_malformed=args.max_selected_malformed,
        max_rescore_mismatches=args.max_rescore_mismatches,
    )
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if summary["pass"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
