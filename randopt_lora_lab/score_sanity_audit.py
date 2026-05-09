from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any


@dataclass
class Check:
    check: str
    passed: bool
    detail: Any


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text().splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def fnum(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def bool_check(check: str, passed: bool, detail: Any) -> Check:
    return Check(check, bool(passed), detail)


def score_key(row: dict[str, Any]) -> tuple[float, float]:
    selection = fnum(row.get("selection_score"))
    exact = fnum(row.get("exact_mean")) or 0.0
    return (exact if selection is None else selection, exact)


def top_rows(rows: list[dict[str, Any]], k: int) -> list[dict[str, Any]]:
    return sorted(rows, key=score_key, reverse=True)[:k]


def max_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [fnum(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def mean_field(rows: list[dict[str, Any]], field: str) -> float | None:
    values = [fnum(row.get(field)) for row in rows]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def best_exact(rows: list[dict[str, Any]]) -> float | None:
    return max_field(rows, "exact_mean")


def prompt_health(summary: dict[str, Any], key: str) -> dict[str, Any] | None:
    prompt_rows = summary.get(key)
    if not isinstance(prompt_rows, dict):
        return None
    variants = []
    for variant, row in sorted(prompt_rows.items()):
        if not isinstance(row, dict):
            continue
        variants.append(
            {
                "prompt_variant": variant,
                "exact_mean": fnum(row.get("exact_mean")),
                "cap_hit_mean": fnum(row.get("cap_hit_mean")),
                "malformed_mean": fnum(row.get("malformed_mean")),
                "answer_closed_mean": fnum(row.get("answer_closed_mean")),
            }
        )
    if not variants:
        return None
    return {
        "variants": variants,
        "max_cap_hit": max((row["cap_hit_mean"] or 0.0) for row in variants),
        "max_malformed": max((row["malformed_mean"] or 0.0) for row in variants),
        "min_answer_closed": min((row["answer_closed_mean"] if row["answer_closed_mean"] is not None else 0.0) for row in variants),
    }


def discover_runs(root: Path) -> list[Path]:
    if (root / "summary.json").exists():
        return [root]
    preferred = [root / name for name in ["dense", "vllm", "confirmed", "lora", "peft"]]
    runs = [path for path in preferred if (path / "summary.json").exists()]
    if runs:
        return runs
    return sorted(path for path in root.iterdir() if path.is_dir() and (path / "summary.json").exists()) if root.exists() else []


def analyze_run(
    run: Path,
    *,
    top_k: int,
    min_screen_prompts: int,
    max_top_cap_hit: float,
    max_top_malformed: float,
    min_top_answer_closed: float,
    min_best_delta_vs_base: float,
    max_base_cap_hit: float,
    max_base_malformed: float,
) -> dict[str, Any]:
    summary = read_json(run / "summary.json")
    rows = read_jsonl(run / "candidate_summary.jsonl")
    checks: list[Check] = []
    checks.append(bool_check("summary_present", summary is not None, {"path": str(run / "summary.json")}))
    checks.append(bool_check("candidate_rows_present", bool(rows), {"rows": len(rows), "path": str(run / "candidate_summary.jsonl")}))

    if summary is None:
        return {
            "run": str(run),
            "name": run.name,
            "pass": False,
            "failed": [check.check for check in checks if not check.passed],
            "checks": [asdict(check) for check in checks],
            "metrics": {},
        }

    screen_prompts = fnum(summary.get("screen_prompts"))
    base_screen = fnum(summary.get("base_screen_exact"))
    base_holdout = fnum(summary.get("base_holdout_exact"))
    selected = top_rows(rows, top_k)
    selected_best = best_exact(selected)
    all_best = best_exact(rows)
    selected_max_cap = max_field(selected, "cap_hit_mean")
    selected_max_malformed = max_field(selected, "malformed_mean")
    selected_min_answer_closed = min((fnum(row.get("answer_closed_mean")) or 0.0) for row in selected) if selected else None
    screen_base_health = prompt_health(summary, "base_screen_by_prompt")
    holdout_base_health = prompt_health(summary, "base_holdout_by_prompt")

    checks.append(
        bool_check(
            "screen_prompt_count_sufficient",
            screen_prompts is not None and screen_prompts >= min_screen_prompts,
            {"screen_prompts": screen_prompts, "threshold": min_screen_prompts},
        )
    )
    checks.append(bool_check("base_screen_exact_present", base_screen is not None, {"base_screen_exact": base_screen}))
    checks.append(
        bool_check(
            "topk_best_exact_clears_base",
            selected_best is not None and base_screen is not None and selected_best - base_screen >= min_best_delta_vs_base,
            {
                "topk_best_exact": selected_best,
                "base_screen_exact": base_screen,
                "min_delta": min_best_delta_vs_base,
                "delta": None if selected_best is None or base_screen is None else selected_best - base_screen,
            },
        )
    )
    checks.append(
        bool_check(
            "topk_cap_hit_below_threshold",
            selected_max_cap is not None and selected_max_cap <= max_top_cap_hit,
            {"topk_max_cap_hit": selected_max_cap, "threshold": max_top_cap_hit},
        )
    )
    checks.append(
        bool_check(
            "topk_malformed_below_threshold",
            selected_max_malformed is not None and selected_max_malformed <= max_top_malformed,
            {"topk_max_malformed": selected_max_malformed, "threshold": max_top_malformed},
        )
    )
    checks.append(
        bool_check(
            "topk_answer_closed_above_threshold",
            selected_min_answer_closed is not None and selected_min_answer_closed >= min_top_answer_closed,
            {"topk_min_answer_closed": selected_min_answer_closed, "threshold": min_top_answer_closed},
        )
    )
    if screen_base_health is not None:
        checks.append(
            bool_check(
                "base_screen_prompt_health",
                screen_base_health["max_cap_hit"] <= max_base_cap_hit and screen_base_health["max_malformed"] <= max_base_malformed,
                {
                    "max_cap_hit": screen_base_health["max_cap_hit"],
                    "max_malformed": screen_base_health["max_malformed"],
                    "max_base_cap_hit": max_base_cap_hit,
                    "max_base_malformed": max_base_malformed,
                },
            )
        )
    if holdout_base_health is not None:
        checks.append(
            bool_check(
                "base_holdout_prompt_health",
                holdout_base_health["max_cap_hit"] <= max_base_cap_hit and holdout_base_health["max_malformed"] <= max_base_malformed,
                {
                    "max_cap_hit": holdout_base_health["max_cap_hit"],
                    "max_malformed": holdout_base_health["max_malformed"],
                    "max_base_cap_hit": max_base_cap_hit,
                    "max_base_malformed": max_base_malformed,
                },
            )
        )

    failed = [check.check for check in checks if not check.passed]
    metrics = {
        "family": summary.get("family"),
        "targets": summary.get("targets"),
        "population": summary.get("population"),
        "screen_prompts": screen_prompts,
        "holdout_prompts": fnum(summary.get("holdout_prompts")),
        "base_screen_exact": base_screen,
        "base_holdout_exact": base_holdout,
        "candidate_rows": len(rows),
        "top_k": top_k,
        "topk_best_exact": selected_best,
        "all_best_exact": all_best,
        "topk_delta_vs_base_screen": None if selected_best is None or base_screen is None else selected_best - base_screen,
        "topk_max_cap_hit": selected_max_cap,
        "topk_max_malformed": selected_max_malformed,
        "topk_min_answer_closed": selected_min_answer_closed,
        "all_max_cap_hit": max_field(rows, "cap_hit_mean"),
        "all_max_malformed": max_field(rows, "malformed_mean"),
        "all_mean_cap_hit": mean_field(rows, "cap_hit_mean"),
        "all_mean_malformed": mean_field(rows, "malformed_mean"),
        "base_screen_prompt_health": screen_base_health,
        "base_holdout_prompt_health": holdout_base_health,
        "top_candidates": [
            {
                "candidate": row.get("candidate"),
                "exact_mean": fnum(row.get("exact_mean")),
                "selection_score": fnum(row.get("selection_score")),
                "cap_hit_mean": fnum(row.get("cap_hit_mean")),
                "malformed_mean": fnum(row.get("malformed_mean")),
                "answer_closed_mean": fnum(row.get("answer_closed_mean")),
            }
            for row in selected
        ],
    }
    return {
        "run": str(run),
        "name": run.name,
        "pass": not failed,
        "failed": failed,
        "checks": [asdict(check) for check in checks],
        "metrics": metrics,
    }


def analyze(
    root: Path,
    *,
    top_k: int = 4,
    min_screen_prompts: int = 32,
    max_top_cap_hit: float = 0.1,
    max_top_malformed: float = 0.1,
    min_top_answer_closed: float = 0.95,
    min_best_delta_vs_base: float = 0.0,
    max_base_cap_hit: float = 0.05,
    max_base_malformed: float = 0.05,
) -> dict[str, Any]:
    runs = [
        analyze_run(
            run,
            top_k=top_k,
            min_screen_prompts=min_screen_prompts,
            max_top_cap_hit=max_top_cap_hit,
            max_top_malformed=max_top_malformed,
            min_top_answer_closed=min_top_answer_closed,
            min_best_delta_vs_base=min_best_delta_vs_base,
            max_base_cap_hit=max_base_cap_hit,
            max_base_malformed=max_base_malformed,
        )
        for run in discover_runs(root)
    ]
    failed = [f"{run['name']}:{check}" for run in runs for check in run["failed"]]
    return {
        "kind": "score_sanity_audit",
        "root": str(root),
        "pass": bool(runs) and not failed,
        "failed": failed if runs else ["no_runs_found"],
        "runs": runs,
        "thresholds": {
            "top_k": top_k,
            "min_screen_prompts": min_screen_prompts,
            "max_top_cap_hit": max_top_cap_hit,
            "max_top_malformed": max_top_malformed,
            "min_top_answer_closed": min_top_answer_closed,
            "min_best_delta_vs_base": min_best_delta_vs_base,
            "max_base_cap_hit": max_base_cap_hit,
            "max_base_malformed": max_base_malformed,
        },
    }


def fmt(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Score Sanity Audit",
        "",
        f"Gate: **{'PASS' if summary['pass'] else 'FAIL'}**",
        "",
        "| run | pass | base screen | top-k best | delta | top-k cap | top-k malformed | failed |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for run in summary["runs"]:
        m = run["metrics"]
        lines.append(
            f"| {run['name']} | {str(run['pass']).lower()} | {fmt(m.get('base_screen_exact'))} | "
            f"{fmt(m.get('topk_best_exact'))} | {fmt(m.get('topk_delta_vs_base_screen'))} | "
            f"{fmt(m.get('topk_max_cap_hit'))} | {fmt(m.get('topk_max_malformed'))} | "
            f"`{json.dumps(run['failed'])}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit aggregate score sanity for Countdown perturbation-search runs.")
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--min-screen-prompts", type=int, default=32)
    parser.add_argument("--max-top-cap-hit", type=float, default=0.1)
    parser.add_argument("--max-top-malformed", type=float, default=0.1)
    parser.add_argument("--min-top-answer-closed", type=float, default=0.95)
    parser.add_argument("--min-best-delta-vs-base", type=float, default=0.0)
    parser.add_argument("--max-base-cap-hit", type=float, default=0.05)
    parser.add_argument("--max-base-malformed", type=float, default=0.05)
    args = parser.parse_args(argv)

    summary = analyze(
        args.root,
        top_k=args.top_k,
        min_screen_prompts=args.min_screen_prompts,
        max_top_cap_hit=args.max_top_cap_hit,
        max_top_malformed=args.max_top_malformed,
        min_top_answer_closed=args.min_top_answer_closed,
        min_best_delta_vs_base=args.min_best_delta_vs_base,
        max_base_cap_hit=args.max_base_cap_hit,
        max_base_malformed=args.max_base_malformed,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_report(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
