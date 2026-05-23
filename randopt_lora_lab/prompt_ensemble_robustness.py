from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

from optimus.tasks.countdown import CountdownExample, extract_numeric_vote, score_completion, voted_answer_exact


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def elite_sort_key(kind: str) -> tuple[int, str]:
    if kind.startswith("elite_"):
        try:
            return int(kind.removeprefix("elite_")), kind
        except ValueError:
            pass
    return 10**9, kind


def condition_key(row: dict) -> tuple[str, int, str]:
    return str(row["prompt_variant"]), int(row["max_new_tokens"]), str(row["split"])


def group_summary_rows(rows: list[dict], split: str) -> dict[tuple[str, int], dict[str, dict]]:
    grouped: dict[tuple[str, int], dict[str, dict]] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        key = (str(row["prompt_variant"]), int(row["max_new_tokens"]))
        grouped.setdefault(key, {})[str(row["candidate_kind"])] = row
    return grouped


def group_per_prompt_rows(rows: list[dict], split: str) -> dict[tuple[str, int, str, int], dict]:
    grouped = {}
    for row in rows:
        if row.get("split") != split:
            continue
        key = (
            str(row["prompt_variant"]),
            int(row["max_new_tokens"]),
            str(row["candidate_kind"]),
            int(row["example_id"]),
        )
        grouped[key] = row
    return grouped


def example_from_row(row: dict) -> CountdownExample:
    return CountdownExample(int(row["example_id"]), tuple(int(x) for x in row["numbers"]), int(row["target"]))


def strict_vote(row: dict, example: CountdownExample, *, strict_rows: bool) -> tuple[str, str]:
    if strict_rows and score_completion(str(row.get("text", "")), example, strict=True)["malformed"]:
        return "", "strict_malformed"
    vote = extract_numeric_vote(str(row.get("text", "")), example)
    if not vote["valid_vote"]:
        return "", str(vote["vote_reject"])
    return str(vote["vote"]), ""


def evaluate_ensemble_condition(
    per_prompt_rows: dict[tuple[str, int, str, int], dict],
    *,
    prompt_variant: str,
    max_new_tokens: int,
    candidate_kinds: list[str],
    strict_rows: bool,
) -> dict:
    example_ids = sorted(
        {
            int(example_id)
            for variant, cap, kind, example_id in per_prompt_rows
            if variant == prompt_variant and cap == max_new_tokens and kind in candidate_kinds
        }
    )
    exact_values = []
    coverage_values = []
    vote_count_values = []
    reject_counter = Counter()
    for example_id in example_ids:
        votes = []
        example = None
        for kind in candidate_kinds:
            row = per_prompt_rows.get((prompt_variant, max_new_tokens, kind, example_id))
            if row is None:
                reject_counter["missing_row"] += 1
                continue
            example = example or example_from_row(row)
            vote, reject = strict_vote(row, example, strict_rows=strict_rows)
            if vote:
                votes.append(vote)
            else:
                reject_counter[reject] += 1
        counter = Counter(votes)
        final_vote = counter.most_common(1)[0][0] if counter else ""
        exact_values.append(voted_answer_exact(final_vote, example) if example is not None else 0.0)
        coverage_values.append(float(bool(counter)))
        vote_count_values.append(len(votes))
    return {
        "exact_mean": float(np.mean(exact_values)) if exact_values else 0.0,
        "coverage_mean": float(np.mean(coverage_values)) if coverage_values else 0.0,
        "valid_votes_per_prompt": float(np.mean(vote_count_values)) if vote_count_values else 0.0,
        "correct": int(sum(exact_values)),
        "total": len(exact_values),
        "reject_counts": dict(reject_counter),
        "strict_rows": bool(strict_rows),
    }


def candidate_quality_valid(
    row: dict,
    *,
    max_candidate_malformed: float,
    max_candidate_cap_hit: float,
    max_malformed_regression: float,
    max_cap_hit_regression: float,
) -> bool:
    return (
        float(row["max_candidate_malformed"]) <= max_candidate_malformed
        and float(row["max_candidate_cap_hit"]) <= max_candidate_cap_hit
        and float(row["max_malformed_regression_vs_base"]) <= max_malformed_regression
        and float(row["max_cap_hit_regression_vs_base"]) <= max_cap_hit_regression
    )


def summarize_ensemble_robustness(
    summary_rows: list[dict],
    per_prompt_rows: list[dict],
    *,
    split: str,
    k: int,
    strict_rows: bool,
    max_base_malformed: float,
    max_base_cap_hit: float,
) -> list[dict]:
    by_condition = group_summary_rows(summary_rows, split)
    per_prompt = group_per_prompt_rows(per_prompt_rows, split)
    rows = []
    for (variant, cap), by_kind in sorted(by_condition.items()):
        base = by_kind.get("base")
        if not base:
            continue
        elites = sorted((kind for kind in by_kind if kind.startswith("elite_")), key=elite_sort_key)[:k]
        if not elites:
            continue
        ev = evaluate_ensemble_condition(
            per_prompt,
            prompt_variant=variant,
            max_new_tokens=cap,
            candidate_kinds=elites,
            strict_rows=strict_rows,
        )
        candidate_rows = [by_kind[kind] for kind in elites]
        max_malformed = max(float(row.get("malformed_mean", 1.0)) for row in candidate_rows)
        max_cap_hit = max(float(row.get("cap_hit_mean", 1.0)) for row in candidate_rows)
        base_malformed = float(base.get("malformed_mean", 1.0))
        base_cap_hit = float(base.get("cap_hit_mean", 1.0))
        row = {
            "split": split,
            "prompt_variant": variant,
            "max_new_tokens": cap,
            "k": len(elites),
            "candidate_kinds": elites,
            "base_exact_mean": float(base.get("exact_mean", 0.0)),
            "base_malformed_mean": base_malformed,
            "base_cap_hit_mean": base_cap_hit,
            "lift_vs_base": ev["exact_mean"] - float(base.get("exact_mean", 0.0)),
            "max_candidate_malformed": max_malformed,
            "max_candidate_cap_hit": max_cap_hit,
            "max_malformed_regression_vs_base": max_malformed - base_malformed,
            "max_cap_hit_regression_vs_base": max_cap_hit - base_cap_hit,
            "protocol_valid": base_malformed <= max_base_malformed and base_cap_hit <= max_base_cap_hit,
            **ev,
        }
        rows.append(row)
    return rows


def gate_rows(
    rows: list[dict],
    *,
    min_valid_prompts: int,
    min_lift: float,
    max_candidate_malformed: float,
    max_candidate_cap_hit: float,
    max_malformed_regression: float,
    max_cap_hit_regression: float,
) -> dict:
    valid_rows = [row for row in rows if row["protocol_valid"]]
    invalid_rows = [row for row in rows if not row["protocol_valid"]]
    by_prompt: dict[str, list[dict]] = {}
    for row in valid_rows:
        by_prompt.setdefault(str(row["prompt_variant"]), []).append(row)
    prompt_rows = []
    for prompt_variant, prompt_group in sorted(by_prompt.items()):
        lifts = [float(row["lift_vs_base"]) for row in prompt_group]
        quality_flags = [
            candidate_quality_valid(
                row,
                max_candidate_malformed=max_candidate_malformed,
                max_candidate_cap_hit=max_candidate_cap_hit,
                max_malformed_regression=max_malformed_regression,
                max_cap_hit_regression=max_cap_hit_regression,
            )
            for row in prompt_group
        ]
        prompt_rows.append(
            {
                "prompt_variant": prompt_variant,
                "valid_conditions": len(prompt_group),
                "min_lift_observed": min(lifts),
                "mean_lift_observed": sum(lifts) / len(lifts),
                "quality_valid_conditions": sum(1 for flag in quality_flags if flag),
                "pass": min(lifts) >= min_lift and all(quality_flags),
            }
        )
    passing_prompts = [row for row in prompt_rows if row["pass"]]
    all_lifts = [float(row["lift_vs_base"]) for row in valid_rows]
    return {
        "valid_prompt_conditions": len(valid_rows),
        "invalid_prompt_conditions": len(invalid_rows),
        "valid_prompt_variants": len(by_prompt),
        "passing_prompt_variants": len(passing_prompts),
        "min_valid_prompts": min_valid_prompts,
        "min_lift": min_lift,
        "max_candidate_malformed": max_candidate_malformed,
        "max_candidate_cap_hit": max_candidate_cap_hit,
        "max_malformed_regression": max_malformed_regression,
        "max_cap_hit_regression": max_cap_hit_regression,
        "min_lift_observed": min(all_lifts) if all_lifts else None,
        "mean_lift_observed": sum(all_lifts) / len(all_lifts) if all_lifts else None,
        "prompt_rows": prompt_rows,
        "pass": len(passing_prompts) >= min_valid_prompts,
    }


def render_markdown(summary: dict) -> str:
    gate = summary["gate"]
    lines = [
        "# Prompt Ensemble Robustness Report",
        "",
        f"Split: `{summary['split']}`",
        f"Ensemble k: `{summary['k']}`",
        f"Strict rows: `{str(summary['strict_rows']).lower()}`",
        "",
        "## Gate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key in [
        "valid_prompt_conditions",
        "invalid_prompt_conditions",
        "valid_prompt_variants",
        "passing_prompt_variants",
        "min_valid_prompts",
        "min_lift",
        "max_candidate_malformed",
        "max_candidate_cap_hit",
        "max_malformed_regression",
        "max_cap_hit_regression",
        "min_lift_observed",
        "mean_lift_observed",
    ]:
        lines.append(f"| {key} | {gate[key]} |")
    lines.extend(["", f"Pass: `{str(gate['pass']).lower()}`", ""])
    lines.extend(
        [
            "## Prompt Variant Gate",
            "",
            "| prompt | valid conditions | min lift | mean lift | quality-valid conditions | pass |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in gate["prompt_rows"]:
        lines.append(
            "| {prompt_variant} | {valid_conditions} | {min_lift_observed:.6f} | {mean_lift_observed:.6f} | {quality_valid_conditions} | {pass_value} |".format(
                pass_value=str(row["pass"]).lower(),
                **row,
            )
        )
    lines.extend(
        [
            "",
            "## Condition Rows",
            "",
            "| prompt | cap | exact | base exact | lift | max malformed | max cap-hit | malformed delta | cap-hit delta | protocol valid |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for row in summary["rows"]:
        lines.append(
            "| {prompt_variant} | {max_new_tokens} | {exact_mean:.6f} | {base_exact_mean:.6f} | {lift_vs_base:.6f} | {max_candidate_malformed:.6f} | {max_candidate_cap_hit:.6f} | {max_malformed_regression_vs_base:.6f} | {max_cap_hit_regression_vs_base:.6f} | {valid} |".format(
                valid=str(row["protocol_valid"]).lower(),
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Gate prompt/cap robustness for top-k elite voting from a cap_stability run.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--per-prompt", type=Path, default=None)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--k", type=int, default=4)
    parser.add_argument("--strict-rows", action="store_true")
    parser.add_argument("--max-base-malformed", type=float, default=0.05)
    parser.add_argument("--max-base-cap-hit", type=float, default=0.05)
    parser.add_argument("--min-valid-prompts", type=int, default=2)
    parser.add_argument("--min-lift", type=float, default=0.0)
    parser.add_argument("--max-candidate-malformed", type=float, default=0.05)
    parser.add_argument("--max-candidate-cap-hit", type=float, default=0.05)
    parser.add_argument("--max-malformed-regression", type=float, default=0.05)
    parser.add_argument("--max-cap-hit-regression", type=float, default=0.05)
    args = parser.parse_args(argv)

    source = json.loads(args.summary.read_text())
    per_prompt_path = args.per_prompt or args.summary.parent / "per_prompt.jsonl"
    rows = summarize_ensemble_robustness(
        source["rows"],
        read_jsonl(per_prompt_path),
        split=args.split,
        k=args.k,
        strict_rows=args.strict_rows,
        max_base_malformed=args.max_base_malformed,
        max_base_cap_hit=args.max_base_cap_hit,
    )
    summary = {
        "kind": "prompt_ensemble_robustness_report",
        "source": str(args.summary),
        "per_prompt": str(per_prompt_path),
        "split": args.split,
        "k": args.k,
        "strict_rows": bool(args.strict_rows),
        "gate": gate_rows(
            rows,
            min_valid_prompts=args.min_valid_prompts,
            min_lift=args.min_lift,
            max_candidate_malformed=args.max_candidate_malformed,
            max_candidate_cap_hit=args.max_candidate_cap_hit,
            max_malformed_regression=args.max_malformed_regression,
            max_cap_hit_regression=args.max_cap_hit_regression,
        ),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
