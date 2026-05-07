from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_summary(path: Path) -> dict:
    return json.loads(path.read_text())


def summarize_rows(
    rows: list[dict],
    *,
    split: str,
    max_base_malformed: float,
    max_base_cap_hit: float,
    target_kind: str,
) -> tuple[list[dict], dict]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        if row.get("split") != split:
            continue
        grouped.setdefault((str(row["prompt_variant"]), int(row["max_new_tokens"])), []).append(row)

    out = []
    prompt_caps = {}
    for (variant, cap), group in sorted(grouped.items()):
        base = next((row for row in group if row.get("candidate_kind") == "base"), None)
        if not base:
            continue
        protocol_valid = (
            float(base.get("malformed_mean", 1.0)) <= max_base_malformed
            and float(base.get("cap_hit_mean", 1.0)) <= max_base_cap_hit
        )
        for row in group:
            if row.get("candidate_kind") == "base":
                continue
            lifted = dict(row)
            lifted["base_exact_mean"] = float(base["exact_mean"])
            lifted["base_malformed_mean"] = float(base["malformed_mean"])
            lifted["base_cap_hit_mean"] = float(base["cap_hit_mean"])
            lifted["lift_vs_base"] = float(row["exact_mean"]) - float(base["exact_mean"])
            lifted["malformed_regression_vs_base"] = float(row["malformed_mean"]) - float(base["malformed_mean"])
            lifted["cap_hit_regression_vs_base"] = float(row["cap_hit_mean"]) - float(base["cap_hit_mean"])
            lifted["protocol_valid"] = protocol_valid
            out.append(lifted)
        target = next((row for row in out if row["prompt_variant"] == variant and row["max_new_tokens"] == cap and row["candidate_kind"] == target_kind), None)
        if target:
            prompt_caps[(variant, cap)] = target
    return out, prompt_caps


def candidate_quality_valid(
    row: dict,
    *,
    max_candidate_malformed: float,
    max_candidate_cap_hit: float,
    max_malformed_regression: float,
    max_cap_hit_regression: float,
) -> bool:
    return (
        float(row["malformed_mean"]) <= max_candidate_malformed
        and float(row["cap_hit_mean"]) <= max_candidate_cap_hit
        and float(row["malformed_regression_vs_base"]) <= max_malformed_regression
        and float(row["cap_hit_regression_vs_base"]) <= max_cap_hit_regression
    )


def gate_prompt_robustness(
    prompt_caps: dict,
    *,
    min_valid_prompts: int,
    min_lift: float,
    max_candidate_malformed: float,
    max_candidate_cap_hit: float,
    max_malformed_regression: float,
    max_cap_hit_regression: float,
) -> dict:
    condition_rows = []
    for key, row in prompt_caps.items():
        lifted = dict(row)
        if "prompt_variant" not in lifted and isinstance(key, tuple) and key:
            lifted["prompt_variant"] = str(key[0])
        if "max_new_tokens" not in lifted and isinstance(key, tuple) and len(key) > 1:
            lifted["max_new_tokens"] = int(key[1])
        condition_rows.append(lifted)

    valid_conditions = [row for row in condition_rows if row["protocol_valid"]]
    invalid_conditions = [row for row in condition_rows if not row["protocol_valid"]]
    all_lifts = [float(row["lift_vs_base"]) for row in valid_conditions]

    by_prompt: dict[str, list[dict]] = {}
    for row in valid_conditions:
        by_prompt.setdefault(str(row["prompt_variant"]), []).append(row)

    prompt_rows = []
    for variant, rows in sorted(by_prompt.items()):
        lifts = [float(row["lift_vs_base"]) for row in rows]
        quality_flags = [
            candidate_quality_valid(
                row,
                max_candidate_malformed=max_candidate_malformed,
                max_candidate_cap_hit=max_candidate_cap_hit,
                max_malformed_regression=max_malformed_regression,
                max_cap_hit_regression=max_cap_hit_regression,
            )
            for row in rows
        ]
        prompt_rows.append(
            {
                "prompt_variant": variant,
                "valid_conditions": len(rows),
                "min_lift_observed": min(lifts),
                "mean_lift_observed": sum(lifts) / len(lifts),
                "quality_valid_conditions": sum(1 for flag in quality_flags if flag),
                "pass": min(lifts) >= min_lift and all(quality_flags),
            }
        )

    passing_prompt_variants = [row for row in prompt_rows if row["pass"]]
    return {
        "valid_prompt_conditions": len(valid_conditions),
        "invalid_prompt_conditions": len(invalid_conditions),
        "valid_prompt_variants": len(by_prompt),
        "passing_prompt_variants": len(passing_prompt_variants),
        "min_valid_prompts": min_valid_prompts,
        "min_lift": min_lift,
        "max_candidate_malformed": max_candidate_malformed,
        "max_candidate_cap_hit": max_candidate_cap_hit,
        "max_malformed_regression": max_malformed_regression,
        "max_cap_hit_regression": max_cap_hit_regression,
        "min_lift_observed": min(all_lifts) if all_lifts else None,
        "mean_lift_observed": sum(all_lifts) / len(all_lifts) if all_lifts else None,
        "prompt_rows": prompt_rows,
        "pass": len(passing_prompt_variants) >= min_valid_prompts,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Prompt Robustness Report",
        "",
        f"Split: `{summary['split']}`",
        "",
        "## Gate",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    gate = summary["gate"]
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
    lines.extend([
        "## Prompt Variant Gate",
        "",
        "| prompt | valid conditions | min lift | mean lift | quality-valid conditions | pass |",
        "| --- | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in gate["prompt_rows"]:
        lines.append(
            "| {prompt_variant} | {valid_conditions} | {min_lift_observed:.6f} | {mean_lift_observed:.6f} | {quality_valid_conditions} | {pass_value} |".format(
                pass_value=str(row["pass"]).lower(),
                **row,
            )
        )
    lines.append("")
    lines.extend([
        "## Candidate Prompt Rows",
        "",
        "| prompt | cap | kind | exact | base exact | lift | malformed | cap-hit | malformed delta | cap-hit delta | protocol valid |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in summary["rows"]:
        lines.append(
            "| {prompt_variant} | {max_new_tokens} | {candidate_kind} | {exact_mean:.6f} | {base_exact_mean:.6f} | {lift_vs_base:.6f} | {malformed_mean:.6f} | {cap_hit_mean:.6f} | {malformed_regression_vs_base:.6f} | {cap_hit_regression_vs_base:.6f} | {valid} |".format(
                valid=str(row["protocol_valid"]).lower(),
                **row,
            )
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Summarize prompt-relative robustness from a cap_stability run.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--split", default="holdout")
    parser.add_argument("--target-kind", default="aggregate")
    parser.add_argument("--max-base-malformed", type=float, default=0.05)
    parser.add_argument("--max-base-cap-hit", type=float, default=0.05)
    parser.add_argument("--min-valid-prompts", type=int, default=2)
    parser.add_argument("--min-lift", type=float, default=0.0)
    parser.add_argument("--max-candidate-malformed", type=float, default=0.05)
    parser.add_argument("--max-candidate-cap-hit", type=float, default=0.05)
    parser.add_argument("--max-malformed-regression", type=float, default=0.05)
    parser.add_argument("--max-cap-hit-regression", type=float, default=0.05)
    args = parser.parse_args(argv)

    source = load_summary(args.summary)
    rows, prompt_caps = summarize_rows(
        source["rows"],
        split=args.split,
        max_base_malformed=args.max_base_malformed,
        max_base_cap_hit=args.max_base_cap_hit,
        target_kind=args.target_kind,
    )
    summary = {
        "kind": "prompt_robustness_report",
        "source": str(args.summary),
        "split": args.split,
        "target_kind": args.target_kind,
        "gate": gate_prompt_robustness(
            prompt_caps,
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
