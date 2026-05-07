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
            lifted["protocol_valid"] = protocol_valid
            out.append(lifted)
        target = next((row for row in out if row["prompt_variant"] == variant and row["max_new_tokens"] == cap and row["candidate_kind"] == target_kind), None)
        if target:
            prompt_caps[(variant, cap)] = target
    return out, prompt_caps


def gate_prompt_robustness(prompt_caps: dict, *, min_valid_prompts: int, min_lift: float) -> dict:
    valid = [row for row in prompt_caps.values() if row["protocol_valid"]]
    invalid = [row for row in prompt_caps.values() if not row["protocol_valid"]]
    lifts = [float(row["lift_vs_base"]) for row in valid]
    return {
        "valid_prompt_conditions": len(valid),
        "invalid_prompt_conditions": len(invalid),
        "min_valid_prompts": min_valid_prompts,
        "min_lift": min_lift,
        "min_lift_observed": min(lifts) if lifts else None,
        "mean_lift_observed": sum(lifts) / len(lifts) if lifts else None,
        "pass": len(valid) >= min_valid_prompts and bool(lifts) and min(lifts) >= min_lift,
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
    for key in ["valid_prompt_conditions", "invalid_prompt_conditions", "min_valid_prompts", "min_lift", "min_lift_observed", "mean_lift_observed"]:
        lines.append(f"| {key} | {gate[key]} |")
    lines.extend(["", f"Pass: `{str(gate['pass']).lower()}`", ""])
    lines.extend([
        "## Candidate Prompt Rows",
        "",
        "| prompt | cap | kind | exact | base exact | lift | malformed | cap-hit | protocol valid |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ])
    for row in summary["rows"]:
        lines.append(
            "| {prompt_variant} | {max_new_tokens} | {candidate_kind} | {exact_mean:.6f} | {base_exact_mean:.6f} | {lift_vs_base:.6f} | {malformed_mean:.6f} | {cap_hit_mean:.6f} | {valid} |".format(
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
        "gate": gate_prompt_robustness(prompt_caps, min_valid_prompts=args.min_valid_prompts, min_lift=args.min_lift),
        "rows": rows,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
