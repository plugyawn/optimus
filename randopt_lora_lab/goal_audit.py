from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class GoalCheck:
    requirement: str
    passed: bool
    evidence: str
    detail: Any


def read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text())


def check_present_pass(requirement: str, path: Path | None, payload: dict | None, *, evidence_name: str) -> GoalCheck:
    if payload is None:
        return GoalCheck(requirement, False, str(path) if path else "missing", "missing evidence")
    return GoalCheck(requirement, bool(payload.get("pass")), evidence_name, payload.get("failed", payload.get("gates", payload)))


def check_parity_report(path: Path | None, payload: dict | None) -> list[GoalCheck]:
    if payload is None:
        return [
            GoalCheck("quality parity", False, str(path) if path else "missing", "missing parity report"),
            GoalCheck("stability parity", False, str(path) if path else "missing", "missing parity report"),
            GoalCheck("speed parity", False, str(path) if path else "missing", "missing parity report"),
        ]
    gates = payload.get("gates", {})
    return [
        GoalCheck(
            "quality parity",
            bool(gates.get("ensemble_quality")) and bool(payload.get("pass")),
            str(path),
            {
                "overall_pass": payload.get("pass"),
                "ensemble_quality": gates.get("ensemble_quality"),
                "ensemble_delta": payload.get("ensemble_holdout_delta_lora_minus_dense"),
            },
        ),
        GoalCheck(
            "stability parity",
            bool(gates.get("spearman")) and bool(gates.get("topk_overlap")) and bool(gates.get("selected_regret")),
            str(path),
            {
                "spearman": payload.get("spearman"),
                "topk_overlap": payload.get("topk_overlap"),
                "selected_regret": payload.get("selected_regret"),
                "gates": {
                    "spearman": gates.get("spearman"),
                    "topk_overlap": gates.get("topk_overlap"),
                    "selected_regret": gates.get("selected_regret"),
                },
            },
        ),
        GoalCheck(
            "speed parity",
            bool(gates.get("speed")),
            str(path),
            {
                "speed_ratio_lora_over_dense": payload.get("speed_ratio_lora_over_dense"),
                "gate": gates.get("speed"),
            },
        ),
    ]


def check_prompt_robustness(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("prompt robustness", False, str(path) if path else "missing", "missing prompt robustness report")
    gate = payload.get("gate", payload)
    return GoalCheck(
        "prompt robustness",
        bool(gate.get("pass")),
        str(path),
        {
            "pass": gate.get("pass"),
            "valid_prompt_variants": gate.get("valid_prompt_variants"),
            "passing_prompt_variants": gate.get("passing_prompt_variants"),
            "min_valid_prompts": gate.get("min_valid_prompts"),
        },
    )


def check_drift(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("drift parity", False, str(path) if path else "missing", "missing drift evidence")
    passed = bool(payload.get("pass") or payload.get("drift_pass"))
    return GoalCheck("drift parity", passed, str(path), payload.get("gates", payload))


def check_adapter_convenience(path: Path | None) -> GoalCheck:
    if path is None:
        return GoalCheck("adapter convenience", False, "missing", "missing adapter run directory")
    adapters = path / "adapters.jsonl"
    summary = path / "summary.json"
    kept = False
    if summary.exists():
        kept = bool(json.loads(summary.read_text()).get("adapters_kept", False))
    passed = adapters.exists() or kept
    return GoalCheck(
        "adapter convenience",
        passed,
        str(path),
        {
            "adapters_jsonl_exists": adapters.exists(),
            "summary_adapters_kept": kept,
        },
    )


def run_goal_audit(args) -> dict:
    checks: list[GoalCheck] = []
    reproduction = read_json(args.reproduction_audit)
    parity = read_json(args.parity_report)
    backend = read_json(args.backend_gate)
    prompt = read_json(args.prompt_robustness)
    drift = read_json(args.drift_report)
    checks.append(
        check_present_pass(
            "official full-Gaussian baseline validity",
            args.reproduction_audit,
            reproduction,
            evidence_name=str(args.reproduction_audit),
        )
    )
    checks.extend(check_parity_report(args.parity_report, parity))
    checks.append(
        check_present_pass(
            "trusted accelerated backend selector",
            args.backend_gate,
            backend,
            evidence_name=str(args.backend_gate),
        )
    )
    checks.append(check_prompt_robustness(args.prompt_robustness, prompt))
    checks.append(check_drift(args.drift_report, drift))
    checks.append(check_adapter_convenience(args.adapter_run))
    rows = [asdict(check) for check in checks]
    return {
        "pass": all(row["passed"] for row in rows),
        "failed": [row["requirement"] for row in rows if not row["passed"]],
        "checks": rows,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# RandOpt LoRA Goal Audit",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
        "",
        "| requirement | pass | evidence | detail |",
        "| --- | ---: | --- | --- |",
    ]
    for row in summary["checks"]:
        lines.append(
            f"| {row['requirement']} | {str(row['passed']).lower()} | "
            f"`{row['evidence']}` | `{json.dumps(row['detail'], sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit whether the end-to-end LoRA perturbation-search goal is met.")
    parser.add_argument("--reproduction-audit", type=Path)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--backend-gate", type=Path)
    parser.add_argument("--prompt-robustness", type=Path)
    parser.add_argument("--drift-report", type=Path)
    parser.add_argument("--adapter-run", type=Path)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    summary = run_goal_audit(args)
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
