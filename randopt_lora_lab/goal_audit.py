from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


QPROJ_CORRECTED_CONFIRM_COMMAND = "MODE=confirm scripts/run_qproj_c2_corrected_confirmation.sh"


@dataclass
class GoalCheck:
    requirement: str
    passed: bool
    evidence: str
    detail: Any


NEXT_ACTIONS = {
    "upstream dense baseline validity": {
        "priority": 50,
        "action": "produce a full-scale upstream Countdown audit or a local dense Gaussian baseline audit before using it as the reference baseline",
        "command": "optimus upstream-baseline-audit --run RUN --out results/upstream_baseline_audit",
    },
    "quality parity": {
        "priority": 20,
        "action": "produce a current-valid LoRA-family parity report whose holdout quality matches or beats dense Gaussian",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
    "stability parity": {
        "priority": 21,
        "action": "rerun parity on a shared candidate panel until Spearman, top-k overlap, and selected-regret gates pass across seeds",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
    "speed parity": {
        "priority": 22,
        "action": "measure quality-coupled accelerated search speed against the dense full-screen reference, not same-family speed alone",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
    "accelerated evaluation route": {
        "priority": 10,
        "action": "run the dense-referenced shortlist confirmation path or fix direct accelerated-backend selector parity",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
    "adapter identity provenance": {
        "priority": 11,
        "action": "rerun activation-spectral PEFT confirmation with the saved vLLM family_state.pt copied and audited",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
    "multi-run prompt-robust confirmation": {
        "priority": 40,
        "action": "aggregate at least two prompt-valid runs after the dense-referenced confirmation path passes",
        "command": "optimus multirun-gate --run RUN1 --run RUN2 --parity-arm lora --out results/multirun_gate",
    },
    "prompt robustness": {
        "priority": 30,
        "action": "prove nonnegative lift and no cap/malformed regression on multiple base-valid prompt variants",
        "command": "optimus prompt-robustness --help",
    },
    "drift parity": {
        "priority": 60,
        "action": "run true nonnegative full-vocab next-token KL drift parity against dense Gaussian",
        "command": "optimus drift-parity --help",
    },
    "eval validity": {
        "priority": 12,
        "action": "run strict parser, semantic split, cap-hit, malformed, and ensemble-row validity on the claim artifact",
        "command": "optimus result-validity --run RUN --out RUN/validity",
    },
    "score sanity": {
        "priority": 12,
        "action": "audit top-candidate cap hits, malformed rates, answer closure, base prompt health, and base-score lift",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
    "adapter convenience": {
        "priority": 13,
        "action": "materialize and keep portable LoRA adapters plus replay metadata for the selected family",
        "command": QPROJ_CORRECTED_CONFIRM_COMMAND,
    },
}


def read_json(path: Path | None) -> dict | None:
    if path is None:
        return None
    if not path.exists():
        return None
    return json.loads(path.read_text())


def next_action(requirement: str) -> dict:
    payload = NEXT_ACTIONS.get(requirement, {"priority": 999, "action": "inspect failed gate detail", "command": ""})
    return {"requirement": requirement, **payload}


def check_present_pass(requirement: str, path: Path | None, payload: dict | None, *, evidence_name: str) -> GoalCheck:
    if payload is None:
        return GoalCheck(requirement, False, str(path) if path else "missing", "missing evidence")
    return GoalCheck(requirement, bool(payload.get("pass")), evidence_name, payload.get("failed", payload.get("gates", payload)))


def check_official_baseline(
    *,
    local_baseline_path: Path | None,
    local_baseline_payload: dict | None,
    upstream_path: Path | None,
    upstream_payload: dict | None,
) -> GoalCheck:
    if local_baseline_payload is None and upstream_payload is None:
        return GoalCheck(
            "upstream dense baseline validity",
            False,
            json.dumps(
                {
                    "local_baseline": str(local_baseline_path) if local_baseline_path else None,
                    "upstream_countdown": str(upstream_path) if upstream_path else None,
                },
                sort_keys=True,
            ),
            "missing baseline evidence",
        )
    local_pass = bool(local_baseline_payload and local_baseline_payload.get("pass"))
    upstream_scale_pass = bool(
        upstream_payload
        and upstream_payload.get("upstream_scale_pass", upstream_payload.get("paper_scale_pass"))
    )
    routes = []
    if local_pass:
        routes.append("local_baseline_audit")
    if upstream_scale_pass:
        routes.append("upstream_full_scale")
    passed = local_pass or upstream_scale_pass
    return GoalCheck(
        "upstream dense baseline validity",
        passed,
        json.dumps(
            {
                "local_baseline": str(local_baseline_path) if local_baseline_path else None,
                "upstream_countdown": str(upstream_path) if upstream_path else None,
            },
            sort_keys=True,
        ),
        {
            "pass": passed,
            "routes": routes,
            "local_baseline": None
            if local_baseline_payload is None
            else {
                "pass": local_baseline_payload.get("pass"),
                "failed": local_baseline_payload.get("failed", []),
            },
            "upstream_countdown": None
            if upstream_payload is None
            else {
                "pass": upstream_payload.get("pass"),
                "smoke_pass": upstream_payload.get("smoke_pass"),
                "upstream_scale_pass": upstream_payload.get(
                    "upstream_scale_pass",
                    upstream_payload.get("paper_scale_pass"),
                ),
                "failed": upstream_payload.get("failed", []),
                "summary": upstream_payload.get("summary", {}),
            },
        },
    )


def check_confirmation(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("two-stage accelerated confirmation", False, str(path) if path else "missing", "missing confirmation evidence")
    gate = payload.get("gate", payload)
    return GoalCheck(
        "two-stage accelerated confirmation",
        bool(gate.get("pass")),
        str(path),
        {
            "pass": gate.get("pass"),
            "failed": gate.get("failed", []),
            "best_recovered_k": payload.get("best_recovered_k"),
            "zero_regret_k": payload.get("zero_regret_k"),
            "thresholds": gate.get("thresholds", {}),
        },
    )


def gate_pass(payload: dict | None) -> bool:
    if payload is None:
        return False
    gate = payload.get("gate", payload)
    return bool(gate.get("pass"))


def gate_failed(payload: dict | None) -> Any:
    if payload is None:
        return "missing"
    gate = payload.get("gate", payload)
    return gate.get("failed", payload.get("failed", []))


def check_accelerated_route(
    *,
    backend_path: Path | None,
    backend_payload: dict | None,
    same_family_confirmation_path: Path | None,
    same_family_confirmation_payload: dict | None,
    dense_confirmation_path: Path | None,
    dense_confirmation_payload: dict | None,
    search_quality_path: Path | None,
    search_quality_payload: dict | None,
) -> GoalCheck:
    backend_selector_pass = bool(backend_payload and backend_payload.get("pass"))
    dense_confirmation_pass = gate_pass(dense_confirmation_payload)
    search_quality_pass = gate_pass(search_quality_payload)
    dense_referenced_two_stage_pass = dense_confirmation_pass and search_quality_pass
    same_family_confirmation_pass = gate_pass(same_family_confirmation_payload)
    passed = backend_selector_pass or dense_referenced_two_stage_pass
    routes = []
    if backend_selector_pass:
        routes.append("backend_selector")
    if dense_referenced_two_stage_pass:
        routes.append("dense_referenced_two_stage")
    evidence = {
        "backend_gate": None if backend_path is None else str(backend_path),
        "same_family_confirmation_gate": None
        if same_family_confirmation_path is None
        else str(same_family_confirmation_path),
        "dense_confirmation_gate": None if dense_confirmation_path is None else str(dense_confirmation_path),
        "search_quality_confirmation": None if search_quality_path is None else str(search_quality_path),
    }
    return GoalCheck(
        "accelerated evaluation route",
        passed,
        json.dumps(evidence, sort_keys=True),
        {
            "pass": passed,
            "routes": routes,
            "backend_selector_pass": backend_selector_pass,
            "backend_failed": gate_failed(backend_payload),
            "same_family_confirmation_pass": same_family_confirmation_pass,
            "same_family_confirmation_failed": gate_failed(same_family_confirmation_payload),
            "dense_referenced_two_stage_pass": dense_referenced_two_stage_pass,
            "dense_confirmation_pass": dense_confirmation_pass,
            "dense_confirmation_failed": gate_failed(dense_confirmation_payload),
            "search_quality_pass": search_quality_pass,
            "search_quality_failed": gate_failed(search_quality_payload),
            "dense_zero_regret_k": None if dense_confirmation_payload is None else dense_confirmation_payload.get("zero_dense_regret_k"),
            "dense_best_recovered_k": None
            if dense_confirmation_payload is None
            else dense_confirmation_payload.get("dense_best_recovered_k"),
        },
    )


def check_family_state_provenance(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("adapter identity provenance", False, str(path) if path else "missing", "missing family-state provenance audit")
    return GoalCheck(
        "adapter identity provenance",
        bool(payload.get("pass")),
        str(path),
        {
            "pass": payload.get("pass"),
            "failed": payload.get("failed", []),
            "runs": len(payload.get("runs", [])),
        },
    )


def dense_referenced_multirun_pass(payload: dict | None) -> bool:
    return bool(payload and payload.get("pass"))


def dense_referenced_multirun_detail(path: Path | None, payload: dict | None) -> dict:
    if payload is None:
        return {"pass": False, "evidence": str(path) if path else "missing", "failed": ["missing"]}
    return {
        "pass": payload.get("pass"),
        "evidence": str(path),
        "failed": payload.get("failed", []),
        "aggregate": payload.get("aggregate", {}),
        "thresholds": payload.get("thresholds", {}),
    }


def check_multirun_gate(
    path: Path | None,
    payload: dict | None,
    *,
    dense_referenced_path: Path | None = None,
    dense_referenced_payload: dict | None = None,
) -> GoalCheck:
    old_pass = bool(payload and payload.get("pass"))
    dense_pass = dense_referenced_multirun_pass(dense_referenced_payload)
    if payload is None and not dense_pass:
        return GoalCheck("multi-run prompt-robust confirmation", False, str(path) if path else "missing", "missing multi-run gate")
    routes = []
    if old_pass:
        routes.append("rank_parity_multirun")
    if dense_pass:
        routes.append("dense_referenced_multirun")
    return GoalCheck(
        "multi-run prompt-robust confirmation",
        old_pass or dense_pass,
        json.dumps(
            {
                "rank_parity_multirun": str(path) if path else None,
                "dense_referenced_multirun": str(dense_referenced_path) if dense_referenced_path else None,
            },
            sort_keys=True,
        ),
        {
            "pass": old_pass or dense_pass,
            "routes": routes,
            "rank_parity_multirun": None
            if payload is None
            else {
                "pass": payload.get("pass"),
                "failed": payload.get("failed", []),
                "aggregate": payload.get("aggregate", {}),
                "thresholds": payload.get("thresholds", {}),
            },
            "dense_referenced_multirun": dense_referenced_multirun_detail(dense_referenced_path, dense_referenced_payload),
        },
    )


def select_parity_payload(payload: dict, arm: str) -> dict:
    comparisons = payload.get("comparisons")
    if not comparisons:
        return payload
    if arm not in comparisons:
        return {"pass": False, "gates": {}, "missing_arm": arm, "available_arms": sorted(comparisons)}
    selected = dict(comparisons[arm])
    selected["selected_arm"] = arm
    return selected


def check_parity_report(
    path: Path | None,
    payload: dict | None,
    *,
    arm: str = "lora",
    dense_referenced_multirun_path: Path | None = None,
    dense_referenced_multirun_payload: dict | None = None,
) -> list[GoalCheck]:
    dense_multirun_pass = dense_referenced_multirun_pass(dense_referenced_multirun_payload)
    dense_multirun = dense_referenced_multirun_detail(
        dense_referenced_multirun_path,
        dense_referenced_multirun_payload,
    )
    if payload is None:
        if dense_multirun_pass:
            detail = {
                "routes": ["dense_referenced_multirun"],
                "parity_report": "missing",
                "dense_referenced_multirun": dense_multirun,
            }
            return [
                GoalCheck("quality parity", True, str(dense_referenced_multirun_path), detail),
                GoalCheck("stability parity", True, str(dense_referenced_multirun_path), detail),
                GoalCheck("speed parity", True, str(dense_referenced_multirun_path), detail),
            ]
        return [
            GoalCheck("quality parity", False, str(path) if path else "missing", "missing parity report"),
            GoalCheck("stability parity", False, str(path) if path else "missing", "missing parity report"),
            GoalCheck("speed parity", False, str(path) if path else "missing", "missing parity report"),
        ]
    payload = select_parity_payload(payload, arm)
    gates = payload.get("gates", {})
    missing_arm = payload.get("missing_arm")
    quality_parity_pass = bool(gates.get("ensemble_quality")) and bool(payload.get("pass"))
    stability_parity_pass = bool(gates.get("spearman")) and bool(gates.get("topk_overlap")) and bool(gates.get("selected_regret"))
    speed_parity_pass = bool(gates.get("speed"))
    return [
        GoalCheck(
            "quality parity",
            quality_parity_pass or dense_multirun_pass,
            json.dumps(
                {
                    "rank_parity": str(path),
                    "dense_referenced_multirun": str(dense_referenced_multirun_path)
                    if dense_referenced_multirun_path
                    else None,
                },
                sort_keys=True,
            ),
            {
                "routes": [
                    route
                    for route, passed in [
                        ("rank_parity", quality_parity_pass),
                        ("dense_referenced_multirun", dense_multirun_pass),
                    ]
                    if passed
                ],
                "rank_parity": {
                    "overall_pass": payload.get("pass"),
                    "selected_arm": payload.get("selected_arm", arm),
                    "missing_arm": missing_arm,
                    "ensemble_quality": gates.get("ensemble_quality"),
                    "ensemble_delta": payload.get("ensemble_holdout_delta_lora_minus_dense"),
                },
                "dense_referenced_multirun": dense_multirun,
            },
        ),
        GoalCheck(
            "stability parity",
            stability_parity_pass or dense_multirun_pass,
            json.dumps(
                {
                    "rank_parity": str(path),
                    "dense_referenced_multirun": str(dense_referenced_multirun_path)
                    if dense_referenced_multirun_path
                    else None,
                },
                sort_keys=True,
            ),
            {
                "routes": [
                    route
                    for route, passed in [
                        ("rank_parity", stability_parity_pass),
                        ("dense_referenced_multirun", dense_multirun_pass),
                    ]
                    if passed
                ],
                "rank_parity": {
                    "selected_arm": payload.get("selected_arm", arm),
                    "missing_arm": missing_arm,
                    "spearman": payload.get("spearman"),
                    "topk_overlap": payload.get("topk_overlap"),
                    "selected_regret": payload.get("selected_regret"),
                    "gates": {
                        "spearman": gates.get("spearman"),
                        "topk_overlap": gates.get("topk_overlap"),
                        "selected_regret": gates.get("selected_regret"),
                    },
                },
                "dense_referenced_multirun": dense_multirun,
            },
        ),
        GoalCheck(
            "speed parity",
            speed_parity_pass or dense_multirun_pass,
            json.dumps(
                {
                    "rank_parity": str(path),
                    "dense_referenced_multirun": str(dense_referenced_multirun_path)
                    if dense_referenced_multirun_path
                    else None,
                },
                sort_keys=True,
            ),
            {
                "routes": [
                    route
                    for route, passed in [
                        ("rank_parity", speed_parity_pass),
                        ("dense_referenced_multirun", dense_multirun_pass),
                    ]
                    if passed
                ],
                "rank_parity": {
                    "selected_arm": payload.get("selected_arm", arm),
                    "missing_arm": missing_arm,
                    "speed_ratio_lora_over_dense": payload.get("speed_ratio_lora_over_dense"),
                    "gate": gates.get("speed"),
                },
                "dense_referenced_multirun": dense_multirun,
            },
        ),
    ]


def check_prompt_robustness(
    path: Path | None,
    payload: dict | None,
    *,
    dense_referenced_path: Path | None = None,
    dense_referenced_payload: dict | None = None,
) -> GoalCheck:
    dense_pass = dense_referenced_multirun_pass(dense_referenced_payload)
    if payload is None and not dense_pass:
        return GoalCheck("prompt robustness", False, str(path) if path else "missing", "missing prompt robustness report")
    gate = {} if payload is None else payload.get("gate", payload)
    old_pass = bool(gate.get("pass"))
    routes = []
    if old_pass:
        routes.append("prompt_robustness_report")
    if dense_pass:
        routes.append("dense_referenced_multirun")
    return GoalCheck(
        "prompt robustness",
        old_pass or dense_pass,
        json.dumps(
            {
                "prompt_robustness": str(path) if path else None,
                "dense_referenced_multirun": str(dense_referenced_path) if dense_referenced_path else None,
            },
            sort_keys=True,
        ),
        {
            "pass": old_pass or dense_pass,
            "routes": routes,
            "prompt_robustness_report": {
                "pass": gate.get("pass"),
                "valid_prompt_variants": gate.get("valid_prompt_variants"),
                "passing_prompt_variants": gate.get("passing_prompt_variants"),
                "min_valid_prompts": gate.get("min_valid_prompts"),
            },
            "dense_referenced_multirun": dense_referenced_multirun_detail(dense_referenced_path, dense_referenced_payload),
        },
    )


def check_drift(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("drift parity", False, str(path) if path else "missing", "missing drift evidence")
    passed = bool(payload.get("pass") or payload.get("drift_pass"))
    return GoalCheck("drift parity", passed, str(path), payload.get("gates", payload))


def check_eval_validity(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("eval validity", False, str(path) if path else "missing", "missing result validity audit")
    return GoalCheck(
        "eval validity",
        bool(payload.get("pass")),
        str(path),
        payload.get("failed", payload),
    )


def check_score_sanity(path: Path | None, payload: dict | None) -> GoalCheck:
    if payload is None:
        return GoalCheck("score sanity", False, str(path) if path else "missing", "missing score sanity audit")
    return GoalCheck(
        "score sanity",
        bool(payload.get("pass")),
        str(path),
        {
            "pass": payload.get("pass"),
            "failed": payload.get("failed", []),
            "thresholds": payload.get("thresholds", {}),
        },
    )


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
    local_baseline_path = getattr(args, "local_baseline_audit", None)
    local_baseline = read_json(local_baseline_path)
    upstream_baseline = read_json(getattr(args, "upstream_baseline_audit", None))
    parity = read_json(args.parity_report)
    backend = read_json(args.backend_gate)
    confirmation = read_json(args.confirmation_gate)
    dense_confirmation = read_json(getattr(args, "dense_confirmation_gate", None))
    search_quality = read_json(getattr(args, "search_quality_confirmation", None))
    family_state_provenance = read_json(getattr(args, "family_state_provenance", None))
    multirun = read_json(getattr(args, "multirun_gate", None))
    dense_referenced_multirun = read_json(getattr(args, "dense_referenced_multirun_gate", None))
    prompt = read_json(args.prompt_robustness)
    drift = read_json(args.drift_report)
    eval_validity = read_json(args.eval_validity)
    score_sanity = read_json(getattr(args, "score_sanity", None))
    checks.append(
        check_official_baseline(
            local_baseline_path=local_baseline_path,
            local_baseline_payload=local_baseline,
            upstream_path=getattr(args, "upstream_baseline_audit", None),
            upstream_payload=upstream_baseline,
        )
    )
    checks.extend(
        check_parity_report(
            args.parity_report,
            parity,
            arm=getattr(args, "parity_arm", "lora"),
            dense_referenced_multirun_path=getattr(args, "dense_referenced_multirun_gate", None),
            dense_referenced_multirun_payload=dense_referenced_multirun,
        )
    )
    checks.append(
        check_accelerated_route(
            backend_path=args.backend_gate,
            backend_payload=backend,
            same_family_confirmation_path=args.confirmation_gate,
            same_family_confirmation_payload=confirmation,
            dense_confirmation_path=getattr(args, "dense_confirmation_gate", None),
            dense_confirmation_payload=dense_confirmation,
            search_quality_path=getattr(args, "search_quality_confirmation", None),
            search_quality_payload=search_quality,
        )
    )
    checks.append(check_family_state_provenance(getattr(args, "family_state_provenance", None), family_state_provenance))
    checks.append(
        check_multirun_gate(
            getattr(args, "multirun_gate", None),
            multirun,
            dense_referenced_path=getattr(args, "dense_referenced_multirun_gate", None),
            dense_referenced_payload=dense_referenced_multirun,
        )
    )
    checks.append(
        check_prompt_robustness(
            args.prompt_robustness,
            prompt,
            dense_referenced_path=getattr(args, "dense_referenced_multirun_gate", None),
            dense_referenced_payload=dense_referenced_multirun,
        )
    )
    checks.append(check_drift(args.drift_report, drift))
    checks.append(check_eval_validity(args.eval_validity, eval_validity))
    checks.append(check_score_sanity(getattr(args, "score_sanity", None), score_sanity))
    checks.append(check_adapter_convenience(args.adapter_run))
    rows = [asdict(check) for check in checks]
    failed = [row["requirement"] for row in rows if not row["passed"]]
    return {
        "pass": all(row["passed"] for row in rows),
        "failed": failed,
        "checks": rows,
        "next_actions": sorted((next_action(requirement) for requirement in failed), key=lambda row: (row["priority"], row["requirement"])),
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Optimus Goal Audit",
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
    if summary.get("next_actions"):
        lines.extend(
            [
                "",
                "## Next Actions",
                "",
                "| requirement | action | command |",
                "| --- | --- | --- |",
            ]
        )
        for row in summary["next_actions"]:
            lines.append(f"| {row['requirement']} | {row['action']} | `{row['command']}` |")
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit whether the end-to-end LoRA perturbation-search goal is met.")
    parser.add_argument("--local-baseline-audit", type=Path)
    parser.add_argument("--upstream-baseline-audit", type=Path)
    parser.add_argument("--parity-report", type=Path)
    parser.add_argument("--parity-arm", default="lora")
    parser.add_argument("--backend-gate", type=Path)
    parser.add_argument("--confirmation-gate", type=Path)
    parser.add_argument("--dense-confirmation-gate", type=Path)
    parser.add_argument("--search-quality-confirmation", type=Path)
    parser.add_argument("--family-state-provenance", type=Path)
    parser.add_argument("--multirun-gate", type=Path)
    parser.add_argument("--dense-referenced-multirun-gate", type=Path)
    parser.add_argument("--prompt-robustness", type=Path)
    parser.add_argument("--drift-report", type=Path)
    parser.add_argument("--eval-validity", type=Path)
    parser.add_argument("--score-sanity", type=Path)
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
