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


def read_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def select_parity_arm(summary: dict, arm: str) -> dict:
    comparisons = summary.get("comparisons")
    if not comparisons:
        return summary
    if arm not in comparisons:
        return {
            "pass": False,
            "missing_arm": arm,
            "available_arms": sorted(comparisons),
            "gates": {},
        }
    selected = dict(comparisons[arm])
    selected["arm"] = arm
    return selected


def zero_regret_row(confirmation: dict) -> dict | None:
    zero_k = confirmation.get("zero_regret_k")
    if zero_k is None:
        return None
    rows = confirmation.get("rows") or []
    eligible = [row for row in rows if int(row.get("k", 0)) >= int(zero_k)]
    return min(eligible, key=lambda row: int(row["k"])) if eligible else None


def validity_pass(run_dir: Path, arms: list[str]) -> tuple[bool, dict[str, bool]]:
    by_arm = {}
    for arm in arms:
        path = run_dir / arm / "validity" / "summary.json"
        by_arm[arm] = bool(path.exists() and read_json(path).get("pass"))
    return all(by_arm.values()), by_arm


def prompt_variant_count(vllm_summary: dict) -> int:
    variants = vllm_summary.get("screen_selection_prompt_variants")
    if variants is None:
        variants = vllm_summary.get("prompt_variants") or []
    return len(variants)


def load_run(run_dir: Path, *, parity_arm: str, validity_arms: list[str]) -> dict:
    parity_raw = read_json(run_dir / "parity" / "summary.json")
    parity = select_parity_arm(parity_raw, parity_arm)
    confirmation = read_json(run_dir / "confirmation" / "summary.json")
    vllm = read_json(run_dir / "vllm_spectral" / "summary.json")
    valid, valid_by_arm = validity_pass(run_dir, validity_arms)
    confirm_gate = confirmation.get("gate", confirmation)
    zero_row = zero_regret_row(confirmation)
    gates = parity.get("gates", {})
    return {
        "run_dir": str(run_dir),
        "parity_arm": parity_arm,
        "validity_pass": valid,
        "validity_by_arm": valid_by_arm,
        "parity_pass": bool(parity.get("pass")),
        "parity_missing_arm": parity.get("missing_arm"),
        "parity_gates": gates,
        "spearman": parity.get("spearman"),
        "topk_overlap": parity.get("topk_overlap"),
        "selected_regret": parity.get("selected_regret"),
        "ensemble_delta": parity.get("ensemble_holdout_delta_lora_minus_dense"),
        "speed_ratio_lora_over_dense": parity.get("speed_ratio_lora_over_dense"),
        "mutation_ratio_lora_over_dense": parity.get("mutation_s_ratio_lora_over_dense"),
        "confirmation_pass": bool(confirm_gate.get("pass")),
        "confirmation_failed": confirm_gate.get("failed", []),
        "zero_regret_k": confirmation.get("zero_regret_k"),
        "best_recovered_k": confirmation.get("best_recovered_k"),
        "eval_only_speedup_at_zero_regret": None
        if zero_row is None
        else zero_row.get("eval_only_speedup_vs_trusted_full"),
        "full_without_load_speedup_at_zero_regret": None
        if zero_row is None
        else zero_row.get("full_without_peft_load_speedup_vs_trusted_full"),
        "prompt_variant_count": prompt_variant_count(vllm),
        "selection_prompt_variants": vllm.get("screen_selection_prompt_variants") or [],
        "stress_prompt_variants": vllm.get("screen_stress_prompt_variants") or [],
        "base_screen_exact": vllm.get("base_screen_exact"),
        "base_screen_cap_hit_by_prompt": {
            key: value.get("cap_hit_mean")
            for key, value in (vllm.get("base_screen_by_prompt") or {}).items()
        },
        "base_screen_malformed_by_prompt": {
            key: value.get("malformed_mean")
            for key, value in (vllm.get("base_screen_by_prompt") or {}).items()
        },
    }


def numeric_values(rows: list[dict], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key) is not None]


def aggregate(
    run_dirs: list[Path],
    *,
    parity_arm: str = "lora",
    validity_arms: list[str] | None = None,
    min_runs: int = 2,
    min_prompt_variants: int = 2,
    max_zero_regret_k: int = 8,
    min_full_without_load_speedup: float = 1.0,
) -> dict:
    validity_arms = validity_arms or ["dense", "control", "spectral"]
    runs = [load_run(path, parity_arm=parity_arm, validity_arms=validity_arms) for path in run_dirs]
    spearman = numeric_values(runs, "spearman")
    regret = numeric_values(runs, "selected_regret")
    ensemble = numeric_values(runs, "ensemble_delta")
    full_speedups = numeric_values(runs, "full_without_load_speedup_at_zero_regret")
    checks = [
        GateCheck("min_runs", len(runs) >= min_runs, {"runs": len(runs), "min_runs": min_runs}),
        GateCheck(
            "all_validity_pass",
            all(row["validity_pass"] for row in runs),
            {row["run_dir"]: row["validity_by_arm"] for row in runs},
        ),
        GateCheck(
            "all_quality_parity_pass",
            all(row["parity_pass"] for row in runs),
            {
                row["run_dir"]: {
                    "pass": row["parity_pass"],
                    "missing_arm": row["parity_missing_arm"],
                    "gates": row["parity_gates"],
                }
                for row in runs
            },
        ),
        GateCheck(
            "all_confirmation_pass",
            all(row["confirmation_pass"] for row in runs),
            {row["run_dir"]: row["confirmation_failed"] for row in runs},
        ),
        GateCheck(
            "zero_regret_within_k",
            all(row["zero_regret_k"] is not None and int(row["zero_regret_k"]) <= max_zero_regret_k for row in runs),
            {row["run_dir"]: row["zero_regret_k"] for row in runs},
        ),
        GateCheck(
            "positive_full_speedup",
            bool(full_speedups)
            and len(full_speedups) == len(runs)
            and min(full_speedups) >= min_full_without_load_speedup,
            {
                "min_full_without_load_speedup": min(full_speedups) if full_speedups else None,
                "threshold": min_full_without_load_speedup,
            },
        ),
        GateCheck(
            "prompt_robust_selection",
            all(int(row["prompt_variant_count"]) >= min_prompt_variants for row in runs),
            {
                row["run_dir"]: {
                    "prompt_variant_count": row["prompt_variant_count"],
                    "selection_prompt_variants": row["selection_prompt_variants"],
                    "min_prompt_variants": min_prompt_variants,
                }
                for row in runs
            },
        ),
    ]
    failed = [check.check for check in checks if not check.passed]
    return {
        "kind": "spectral_vllm_multirun_gate",
        "pass": not failed,
        "failed": failed,
        "thresholds": {
            "parity_arm": parity_arm,
            "min_runs": min_runs,
            "min_prompt_variants": min_prompt_variants,
            "max_zero_regret_k": max_zero_regret_k,
            "min_full_without_load_speedup": min_full_without_load_speedup,
        },
        "aggregate": {
            "runs": len(runs),
            "parity_pass_count": sum(1 for row in runs if row["parity_pass"]),
            "confirmation_pass_count": sum(1 for row in runs if row["confirmation_pass"]),
            "validity_pass_count": sum(1 for row in runs if row["validity_pass"]),
            "min_spearman": min(spearman) if spearman else None,
            "mean_spearman": mean(spearman) if spearman else None,
            "max_selected_regret": max(regret) if regret else None,
            "min_ensemble_delta": min(ensemble) if ensemble else None,
            "min_full_without_load_speedup": min(full_speedups) if full_speedups else None,
        },
        "checks": [asdict(check) for check in checks],
        "runs": runs,
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Spectral vLLM Multi-Run Gate",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
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
            "| run | parity | confirmation | zero-regret k | full speedup | prompts | spearman | regret | ensemble delta |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in summary["runs"]:
        lines.append(
            "| {run} | {parity} | {confirmation} | {zero} | {speed} | {prompts} | {spearman} | {regret} | {ensemble} |".format(
                run=f"`{row['run_dir']}`",
                parity=str(row["parity_pass"]).lower(),
                confirmation=str(row["confirmation_pass"]).lower(),
                zero=row["zero_regret_k"],
                speed=row["full_without_load_speedup_at_zero_regret"],
                prompts=row["prompt_variant_count"],
                spearman=row["spearman"],
                regret=row["selected_regret"],
                ensemble=row["ensemble_delta"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def parse_validity_arms(text: str) -> list[str]:
    return [part.strip() for part in text.split(",") if part.strip()]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Aggregate strict gates across spectral vLLM confirmation runs.")
    parser.add_argument("--run", type=Path, action="append", required=True, help="Run root containing parity/confirmation/vllm_spectral")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--parity-arm", default="lora")
    parser.add_argument("--validity-arms", default="dense,control,spectral")
    parser.add_argument("--min-runs", type=int, default=2)
    parser.add_argument("--min-prompt-variants", type=int, default=2)
    parser.add_argument("--max-zero-regret-k", type=int, default=8)
    parser.add_argument("--min-full-without-load-speedup", type=float, default=1.0)
    args = parser.parse_args(argv)

    summary = aggregate(
        args.run,
        parity_arm=args.parity_arm,
        validity_arms=parse_validity_arms(args.validity_arms),
        min_runs=args.min_runs,
        min_prompt_variants=args.min_prompt_variants,
        max_zero_regret_k=args.max_zero_regret_k,
        min_full_without_load_speedup=args.min_full_without_load_speedup,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
