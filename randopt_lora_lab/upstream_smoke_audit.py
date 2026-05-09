from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .reproduction_audit import (
    OFFICIAL_COUNTDOWN_DATA_SOURCE,
    OFFICIAL_COUNTDOWN_MODEL,
    OFFICIAL_COUNTDOWN_SIGMAS,
    OFFICIAL_COUNTDOWN_TOP_K_RATIOS,
    official_countdown_ensemble_ks,
)


@dataclass
class Check:
    name: str
    passed: bool
    actual: Any
    expected: Any
    note: str = ""


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def _check(name: str, actual: Any, expected: Any, *, note: str = "") -> Check:
    return Check(name=name, passed=actual == expected, actual=actual, expected=expected, note=note)


def _check_bool(name: str, passed: bool, actual: Any, expected: Any, *, note: str = "") -> Check:
    return Check(name=name, passed=bool(passed), actual=actual, expected=expected, note=note)


def _norm_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, str):
        return [float(value) for value in values.split(",") if value.strip()]
    return [float(value) for value in values]


def _norm_ratio_text(values: Any) -> str:
    if values is None:
        return ""
    return ",".join(str(float(value)).rstrip("0").rstrip(".") for value in _norm_float_list(values))


def _is_official_data(path: Any) -> bool:
    text = str(path or "")
    return (
        text == OFFICIAL_COUNTDOWN_DATA_SOURCE
        or "VsonicV/es-fine-tuning-paper" in text
        or "countdown_official.json" in text
        or text.endswith("es-fine-tuning-paper/countdown/data/countdown.json")
    )


def load_upstream_payload(run_dir: Path) -> dict[str, Any]:
    args = read_json(run_dir / "args.json")
    results = read_json(run_dir / "results.json")
    seeds_path = run_dir / "model_saves" / "top_k_seeds.json"
    seeds = read_json(seeds_path) if seeds_path.exists() else None
    return {"args": args, "results": results, "top_k_seeds": seeds}


def audit_upstream_countdown_smoke(
    payload: dict[str, Any],
    *,
    require_paper_scale: bool = False,
    min_population: int = 1,
    min_test_samples: int = 1,
) -> dict[str, Any]:
    args = payload["args"]
    results = payload["results"]
    seeds = payload.get("top_k_seeds")
    population = int(args.get("population_size") or 0)
    test_samples = int(args.get("test_samples") or results.get("test_samples") or 0)
    train_samples = int(args.get("train_samples") or results.get("train_samples") or 0)
    expected_top_k = official_countdown_ensemble_ks(population) if population else []
    actual_top_k = sorted(int(value) for value in (args.get("top_k_list") or []))
    checks = [
        _check("dataset", args.get("dataset"), "countdown"),
        _check("model", args.get("model_name"), OFFICIAL_COUNTDOWN_MODEL),
        _check_bool(
            "official_train_data",
            _is_official_data(args.get("train_data_path")),
            args.get("train_data_path"),
            OFFICIAL_COUNTDOWN_DATA_SOURCE,
        ),
        _check_bool(
            "official_test_data",
            _is_official_data(args.get("test_data_path")),
            args.get("test_data_path"),
            OFFICIAL_COUNTDOWN_DATA_SOURCE,
        ),
        _check("train_samples", train_samples, 200),
        _check_bool("population_minimum", population >= min_population, population, f">={min_population}"),
        _check_bool("test_samples_minimum", test_samples >= min_test_samples, test_samples, f">={min_test_samples}"),
        _check(
            "sigma_values",
            sorted(_norm_float_list(args.get("sigma_list", args.get("sigma_values")))),
            sorted(OFFICIAL_COUNTDOWN_SIGMAS),
        ),
        _check(
            "top_k_ratios",
            _norm_ratio_text(args.get("top_k_ratios")),
            _norm_ratio_text(OFFICIAL_COUNTDOWN_TOP_K_RATIOS),
        ),
        _check("top_k_list", actual_top_k, expected_top_k),
        _check("max_tokens", int(args.get("max_tokens") or 0), 1024),
        _check_bool("base_train_accuracy_present", "base_train_accuracy" in results, results.get("base_train_accuracy"), "present"),
        _check_bool("base_test_accuracy_present", "base_test_accuracy" in results, results.get("base_test_accuracy"), "present"),
        _check_bool(
            "top_k_perturbs_present",
            bool(results.get("top_k_perturbs")),
            len(results.get("top_k_perturbs") or []),
            ">0",
        ),
        _check_bool(
            "ensemble_results_present",
            bool(results.get("ensemble_results")),
            sorted((results.get("ensemble_results") or {}).keys()),
            "present",
        ),
        _check_bool(
            "top_k_seed_manifest_present",
            bool(seeds and seeds.get("top_k_models")),
            0 if not seeds else len(seeds.get("top_k_models") or []),
            ">0",
        ),
        _check_bool(
            "paper_scale_population",
            (not require_paper_scale) or population == 5000,
            population,
            5000,
            note="set --require-paper-scale for a true paper-scale reproduction gate",
        ),
    ]
    failed = [check.name for check in checks if not check.passed]
    smoke_pass = not [check for check in checks if not check.passed and check.name != "paper_scale_population"]
    return {
        "kind": "upstream_countdown_smoke_audit",
        "pass": not failed,
        "smoke_pass": smoke_pass,
        "paper_scale_pass": smoke_pass and population == 5000,
        "failed": failed,
        "checks": [asdict(check) for check in checks],
        "summary": {
            "model": args.get("model_name"),
            "population": population,
            "train_samples": train_samples,
            "test_samples": test_samples,
            "base_train_accuracy": results.get("base_train_accuracy"),
            "base_test_accuracy": results.get("base_test_accuracy"),
            "ensemble_results": results.get("ensemble_results", {}),
            "top_k_train_rewards": results.get("top_k_train_rewards", []),
        },
    }


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Upstream Countdown Smoke Audit",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
        f"Smoke pass: `{str(summary['smoke_pass']).lower()}`",
        f"Paper-scale pass: `{str(summary['paper_scale_pass']).lower()}`",
        "",
        "## Summary",
        "",
        "| metric | value |",
        "| --- | ---: |",
    ]
    for key, value in summary["summary"].items():
        lines.append(f"| {key} | `{json.dumps(value, sort_keys=True)}` |")
    lines.extend(["", "## Checks", "", "| check | pass | actual | expected |", "| --- | ---: | --- | --- |"])
    for check in summary["checks"]:
        lines.append(
            f"| {check['name']} | {str(check['passed']).lower()} | "
            f"`{json.dumps(check['actual'], sort_keys=True)}` | "
            f"`{json.dumps(check['expected'], sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Audit upstream RandOpt Countdown args/results artifacts.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--require-paper-scale", action="store_true")
    parser.add_argument("--min-population", type=int, default=1)
    parser.add_argument("--min-test-samples", type=int, default=1)
    args = parser.parse_args(argv)

    summary = audit_upstream_countdown_smoke(
        load_upstream_payload(args.run),
        require_paper_scale=args.require_paper_scale,
        min_population=args.min_population,
        min_test_samples=args.min_test_samples,
    )
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
