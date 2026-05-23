from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


UPSTREAM_COUNTDOWN_MODEL = "allenai/Olmo-3-7B-Instruct"
UPSTREAM_COUNTDOWN_SIGMAS = [0.0005, 0.001, 0.002]
UPSTREAM_COUNTDOWN_TOP_K_RATIOS = [0.04, 0.01, 0.05, 0.1]
UPSTREAM_COUNTDOWN_DATA_SOURCE = "VsonicV/es-fine-tuning-paper/countdown/data/countdown.json"
UPSTREAM_COUNTDOWN_SCORE_METRIC = "upstream_countdown_reward"
UPSTREAM_COUNTDOWN_VOTE_METRIC = "valid_numeric_majority_vote"


@dataclass
class Check:
    name: str
    passed: bool
    actual: Any
    expected: Any
    note: str = ""


def _norm_float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if isinstance(values, str):
        return [float(x) for x in values.split(",") if x.strip()]
    return [float(x) for x in values]


def _norm_int_list(values: Any) -> list[int]:
    if values is None:
        return []
    if isinstance(values, str):
        return [int(x) for x in values.split(",") if x.strip()]
    return [int(x) for x in values]


def _check(name: str, actual: Any, expected: Any, *, note: str = "") -> Check:
    return Check(name=name, passed=actual == expected, actual=actual, expected=expected, note=note)


def _check_bool(name: str, passed: bool, actual: Any, expected: Any, *, note: str = "") -> Check:
    return Check(name=name, passed=bool(passed), actual=actual, expected=expected, note=note)


def _check_one_of(name: str, actual: Any, expected: set[Any], *, preferred: Any, note: str = "") -> Check:
    return Check(name=name, passed=actual in expected, actual=actual, expected=preferred, note=note)


def _is_upstream_countdown_data(value: Any) -> bool:
    if value == UPSTREAM_COUNTDOWN_DATA_SOURCE:
        return True
    text = str(value or "")
    return (
        "VsonicV/es-fine-tuning-paper" in text
        or "countdown_official.json" in text
        or text.endswith("es-fine-tuning-paper/countdown/data/countdown.json")
    )


def upstream_countdown_ensemble_ks(population: int) -> list[int]:
    return sorted({max(1, int(population * ratio)) for ratio in UPSTREAM_COUNTDOWN_TOP_K_RATIOS})


def audit_upstream_countdown_run(summary: dict) -> list[Check]:
    population = int(summary.get("population") or 0)
    expected_ks = upstream_countdown_ensemble_ks(population) if population else []
    screen_prompts = summary.get("screen_prompts")
    holdout_prompts = summary.get("holdout_prompts")
    screen_unique = summary.get("screen_unique_semantic_prompts")
    holdout_unique = summary.get("holdout_unique_semantic_prompts")
    targets = str(summary.get("targets", ""))
    data_source = summary.get("data_source", summary.get("data"))
    checks = [
        _check("model", summary.get("model"), UPSTREAM_COUNTDOWN_MODEL),
        _check_bool(
            "upstream_countdown_data",
            _is_upstream_countdown_data(data_source),
            data_source,
            UPSTREAM_COUNTDOWN_DATA_SOURCE,
        ),
        _check("perturbation_backend", summary.get("perturbation_backend"), "dense"),
        _check("family", summary.get("family"), "dense_gaussian"),
        _check_bool("full_parameter_targets", targets in {"all", "all_params", "*"}, targets, "all_params"),
        _check_one_of(
            "dense_noise_mode",
            summary.get("dense_noise_mode"),
            {"upstream", "paper"},
            preferred="upstream",
            note="'paper' is accepted only for older saved summaries",
        ),
        _check("candidate_score_metric", summary.get("candidate_score_metric"), UPSTREAM_COUNTDOWN_SCORE_METRIC),
        _check("ensemble_vote_metric", summary.get("ensemble_vote_metric"), UPSTREAM_COUNTDOWN_VOTE_METRIC),
        _check("train_or_screen_samples", screen_prompts, 200),
        _check("population", population, 5000),
        _check(
            "sigma_values",
            sorted(_norm_float_list(summary.get("sigma_values"))),
            sorted(UPSTREAM_COUNTDOWN_SIGMAS),
        ),
        _check("max_new_tokens", summary.get("max_new_tokens"), 1024),
        _check_one_of(
            "prompt_variant",
            summary.get("prompt_variant"),
            {"upstream", "paper"},
            preferred="upstream",
            note="'paper' is accepted only for older saved summaries",
        ),
        _check("use_chat_template", summary.get("use_chat_template"), True),
        _check("ensemble_ks", sorted(_norm_int_list(summary.get("ensemble_ks"))), expected_ks),
        _check("screen_holdout_overlap", summary.get("screen_holdout_overlap"), 0),
        _check_bool(
            "screen_semantic_uniqueness",
            screen_unique == screen_prompts,
            screen_unique,
            screen_prompts,
            note="screen split should not repeat semantic Countdown examples",
        ),
        _check_bool(
            "holdout_semantic_uniqueness",
            holdout_unique == holdout_prompts,
            holdout_unique,
            holdout_prompts,
            note="holdout split should not repeat semantic Countdown examples",
        ),
        _check_bool(
            "ensemble_metric_present",
            bool(summary.get("ensemble_holdout")),
            bool(summary.get("ensemble_holdout")),
            True,
        ),
    ]
    return checks


def summarize(checks: list[Check]) -> dict:
    return {
        "pass": all(check.passed for check in checks),
        "failed": [check.name for check in checks if not check.passed],
        "checks": [asdict(check) for check in checks],
    }


def render_markdown(summary: dict) -> str:
    lines = [
        "# Upstream Baseline Audit",
        "",
        f"Pass: `{str(summary['pass']).lower()}`",
        "",
        "| check | pass | actual | expected |",
        "| --- | ---: | --- | --- |",
    ]
    for check in summary["checks"]:
        lines.append(
            f"| {check['name']} | {str(check['passed']).lower()} | "
            f"`{json.dumps(check['actual'], sort_keys=True)}` | "
            f"`{json.dumps(check['expected'], sort_keys=True)}` |"
        )
    lines.append("")
    return "\n".join(lines)


def load_summary(run_dir: Path) -> dict:
    path = run_dir / "summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    return json.loads(path.read_text())


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit whether a run matches upstream Countdown baseline semantics.")
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    audit = summarize(audit_upstream_countdown_run(load_summary(args.run)))
    if args.out:
        args.out.mkdir(parents=True, exist_ok=True)
        (args.out / "summary.json").write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n")
        (args.out / "report.md").write_text(render_markdown(audit))
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


OFFICIAL_COUNTDOWN_MODEL = UPSTREAM_COUNTDOWN_MODEL
OFFICIAL_COUNTDOWN_SIGMAS = UPSTREAM_COUNTDOWN_SIGMAS
OFFICIAL_COUNTDOWN_TOP_K_RATIOS = UPSTREAM_COUNTDOWN_TOP_K_RATIOS
OFFICIAL_COUNTDOWN_DATA_SOURCE = UPSTREAM_COUNTDOWN_DATA_SOURCE
OFFICIAL_COUNTDOWN_SCORE_METRIC = UPSTREAM_COUNTDOWN_SCORE_METRIC
OFFICIAL_COUNTDOWN_VOTE_METRIC = UPSTREAM_COUNTDOWN_VOTE_METRIC
official_countdown_ensemble_ks = upstream_countdown_ensemble_ks
audit_official_countdown_run = audit_upstream_countdown_run
