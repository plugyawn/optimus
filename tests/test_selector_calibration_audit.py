import json
from pathlib import Path

from randopt_lora_lab.selector_calibration_audit import analyze


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def dense(seed: int, score: float) -> dict:
    return {"candidate": f"dense_gaussian:seed{seed}:s0.001:sign1", "exact_mean": score}


def proposal(seed: int, score: float, *, default: float | None = None, reordered: float | None = None) -> dict:
    default = score if default is None else default
    reordered = score if reordered is None else reordered
    return {
        "candidate": f"activation_spectral_lora_c2:seed{seed}:s0.001:sign1",
        "exact_mean": (default + reordered) / 2.0,
        "selection_score": score,
        "mean_condition_selection_score": score,
        "min_condition_selection_score": min(default, reordered),
        "mean_exact_lift_vs_base": score,
        "min_exact_lift_vs_base": min(default, reordered),
        "max_malformed_regression_vs_base": 0.0,
        "max_cap_hit_regression_vs_base": 0.0,
        "sigma": 0.001,
        "sign": 1,
    }


def condition(seed: int, variant: str, exact: float, *, base: float = 0.0) -> dict:
    return {
        "candidate": f"activation_spectral_lora_c2:seed{seed}:s0.001:sign1",
        "prompt_variant": variant,
        "exact_mean": exact,
        "exact_lift_vs_base": exact - base,
        "condition_selection_score": exact - base,
        "malformed_mean": 0.0,
        "cap_hit_mean": 0.0,
        "output_tokens": 10,
    }


def make_run(root: Path, *, dense_scores: list[float], selector_scores: list[float]) -> None:
    seeds = list(range(1, len(dense_scores) + 1))
    write_jsonl(root / "dense" / "candidate_summary.jsonl", [dense(seed, score) for seed, score in zip(seeds, dense_scores)])
    write_jsonl(
        root / "vllm" / "candidate_summary.jsonl",
        [proposal(seed, score) for seed, score in zip(seeds, selector_scores)],
    )
    conditions = []
    for seed, score in zip(seeds, selector_scores):
        conditions.extend(
            [
                condition(seed, "default", score),
                condition(seed, "reordered", score),
                condition(seed, "xml", score),
            ]
        )
    write_jsonl(root / "vllm" / "candidate_condition_summary.jsonl", conditions)


def test_selector_calibration_reports_heldout_failure_when_train_choice_does_not_transfer(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    make_run(run_a, dense_scores=[1.0, 0.5, 0.0], selector_scores=[1.0, 0.5, 0.0])
    make_run(run_b, dense_scores=[1.0, 0.5, 0.0], selector_scores=[0.0, 0.5, 1.0])

    summary = analyze([run_a, run_b], ks=[1, 2], select_k=1)

    assert summary["kind"] == "selector_calibration_audit"
    assert summary["verdict"]["pass"] is False
    assert len(summary["folds"]) == 2
    heldout = {fold["test_run"]: fold["chosen_fixed_test"] for fold in summary["folds"]}
    assert heldout["run_b"]["dense_best_rank"] == 3


def test_selector_calibration_can_pass_when_selector_transfers(tmp_path: Path):
    run_a = tmp_path / "run_a"
    run_b = tmp_path / "run_b"
    make_run(run_a, dense_scores=[1.0, 0.5, 0.0], selector_scores=[1.0, 0.5, 0.0])
    make_run(run_b, dense_scores=[0.9, 0.2, 0.1], selector_scores=[0.8, 0.3, 0.0])

    summary = analyze([run_a, run_b], ks=[1, 2], select_k=1)

    assert summary["verdict"]["pass"] is True
    assert all(fold["chosen_fixed_test"]["rows"][0]["contains_dense_best"] for fold in summary["folds"])
