import json
from pathlib import Path

from randopt_lora_lab.selector_union_audit import analyze, shortlist_for_run


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def dense(seed: int, score: float) -> dict:
    return {"candidate": f"dense_gaussian:seed{seed}:s0.001:sign1", "exact_mean": score}


def proposal(seed: int, selection: float, exact: float) -> dict:
    return {
        "candidate": f"activation_spectral_lora_c2:seed{seed}:s0.001:sign1",
        "exact_mean": exact,
        "selection_score": selection,
        "mean_condition_selection_score": selection,
        "min_condition_selection_score": selection,
        "mean_exact_lift_vs_base": selection,
        "min_exact_lift_vs_base": selection,
        "max_malformed_regression_vs_base": 0.0,
        "max_cap_hit_regression_vs_base": 0.0,
        "sigma": 0.001,
        "sign": 1,
    }


def condition(seed: int, variant: str, exact: float) -> dict:
    return {
        "candidate": f"activation_spectral_lora_c2:seed{seed}:s0.001:sign1",
        "prompt_variant": variant,
        "exact_mean": exact,
        "exact_lift_vs_base": exact,
        "condition_selection_score": exact,
        "malformed_mean": 0.0,
        "cap_hit_mean": 0.0,
        "output_tokens": 10,
    }


def make_run(root: Path) -> None:
    write_jsonl(
        root / "dense" / "candidate_summary.jsonl",
        [
            dense(1, 1.0),
            dense(2, 0.5),
            dense(3, 0.0),
        ],
    )
    write_jsonl(
        root / "vllm" / "candidate_summary.jsonl",
        [
            proposal(1, 0.0, 0.0),
            proposal(2, 0.5, 0.5),
            proposal(3, 1.0, 1.0),
        ],
    )
    write_jsonl(
        root / "vllm" / "candidate_condition_summary.jsonl",
        [
            condition(1, "default", 1.0),
            condition(1, "reordered", 0.0),
            condition(1, "xml", 0.0),
            condition(2, "default", 0.5),
            condition(2, "reordered", 0.5),
            condition(2, "xml", 0.5),
            condition(3, "default", 0.0),
            condition(3, "reordered", 1.0),
            condition(3, "xml", 1.0),
        ],
    )


def test_round_robin_union_recovers_dense_best_when_current_selector_misses(tmp_path: Path):
    run = tmp_path / "run"
    make_run(run)

    summary = analyze([run], ks=[1, 2])

    current = summary["per_run"]["run"]["policies"]["current_selection"]
    assert current["dense_best_rank"] == 3
    assert current["rows"][0]["contains_dense_best"] is False

    union = summary["per_run"]["run"]["policies"]["prompt_exact_rr"]
    assert union["dense_best_rank"] == 1
    assert union["rows"][0]["contains_dense_best"] is True
    assert summary["verdict"]["pass_at_k8"] is True
    assert summary["verdict"]["first_policy_with_max_regret_at_most"]["zero"]["k"] == 1


def test_shortlist_writer_preserves_candidate_rows_with_selector_metadata(tmp_path: Path):
    run = tmp_path / "run"
    make_run(run)

    rows = shortlist_for_run(run, policy="prompt_exact_rr", k=2)

    assert [row["candidate"] for row in rows] == [
        "activation_spectral_lora_c2:seed1:s0.001:sign1",
        "activation_spectral_lora_c2:seed3:s0.001:sign1",
    ]
    assert rows[0]["selector_union_policy"] == "prompt_exact_rr"
    assert rows[0]["selector_union_rank"] == 1
    assert rows[0]["selector_union_dense_exact_offline"] == 1.0
