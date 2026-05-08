import json
from pathlib import Path

from randopt_lora_lab.family_sweep_report import aggregate


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


def write_variant(root: Path, *, sparse_delta: float = 0.02, sparse_valid: bool = True) -> None:
    for arm in ["dense", "factor", "sparse_d0p25"]:
        write_json(root / arm / "summary.json", {"holdout_prompts": 128})
        write_json(root / arm / "validity" / "summary.json", {"pass": sparse_valid if arm == "sparse_d0p25" else True})
    write_json(
        root / "parity" / "summary.json",
        {
            "comparisons": {
                "factor": {
                    "dense_best_ensemble_holdout_exact": 0.10,
                    "lora_best_ensemble_holdout_exact": 0.10,
                    "selected_regret": 0.0,
                    "spearman": 0.9,
                    "topk_overlap": 8,
                    "speed_ratio_lora_over_dense": 2.0,
                    "lora_pick_cap_hit_mean": 0.0,
                    "lora_pick_malformed_mean": 0.01,
                },
                "sparse_d0p25": {
                    "dense_best_ensemble_holdout_exact": 0.10,
                    "lora_best_ensemble_holdout_exact": 0.10 + sparse_delta,
                    "selected_regret": 0.0,
                    "spearman": 0.8,
                    "topk_overlap": 6,
                    "speed_ratio_lora_over_dense": 1.5,
                    "lora_pick_cap_hit_mean": 0.0,
                    "lora_pick_malformed_mean": 0.01,
                },
            }
        },
    )


def test_family_sweep_passes_when_same_sparse_arm_beats_factor_across_variants(tmp_path: Path):
    write_variant(tmp_path / "default")
    write_variant(tmp_path / "reordered")

    summary = aggregate(
        [("default", tmp_path / "default"), ("reordered", tmp_path / "reordered")],
        min_variants=2,
        min_improvement_examples=2,
    )

    assert summary["pass"]
    assert summary["arm_pass"] == {"sparse_d0p25": True}
    assert [row["pass"] for row in summary["rows"] if row["arm"] == "sparse_d0p25"] == [True, True]


def test_family_sweep_fails_when_improvement_is_too_small(tmp_path: Path):
    write_variant(tmp_path / "default", sparse_delta=0.005)
    write_variant(tmp_path / "reordered", sparse_delta=0.005)

    summary = aggregate(
        [("default", tmp_path / "default"), ("reordered", tmp_path / "reordered")],
        min_variants=2,
        min_improvement_examples=2,
    )

    assert not summary["pass"]
    assert "no_family_beats_baseline_across_variants" in summary["failed"]


def test_family_sweep_fails_invalid_arm(tmp_path: Path):
    write_variant(tmp_path / "default", sparse_valid=False)
    write_variant(tmp_path / "reordered")

    summary = aggregate(
        [("default", tmp_path / "default"), ("reordered", tmp_path / "reordered")],
        min_variants=2,
        min_improvement_examples=2,
    )

    assert not summary["pass"]
    assert summary["arm_pass"] == {"sparse_d0p25": False}
