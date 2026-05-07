from pathlib import Path
import json

from randopt_lora_lab.rank_sweep_report import collect_rows, render_markdown


def test_collect_rows(tmp_path: Path):
    report_dir = tmp_path / "rank8" / "report"
    report_dir.mkdir(parents=True)
    (report_dir / "summary.json").write_text(
        json.dumps(
            {
                "comparisons": {
                    "lora": {
                        "pass": False,
                        "spearman": 0.5,
                        "topk_overlap": 3,
                        "selected_regret": 0.125,
                        "speed_ratio_lora_over_dense": 2.0,
                        "dense_best_ensemble_holdout_exact": 0.1,
                        "lora_best_ensemble_holdout_exact": 0.2,
                        "ensemble_holdout_delta_lora_minus_dense": 0.1,
                        "dense_best_score": 0.3,
                        "lora_pick_score": 0.4,
                        "dense_score_at_lora_pick": 0.2,
                        "lora_pick_cap_hit_mean": 0.0,
                        "lora_pick_malformed_mean": 0.05,
                    }
                }
            }
        )
    )
    rows = collect_rows(tmp_path)
    assert rows == [
        {
            "rank": 8,
            "arm": "lora",
            "pass": False,
            "spearman": 0.5,
            "topk_overlap": 3,
            "selected_regret": 0.125,
            "speed_ratio_over_dense": 2.0,
            "dense_ensemble": 0.1,
            "arm_ensemble": 0.2,
            "ensemble_delta": 0.1,
            "dense_best_score": 0.3,
            "arm_pick_score": 0.4,
            "dense_score_at_arm_pick": 0.2,
            "arm_pick_cap_hit": 0.0,
            "arm_pick_malformed": 0.05,
        }
    ]
    assert "rank | arm" in render_markdown(rows)
