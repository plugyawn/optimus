import json
from pathlib import Path

import pytest

from randopt_lora_lab.search_quality_confirmation import analyze


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload) + "\n")


def run_summary(screen: float, strict_by_k: dict[int, float]) -> dict:
    return {
        "top_screen": [{"exact_mean": screen}],
        "strict_ensemble_holdout": [{"k": k, "exact_mean": value} for k, value in strict_by_k.items()],
    }


def speed_summary(rows: list[dict]) -> dict:
    return {"rows": rows}


def test_search_quality_passes_on_strict_holdout_and_speed(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "dense" / "summary.json", run_summary(0.9, {1: 0.4, 4: 0.5}))
    write_json(root / "confirmed" / "summary.json", run_summary(0.7, {1: 0.3, 4: 0.6}))
    write_json(root / "dense" / "validity" / "summary.json", {"pass": True})
    write_json(root / "confirmed" / "validity" / "summary.json", {"pass": True})
    write_json(
        root / "shortlist_dense_confirmation" / "summary.json",
        speed_summary(
            [
                {"k": 1, "full_without_dense_load_speedup_vs_dense_full": 2.0, "eval_only_speedup_vs_dense_full": 3.0},
                {"k": 4, "full_without_dense_load_speedup_vs_dense_full": 1.5, "eval_only_speedup_vs_dense_full": 2.0},
            ]
        ),
    )

    summary = analyze(root, max_confirm_k=4)

    assert summary["screen_delta_vs_dense"] == pytest.approx(-0.2)
    assert summary["confirmed_best_strict_holdout_exact"] == 0.6
    assert summary["rows"][1]["passes_quality"] is True
    assert summary["gate"]["pass"] is True


def test_search_quality_fails_when_validity_fails(tmp_path: Path):
    root = tmp_path / "run"
    write_json(root / "dense" / "summary.json", run_summary(0.9, {4: 0.5}))
    write_json(root / "confirmed" / "summary.json", run_summary(0.9, {4: 0.6}))
    write_json(root / "dense" / "validity" / "summary.json", {"pass": True})
    write_json(root / "confirmed" / "validity" / "summary.json", {"pass": False})
    write_json(
        root / "shortlist_dense_confirmation" / "summary.json",
        speed_summary([{"k": 4, "full_without_dense_load_speedup_vs_dense_full": 2.0}]),
    )

    summary = analyze(root, max_confirm_k=4)

    assert summary["gate"]["pass"] is False
    assert "confirmed_validity_pass" in summary["gate"]["failed"]
