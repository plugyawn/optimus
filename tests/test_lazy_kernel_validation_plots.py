from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from scripts import plot_lazy_kernel_validation as plot
from scripts.validate_lazy_kernel_validation import validate


PNG_HEADER = b"\x89PNG\r\n\x1a\n"


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))


def _run_dir(root: Path, name: str, *, kind: str, population: int, best_id: str) -> Path:
    run = root / name
    run.mkdir()
    (run / "summary.json").write_text(
        json.dumps(
            {
                "kind": kind,
                "population": population,
                "basis_rank": 64,
                "effective_rank": 16,
                "scale_multiplier": 2.0,
                "targets": ["q_proj", "v_proj"],
                "base_final_score": 0.125,
                "best_candidate_final_score": 0.25,
                "confirmed_best_candidate_final_score": 0.25,
                "best_candidate_id": best_id,
                "confirmed_best_candidate_id": best_id,
                "mixed_candidate_sec": 0.5,
                "confirmed_mixed_candidate_sec": 0.4,
                "lazy_timing": {
                    "elapsed_s": 10.0,
                    "lazy_delta_time_s": 4.0,
                    "lazy_kernel_time_s": 1.5,
                    "lazy_meta_time_s": 0.5,
                    "lazy_stack_time_s": 0.25,
                    "output_tokens": 1000,
                    "qx_time_s": 0.0,
                },
            },
            sort_keys=True,
        )
    )
    return run


def _score_row(candidate_id: str, exact: float, *, stage: str = "k1_final_replay") -> dict:
    return {
        "candidate_id": candidate_id,
        "split": "final",
        "selection_stage": stage,
        "aggregate_metrics": {"exact": exact},
        "sample_count": 8,
    }


def test_plotter_reports_primary_and_confirmed_parity(tmp_path: Path, monkeypatch) -> None:
    trusted = _run_dir(tmp_path, "trusted", kind="vllm_subspace_adapter_k1_final_replay", population=2, best_id="b")
    candidate = _run_dir(tmp_path, "candidate", kind="vllm_lazy_k1_final_replay", population=1024, best_id="b")
    _write_jsonl(trusted / "candidate_scores.jsonl", [_score_row("a", 0.125), _score_row("b", 0.25)])
    _write_jsonl(candidate / "candidate_scores.jsonl", [_score_row("a", 0.125), _score_row("b", 0.125)])
    _write_jsonl(
        trusted / "confirmed_candidate_scores.jsonl",
        [_score_row("b", 0.25, stage="k1_final_confirmed_chunk1")],
    )
    _write_jsonl(
        candidate / "confirmed_candidate_scores.jsonl",
        [_score_row("b", 0.25, stage="k1_final_confirmed_chunk1")],
    )

    def fake_plot(path: Path, *args, **kwargs) -> None:
        path.write_bytes(PNG_HEADER + b"fake")

    monkeypatch.setattr(plot, "_plot_quality", fake_plot)
    monkeypatch.setattr(plot, "_plot_throughput", fake_plot)
    monkeypatch.setattr(plot, "_plot_lazy_timing", fake_plot)
    monkeypatch.setattr(plot, "_plot_parity", fake_plot)
    out = tmp_path / "plots"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "plot_lazy_kernel_validation.py",
            "--out",
            str(out),
            "--run",
            f"adapter={trusted}",
            "--run",
            f"lazy={candidate}",
            "--trusted-run",
            str(trusted),
            "--candidate-run",
            str(candidate),
        ],
    )

    assert plot.main() == 0

    summary = json.loads((out / "parity_summary.json").read_text())
    assert summary["candidate_scores"]["max_abs_score_diff"] == 0.125
    assert summary["confirmed_candidate_scores"]["exact_score_match"] is True
    assert summary["confirmed_candidate_scores"]["best_candidate_match"] is True
    with (out / "candidate_parity.csv").open(newline="") as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["abs_diff"] == "0.0"
    assert rows[1]["exact_match"] == "False"


def test_validation_report_flags_strict_parity_threshold(tmp_path: Path) -> None:
    plot_dir = tmp_path / "plots"
    plot_dir.mkdir()
    run = tmp_path / "lazy_p1024"
    run.mkdir()
    (run / "summary.json").write_text("{}")
    (plot_dir / "summary.csv").write_text(
        "label,kind,population,base_score,best_score,candidate_sec,run_dir\n"
        f"lazy,vllm_lazy_k1_final_replay,1024,0.125,0.25,0.5,{run}\n"
    )
    for name in (
        "quality.png",
        "throughput.png",
        "lazy_timing_breakdown.png",
        "candidate_score_parity.png",
        "confirmed_candidate_score_parity.png",
    ):
        (plot_dir / name).write_bytes(PNG_HEADER + b"fake")
    (plot_dir / "parity_summary.json").write_text(
        json.dumps(
            {
                "candidate_scores": {
                    "common_candidates": 2,
                    "max_abs_score_diff": 0.125,
                    "plot": "candidate_score_parity.png",
                },
                "confirmed_candidate_scores": {
                    "common_candidates": 1,
                    "max_abs_score_diff": 0.0,
                    "plot": "confirmed_candidate_score_parity.png",
                },
            }
        )
    )

    report = validate(
        argparse.Namespace(
            plot_dir=plot_dir,
            min_common_candidates=1,
            max_candidate_score_diff=0.0,
            max_confirmed_score_diff=0.0,
            require_p1024=True,
            require_positive_p1024=True,
            require_confirmed_parity=True,
        )
    )

    assert report["status"] == "fail"
    assert any("candidate parity max_abs_score_diff" in failure for failure in report["failures"])
