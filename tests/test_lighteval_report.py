from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.lighteval_report import add_deltas, result_rows


def write_lighteval_result(path: Path, model: str, score: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "config_general": {"model_name": model},
                "results": {
                    "ifeval": {
                        "prompt_level_strict_acc": score,
                        "ignored_text": "not numeric",
                    }
                },
            }
        )
        + "\n"
    )


def test_lighteval_report_parses_base_and_population_results(tmp_path: Path):
    root = tmp_path / "eval"
    write_lighteval_result(root / "base" / "results_2026.json", "Qwen/Qwen3-4B", 0.4)
    write_lighteval_result(root / "population_sweep" / "p128" / "results_2026.json", "runs/p128", 0.5)

    rows = add_deltas(result_rows(root))

    population = next(row for row in rows if row["population"] == 128)
    assert population["base_value"] == 0.4
    assert population["delta_vs_base"] == 0.09999999999999998


def test_lighteval_report_cli_writes_csv_markdown_and_plots(tmp_path: Path):
    root = tmp_path / "eval"
    out = tmp_path / "report"
    write_lighteval_result(root / "base" / "results_2026.json", "Qwen/Qwen3-4B", 0.4)
    write_lighteval_result(root / "population_sweep" / "p128" / "results_2026.json", "runs/p128", 0.5)
    write_lighteval_result(root / "population_sweep" / "p256" / "results_2026.json", "runs/p256", 0.55)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "lighteval-report",
            "--root",
            str(root),
            "--out",
            str(out),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["rows"] == 3
    assert (out / "report.md").exists()
    assert (out / "plots" / "ifeval__prompt_level_strict_acc.png").exists()
    assert (out / "plots" / "ifeval__prompt_level_strict_acc.pdf").exists()
    with (out / "lighteval_metrics.csv").open() as f:
        rows = list(csv.DictReader(f))
    assert {row["population"] for row in rows} == {"0", "128", "256"}
