from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.materialize import run_dirs_from_args


def write_search_run(root: Path, population: int = 128) -> Path:
    run = root / f"search_p{population}_chunk32"
    adapter = run / "adapters" / "00000_candidate"
    adapter.mkdir(parents=True)
    (adapter / "adapter_config.json").write_text("{}\n")
    candidate = "lora:isotropic:seed1:s0.0075:sign1:r8:tq_proj,v_proj"
    (run / "summary.json").write_text(
        json.dumps(
            {
                "kind": "vllm_lora_search",
                "model": "Qwen/Qwen3-4B",
                "population": population,
                "top_screen": [{"candidate": candidate, "exact_mean": 0.2}],
            }
        )
        + "\n"
    )
    (run / "adapters.jsonl").write_text(
        json.dumps({"candidate": candidate, "path": "/stale/path/00000_candidate"}) + "\n"
    )
    return run


def test_materialize_selected_adapter_mode_copies_selected_adapter(tmp_path: Path):
    root = tmp_path / "runs"
    out = tmp_path / "materialized"
    write_search_run(root, 128)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "materialize-selected",
            "--root",
            str(root),
            "--out-root",
            str(out),
            "--mode",
            "adapter",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["rows"][0]["population"] == 128
    assert (out / "p128" / "adapter_config.json").exists()
    assert json.loads((out / "manifest.json").read_text())["rows"][0]["output_dir"].endswith("/p128")


def test_materialize_root_discovery_ignores_incomplete_runs(tmp_path: Path):
    complete = write_search_run(tmp_path / "runs", 256)
    incomplete = tmp_path / "runs" / "search_p512_chunk32"
    incomplete.mkdir(parents=True)
    (incomplete / "summary.json").write_text("{}\n")
    args = type("Args", (), {"run": [], "root": tmp_path / "runs"})()

    assert run_dirs_from_args(args) == [complete.resolve()]


def test_materialize_copy_replaces_existing_output(tmp_path: Path):
    root = tmp_path / "runs"
    out = tmp_path / "materialized"
    write_search_run(root, 128)
    (out / "p128").mkdir(parents=True)
    (out / "p128" / "old").write_text("old\n")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "materialize-selected",
            "--root",
            str(root),
            "--out-root",
            str(out),
            "--mode",
            "adapter",
        ],
        check=True,
    )

    assert not (out / "p128" / "old").exists()
    assert (out / "p128" / "adapter_config.json").exists()
