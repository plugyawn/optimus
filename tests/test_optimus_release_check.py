from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.release import build_release_checks, summary


def write_minimal_release_tree(root: Path, *, include_legacy_package: bool = False) -> tuple[Path, Path]:
    package_include = '["optimus*", "randopt_lora_lab*"]' if include_legacy_package else '["optimus*"]'
    (root / "pyproject.toml").write_text(
        f"""
[project]
name = "optimus"
version = "0.1.0"

[project.scripts]
optimus = "optimus.cli:main"

[tool.setuptools.packages.find]
include = {package_include}
""".lstrip()
    )
    package = root / "optimus"
    package.mkdir()
    (package / "__init__.py").write_text("__version__ = '0.1.0'\n")
    docs = root / "docs"
    docs.mkdir()
    for name in ["api.md", "gpu_suite.md", "index.md", "optimus_design.md", "release_checklist.md"]:
        (docs / name).write_text(f"# {name}\n\nUse `optimus` commands.\n")
    (root / "README.md").write_text("# Optimus\n\nUse `optimus` commands.\n")
    systems = root / "results" / "report" / "optimus_systems"
    systems.mkdir(parents=True)
    (systems / "report.md").write_text("# Report\n\nScreen-selected heldout transfer is checked.\n")
    (systems / "quality_scaling.csv").write_text(
        "screen_selected_holdout_exact,screen_selected_holdout_delta_vs_base,promoted_holdout_oracle_exact,promoted_holdout_oracle_delta_vs_base\n"
        "0.1,0.01,0.2,0.11\n"
    )
    gpu = root / "results" / "optimus_gpu_suite"
    (gpu / "bench_a8_p64").mkdir(parents=True)
    for name in ["summary.json", "adapter_rows.jsonl", "per_prompt.jsonl"]:
        (gpu / "bench_a8_p64" / name).write_text("{}\n")
    for population in [1024, 4096]:
        run = gpu / f"search_p{population}_chunk8"
        run.mkdir()
        for name in ["summary.json", "candidate_summary.jsonl", "per_prompt.jsonl", "holdout_per_prompt.jsonl"]:
            (run / name).write_text("{}\n")
    for name in [
        "bench.csv",
        "adapter_throughput.png",
        "full_search.csv",
        "full_search_candidate_sec.png",
        "best_of_n.csv",
        "best_of_n.png",
        "quality_scaling.png",
        "token_throughput.png",
        "parity.csv",
        "halving.csv",
    ]:
        (systems / name).write_text("x\n")
    ledger = root / ".opencode"
    ledger.mkdir()
    (ledger / "prime-gpu-ledger.md").write_text(
        "## Active Pods\n\nNo active Prime pods. `prime pods list --plain` reported `Compute Pods (Total: 0)`.\n"
    )
    return gpu, systems


def test_release_check_passes_clean_optimus_tree(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        remote="https://github.com/plugyawn/optimus.git",
    )

    payload = summary(checks)
    assert payload["pass"] is True


def test_release_check_flags_legacy_package_and_old_remote(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path, include_legacy_package=True)

    checks = build_release_checks(
        root=tmp_path,
        systems_out=systems,
        gpu_root=gpu,
        populations=(1024, 4096),
        bench_adapters=(8,),
        run_halving=False,
        remote="https://github.com/plugyawn/randopt-lora-lab.git",
    )
    payload = summary(checks)
    failed = {check["name"] for check in payload["checks"] if not check["passed"]}

    assert payload["pass"] is False
    assert "published_package_excludes_legacy_namespace" in failed
    assert "github_remote_is_optimus" in failed


def test_release_check_cli_is_lightweight(tmp_path: Path):
    gpu, systems = write_minimal_release_tree(tmp_path)

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "release-check",
            "--root",
            str(tmp_path),
            "--gpu-root",
            str(gpu),
            "--systems-out",
            str(systems),
            "--populations",
            "1024,4096",
            "--bench-adapters",
            "8",
            "--skip-halving",
            "--remote-url",
            "https://github.com/plugyawn/optimus.git",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(result.stdout)
    assert payload["pass"] is True
    assert "PyTorch" not in result.stderr
    assert "NumPy" not in result.stderr
