from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPTS = [
    Path("scripts/prime_sync_and_run.sh"),
    Path("scripts/remote/optimus_prime_bootstrap.sh"),
    Path("scripts/remote/optimus_prime_smoke.sh"),
    Path("scripts/remote/optimus_prime_gpu_suite.sh"),
    Path("scripts/run_backend_parity_gate.sh"),
    Path("scripts/run_optimus_gpu_suite.sh"),
]


def test_prime_scripts_parse_as_bash():
    for script in SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_prime_bootstrap_installs_declared_dev_extra():
    text = Path("scripts/remote/optimus_prime_bootstrap.sh").read_text()

    assert 'python -m pip install -e ".[dev]"' in text
    assert "python -m pip install pytest" not in text


def test_gpu_suite_launcher_delegates_execution_to_optimus_runner():
    text = Path("scripts/run_optimus_gpu_suite.sh").read_text()

    assert "optimus run-plan" in text
    assert "optimus run-suite" in text
    assert "--execution-log \"$OUT_ROOT/execution.json\"" in text
    assert "run_bench()" not in text
    assert "run_search()" not in text


def test_backend_parity_launcher_uses_supported_cli_commands():
    text = Path("scripts/run_backend_parity_gate.sh").read_text()

    assert "optimus peft-search" in text
    assert "optimus vllm-search" in text
    assert "optimus backend-parity-gate" in text
    assert "backend-output-diff" not in text
    assert "--allow-missing-output-diff" not in text
