from __future__ import annotations

import subprocess
from pathlib import Path


SCRIPTS = [
    Path("scripts/prime_sync_and_run.sh"),
    Path("scripts/remote/optimus_prime_bootstrap.sh"),
    Path("scripts/remote/optimus_prime_smoke.sh"),
    Path("scripts/remote/optimus_prime_gpu_suite.sh"),
    Path("scripts/remote/optimus_prime_lighteval_sweep.sh"),
    Path("scripts/remote/optimus_prime_population_lighteval.sh"),
    Path("scripts/run_backend_parity_gate.sh"),
    Path("scripts/run_lighteval_population_sweep.sh"),
    Path("scripts/run_optimus_gpu_suite.sh"),
    Path("scripts/run_population_lighteval_pipeline.sh"),
]


def test_prime_scripts_parse_as_bash():
    for script in SCRIPTS:
        subprocess.run(["bash", "-n", str(script)], check=True)


def test_prime_bootstrap_installs_declared_dev_extra():
    text = Path("scripts/remote/optimus_prime_bootstrap.sh").read_text()

    assert 'python -m pip install -e ".[dev,eval]"' in text
    assert "python -m pip install pytest" not in text
    assert 'OPTIMUS_INSTALL_FLASHINFER:-0' in text
    assert "OPTIMUS_INSTALL_CUDA_COMPILER" in text
    assert "OPTIMUS_REQUIRE_CUDA_DEV_HEADERS" in text
    assert "cuda-libraries-dev-13-0" in text
    assert "cublasLt.h" in text
    assert "nvrtc.h" in text
    assert "OPTIMUS_PATCH_VLLM09_AIMV2" in text
    assert 'OPTIMUS_VLLM_PACKAGE:-vllm==0.19.0' in text
    assert "transformers>=4.54,<4.55" in text
    assert "runtime import check failed" in text


def test_gpu_suite_launcher_delegates_execution_to_optimus_runner():
    text = Path("scripts/run_optimus_gpu_suite.sh").read_text()

    assert "optimus run-plan" in text
    assert "optimus run-suite" in text
    assert "--execution-log \"$OUT_ROOT/execution.json\"" in text
    assert "--strict" in text
    assert "PROMPT_VARIANTS=${PROMPT_VARIANTS:-default}" in text
    assert "--prompt-variants \"$PROMPT_VARIANTS\"" in text
    assert "run_bench()" not in text
    assert "run_search()" not in text


def test_population_lighteval_pipeline_closes_eval_loop():
    text = Path("scripts/run_population_lighteval_pipeline.sh").read_text()

    assert "scripts/run_optimus_gpu_suite.sh" in text
    assert "optimus materialize-selected" in text
    assert "optimus lighteval " in text
    assert "optimus lighteval-sweep" in text
    assert "optimus lighteval-report" in text
    assert "KEEP_ADAPTERS=${KEEP_ADAPTERS:-1}" in text
    assert "MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}" in text
    assert "PROMPT_VARIANTS=${PROMPT_VARIANTS:-bare}" in text
    assert "LIGHTEVAL_USE_CHAT_TEMPLATE=${LIGHTEVAL_USE_CHAT_TEMPLATE:-auto}" in text


def test_prime_sync_bundle_does_not_copy_local_agent_state():
    text = Path("scripts/prime_sync_and_run.sh").read_text()

    assert "--exclude='.git'" in text
    assert "--exclude='.opencode'" in text
    assert "MODEL=${MODEL:-Qwen/Qwen3-4B-Instruct-2507}" in text
    assert "PROMPT_VARIANTS=${PROMPT_VARIANTS:-bare}" in text
    assert "PROMPT_VARIANTS='$PROMPT_VARIANTS'" in text
    assert "OPTIMUS_VLLM_PACKAGE" in text
    assert "OPTIMUS_REQUIRE_CUDA_DEV_HEADERS" in text


def test_backend_parity_launcher_uses_supported_cli_commands():
    text = Path("scripts/run_backend_parity_gate.sh").read_text()

    assert "optimus search" in text
    assert "--backend transformers" in text
    assert "--backend vllm" in text
    assert "--method lora" in text
    assert "optimus peft-search" not in text
    assert "optimus vllm-search" not in text
    assert "optimus backend-parity-gate" in text
    assert "backend-output-diff" not in text
    assert "--allow-missing-output-diff" not in text
