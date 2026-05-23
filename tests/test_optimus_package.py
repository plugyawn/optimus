from __future__ import annotations

import types
import importlib.util
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

from optimus import __version__
from optimus.cli import resolve_command
from optimus.core.candidates import SearchCandidate, parse_candidate_key
from optimus.modeling.qwen import qwen_lora_shapes


def test_optimus_version_is_available():
    assert __version__ == "0.1.0"


def test_cli_resolves_vllm_search_entrypoint():
    assert resolve_command("vllm-search") == "optimus.commands.vllm_search"


def test_cli_exposes_professional_run_commands():
    assert resolve_command("run-plan") == "optimus.runs.gpu_suite"
    assert resolve_command("validate-run") == "optimus.evaluation.validation"
    assert resolve_command("release-check") == "optimus.evaluation.release"
    assert resolve_command("peft-search") == "optimus.commands.peft_search"
    assert resolve_command("run-suite") == "optimus.runs.gpu_suite_runner"


def test_cli_does_not_resolve_source_only_legacy_commands():
    for command in ["upstream-baseline-audit", "multirun-gate", "prompt-robustness", "score-sanity-audit"]:
        try:
            resolve_command(command)
        except ValueError as exc:
            assert "unknown Optimus command" in str(exc)
        else:
            raise AssertionError(command)


def test_cli_command_modules_stay_under_optimus_namespace():
    for command, module in resolve_command.__globals__["COMMANDS"].items():
        assert module.startswith("optimus."), (command, module)


def test_supported_command_wrappers_do_not_call_legacy_modules():
    supported = resolve_command.__globals__["SUPPORTED_COMMANDS"]
    for command, module_name in supported.items():
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None, (command, module_name)
        source = Path(spec.origin).read_text()
        assert "randopt_lora_lab" not in source, (command, module_name)


def test_no_generic_experiment_command_module_is_packaged():
    assert not Path("optimus/commands/experiment.py").exists()


def test_pyproject_declares_serving_and_dev_extras():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    assert "vllm" in extras["serving"]
    assert "pytest" in extras["dev"]
    assert pyproject["build-system"]["build-backend"] == "setuptools.build_meta"
    assert pyproject["tool"]["setuptools"]["packages"]["find"]["include"] == ["optimus*"]


def test_vllm_help_is_lightweight_and_optimus_owned():
    result = subprocess.run(
        [sys.executable, "-m", "optimus.cli", "vllm-search", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: optimus vllm-search" in result.stdout
    assert "PyTorch" not in result.stderr
    assert "NumPy" not in result.stderr


def test_backend_parity_help_is_lightweight_and_optimus_owned():
    result = subprocess.run(
        [sys.executable, "-m", "optimus.cli", "backend-parity-gate", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: optimus backend-parity-gate" in result.stdout
    assert "PyTorch" not in result.stderr
    assert "NumPy" not in result.stderr


def test_peft_search_help_is_lightweight_and_optimus_owned():
    result = subprocess.run(
        [sys.executable, "-m", "optimus.cli", "peft-search", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "usage: optimus peft-search" in result.stdout
    assert "PyTorch" not in result.stderr
    assert "NumPy" not in result.stderr
    assert "randopt_lora_lab" not in result.stdout


def test_peft_search_driver_implementation_is_optimus_owned():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from optimus.search.peft import run_search; from randopt_lora_lab.experiments import run_search as legacy; print(run_search.__module__, run_search is legacy)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.search.peft True" in result.stdout


def test_source_only_legacy_command_is_not_available_through_cli():
    result = subprocess.run(
        [sys.executable, "-m", "optimus.cli", "prompt-robustness", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unknown Optimus command" in result.stderr


def test_public_cli_does_not_expose_legacy_experiment_catchall():
    result = subprocess.run(
        [sys.executable, "-m", "optimus.cli", "--help"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "peft-search" in result.stdout
    assert "experiment" not in result.stdout
    assert "upstream-baseline-audit" not in result.stdout
    assert "goal-audit" not in result.stdout
    assert "score-sanity-audit" not in result.stdout


def test_search_candidate_surface_uses_existing_candidate_key_contract():
    candidate = SearchCandidate("isotropic", seed=123, sigma=0.0075, sign=-1)
    assert parse_candidate_key(candidate.key) == candidate


def test_candidate_core_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "import optimus.core.candidates"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stderr == ""


def test_countdown_task_import_is_lightweight():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from optimus.tasks import CountdownExample, generate_examples, prompt_fn, score_completion; print(CountdownExample.__name__, generate_examples.__module__, prompt_fn('default').__name__)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "CountdownExample" in result.stdout
    assert "optimus.tasks.generation" in result.stdout
    assert result.stderr == ""


def test_lora_modeling_metadata_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.modeling import AdapterSpec, parse_targets; print(parse_targets('q_proj'))"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "['q_proj']" in result.stdout
    assert result.stderr == ""


def test_lora_noise_implementation_is_optimus_owned():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from optimus.modeling.noise import lora_noise_tensors; from randopt_lora_lab.lora_space import lora_noise_tensors as legacy; print(lora_noise_tensors.__module__, lora_noise_tensors is legacy)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.modeling.noise True" in result.stdout


def test_transformers_backend_implementation_is_optimus_owned():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from optimus.serving.transformers import visible_token_count; from randopt_lora_lab.backends import visible_token_count as legacy; print(visible_token_count.__module__, visible_token_count is legacy)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.transformers True" in result.stdout


def test_serving_namespace_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "import optimus.serving; print(optimus.serving.__all__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "run_vllm_search" in result.stdout
    assert "backend_contract" in result.stdout
    assert "score_rows" in result.stdout
    assert result.stderr == ""


def test_vllm_serving_metadata_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving.vllm import AdapterSpec, candidate_panel; print(AdapterSpec.__name__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AdapterSpec" in result.stdout
    assert result.stderr == ""


def test_serving_runtime_helpers_import_from_public_namespace():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving import backend_contract, make_vllm_prompt_inputs, score_rows; print(backend_contract.__module__, make_vllm_prompt_inputs.__module__, score_rows.__module__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.contracts" in result.stdout
    assert "optimus.serving.prompting" in result.stdout
    assert "optimus.serving.runtime" in result.stdout
    assert result.stderr == ""


def test_vllm_search_driver_import_is_metadata_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving.search import run_search; from randopt_lora_lab.vllm_lora_search import run_search as legacy; print(run_search.__module__, run_search is legacy)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.search True" in result.stdout
    assert "NumPy" not in result.stderr
    assert "PyTorch" not in result.stderr


def test_vllm_benchmark_driver_import_is_metadata_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving.benchmark import run_benchmark; from randopt_lora_lab.vllm_lora_bench import run_benchmark as legacy; print(run_benchmark.__module__, run_benchmark is legacy)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.benchmark True" in result.stdout
    assert "NumPy" not in result.stderr
    assert "PyTorch" not in result.stderr


def test_vllm_halving_driver_import_is_metadata_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving.halving import run_halving; from randopt_lora_lab.vllm_lora_halving import run_halving as legacy; print(run_halving.__module__, run_halving is legacy)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.halving True" in result.stdout
    assert "NumPy" not in result.stderr
    assert "PyTorch" not in result.stderr


def test_runs_namespace_exports_execution_helpers_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.runs import execute_specs, plan_payload; print(execute_specs.__name__, plan_payload.__name__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "execute_specs plan_payload" in result.stdout
    assert result.stderr == ""


def test_evaluation_namespace_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "import optimus.evaluation; print(optimus.evaluation.__all__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "build_systems_report" in result.stdout
    assert "backend_parity_main" in result.stdout
    assert result.stderr == ""


def test_systems_report_implementation_is_optimus_owned():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from optimus.evaluation.systems import systems_summaries; from randopt_lora_lab.systems_report import systems_summaries as legacy; print(systems_summaries.__module__, systems_summaries is legacy)",
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.evaluation.systems True" in result.stdout
    assert result.stderr == ""


def test_qwen3_vl_text_shapes_use_language_model_prefix():
    config = types.SimpleNamespace(
        model_type="qwen3_vl_text",
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=4,
    )
    assert qwen_lora_shapes(config, ["k_proj"]) == [
        ("model.language_model.layers.0.self_attn.k_proj", 16, 8)
    ]
