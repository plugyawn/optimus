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
from optimus.core.perturbations import PerturbationSpec, parse_perturbation_key
from optimus.evaluation.release import FORBIDDEN_PACKAGE
from optimus.modeling.qwen import qwen_lora_shapes


def test_optimus_version_is_available():
    assert __version__ == "0.1.0"


def test_cli_resolves_vllm_search_entrypoint():
    assert resolve_command("vllm-search") == "optimus.commands.vllm_search"


def test_cli_exposes_professional_run_commands():
    assert resolve_command("run-plan") == "optimus.runs.gpu_suite"
    assert resolve_command("validate-run") == "optimus.evaluation.validation"
    assert resolve_command("release-check") == "optimus.evaluation.release"
    assert resolve_command("lighteval") == "optimus.commands.lighteval"
    assert resolve_command("peft-search") == "optimus.commands.peft_search"
    assert resolve_command("perturbation-panel") == "optimus.commands.perturbation_panel"
    assert resolve_command("run-suite") == "optimus.runs.gpu_suite_runner"


def test_cli_does_not_resolve_unsupported_commands():
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


def test_supported_command_wrappers_do_not_call_old_modules():
    supported = resolve_command.__globals__["SUPPORTED_COMMANDS"]
    for command, module_name in supported.items():
        spec = importlib.util.find_spec(module_name)
        assert spec is not None and spec.origin is not None, (command, module_name)
        source = Path(spec.origin).read_text()
        assert FORBIDDEN_PACKAGE not in source, (command, module_name)


def test_no_generic_experiment_command_module_is_packaged():
    assert not Path("optimus/commands/experiment.py").exists()


def test_pyproject_declares_serving_and_dev_extras():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text())
    extras = pyproject["project"]["optional-dependencies"]
    assert "vllm" in extras["serving"]
    assert any(dep.startswith("lighteval") for dep in extras["eval"])
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
    assert FORBIDDEN_PACKAGE not in result.stdout


def test_unsupported_command_is_not_available_through_cli():
    result = subprocess.run(
        [sys.executable, "-m", "optimus.cli", "prompt-robustness", "--help"],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "unknown Optimus command" in result.stderr


def test_public_cli_does_not_expose_experiment_catchall():
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


def test_perturbation_surface_uses_stable_key_contract():
    candidate = PerturbationSpec("isotropic", seed=123, sigma=0.0075, sign=-1, method="lora")
    assert parse_perturbation_key(candidate.key) == candidate
    assert parse_perturbation_key(candidate.legacy_key) == candidate


def test_core_perturbation_spec_supports_dense_and_lora_keys():
    lora = PerturbationSpec("isotropic", seed=123, sigma=0.0075, sign=1, method="lora", rank=8, targets="q_proj,v_proj")
    dense = PerturbationSpec("dense_gaussian", seed=456, sigma=0.01, sign=-1, method="dense")

    assert lora.key.startswith("lora:")
    assert dense.key.startswith("dense:")
    assert parse_perturbation_key(lora.key) == lora
    assert parse_perturbation_key(dense.legacy_key) == dense


def test_perturbation_core_import_is_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "import optimus.core.perturbations"],
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
        [sys.executable, "-c", "from optimus.serving.vllm import AdapterSpec, perturbation_panel; print(AdapterSpec.__name__, perturbation_panel.__name__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "AdapterSpec" in result.stdout
    assert "perturbation_panel" in result.stdout
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
        [sys.executable, "-c", "from optimus.serving.search import run_search; print(run_search.__module__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.search" in result.stdout
    assert "NumPy" not in result.stderr
    assert "PyTorch" not in result.stderr


def test_vllm_benchmark_driver_import_is_metadata_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving.benchmark import run_benchmark; print(run_benchmark.__module__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.benchmark" in result.stdout
    assert "NumPy" not in result.stderr
    assert "PyTorch" not in result.stderr


def test_vllm_halving_driver_import_is_metadata_lightweight():
    result = subprocess.run(
        [sys.executable, "-c", "from optimus.serving.halving import run_halving; print(run_halving.__module__)"],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "optimus.serving.halving" in result.stdout
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

    assert "lighteval_command" in result.stdout
    assert "backend_parity_main" in result.stdout
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


def test_repo_root_has_no_old_project_namespace_or_tracked_results():
    assert not any(Path(".").glob(f"{FORBIDDEN_PACKAGE}/*.py"))
    tracked = subprocess.run(["git", "ls-files"], check=True, capture_output=True, text=True).stdout.splitlines()
    assert not any(path.startswith(f"{FORBIDDEN_PACKAGE}/") for path in tracked)
    assert not any(path.startswith("results/") for path in tracked)
    assert not any(path.startswith("docs/reports/") for path in tracked)
    assert not any(path.startswith("data/") for path in tracked)
