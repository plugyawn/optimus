from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.lighteval import build_lighteval_command, build_sweep, model_args_from_options


def test_lighteval_model_args_include_tensor_parallel_size():
    assert (
        model_args_from_options("Qwen/Qwen3-4B", 4, data_parallel_size=2, max_model_length=2048)
        == "model_name=Qwen/Qwen3-4B,dtype=bfloat16,tensor_parallel_size=4,data_parallel_size=2,"
        "max_model_length=2048,trust_remote_code=True"
    )


def test_lighteval_command_uses_output_details_and_custom_tasks(tmp_path: Path):
    command = build_lighteval_command(
        backend="vllm",
        tasks="ifeval",
        model_args="model_name=Qwen/Qwen3-4B,tensor_parallel_size=4",
        output_dir=tmp_path / "eval",
        custom_tasks=tmp_path / "tasks.py",
        max_samples=8,
    )

    assert command[:4] == ("lighteval", "vllm", "model_name=Qwen/Qwen3-4B,tensor_parallel_size=4", "ifeval")
    assert "--output-dir" in command
    assert "--save-details" in command
    assert "--custom-tasks" in command
    assert "--max-samples" in command


def test_lighteval_cli_writes_plan_without_importing_lighteval(tmp_path: Path):
    plan = tmp_path / "plan.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "lighteval",
            "--tasks",
            "ifeval",
            "--model",
            "Qwen/Qwen3-4B",
            "--tensor-parallel-size",
            "4",
            "--out",
            str(tmp_path / "out"),
            "--plan-out",
            str(plan),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    payload = json.loads(plan.read_text())
    assert payload["backend"] == "vllm"
    assert payload["tasks"] == "ifeval"
    assert payload["command"][:2] == ["lighteval", "vllm"]


def test_lighteval_vllm_plan_omits_unsupported_chat_template_arg(tmp_path: Path):
    plan = tmp_path / "plan.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "lighteval",
            "--tasks",
            "ifeval",
            "--model",
            "Qwen/Qwen3-4B",
            "--use-chat-template",
            "--out",
            str(tmp_path / "out"),
            "--plan-out",
            str(plan),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    payload = json.loads(plan.read_text())
    assert "use_chat_template" not in payload["model_args"]
    assert all("use_chat_template" not in item for item in payload["command"])


def test_lighteval_sweep_builds_population_model_paths(tmp_path: Path):
    args = type(
        "Args",
        (),
        {
            "backend": "vllm",
            "tasks": "ifeval",
            "model": "Qwen/Qwen3-4B",
            "model_template": str(tmp_path / "models" / "p{population}"),
            "model_args": "",
            "model_key": "model_name",
            "model_arg": ["enable_prefix_caching=True"],
            "tensor_parallel_size": 1,
            "data_parallel_size": 8,
            "pipeline_parallel_size": None,
            "dtype": "bfloat16",
            "gpu_memory_utilization": 0.9,
            "max_model_length": 4096,
            "trust_remote_code": True,
            "use_chat_template": None,
            "custom_tasks": None,
            "max_samples": 32,
            "no_save_details": False,
            "populations": "128,256",
            "out_root": tmp_path / "eval",
            "out_template": "",
        },
    )()

    plan = build_sweep(args)

    assert plan.populations == (128, 256)
    assert plan.runs[0].model.endswith("/p128")
    assert "data_parallel_size=8" in plan.runs[0].model_args
    assert "enable_prefix_caching=True" in plan.runs[0].model_args
    assert plan.runs[1].command[-2:] == ("--max-samples", "32")


def test_lighteval_sweep_cli_writes_plan(tmp_path: Path):
    plan = tmp_path / "sweep.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "optimus.cli",
            "lighteval-sweep",
            "--tasks",
            "ifeval",
            "--model-template",
            str(tmp_path / "models" / "p{population}"),
            "--populations",
            "128,256",
            "--data-parallel-size",
            "8",
            "--out-root",
            str(tmp_path / "eval"),
            "--plan-out",
            str(plan),
        ],
        check=True,
        capture_output=True,
        text=True,
    )

    assert result.stdout == ""
    payload = json.loads(plan.read_text())
    assert payload["populations"] == [128, 256]
    assert payload["runs"][0]["model"].endswith("/p128")
