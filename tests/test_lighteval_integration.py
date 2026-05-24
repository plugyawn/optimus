from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from optimus.evaluation.lighteval import build_lighteval_command, model_args_from_options


def test_lighteval_model_args_include_tensor_parallel_size():
    assert model_args_from_options("Qwen/Qwen2.5-3B-Instruct", 4) == "model_name=Qwen/Qwen2.5-3B-Instruct,tensor_parallel_size=4"


def test_lighteval_command_uses_output_details_and_custom_tasks(tmp_path: Path):
    command = build_lighteval_command(
        backend="vllm",
        tasks="ifeval",
        model_args="model_name=Qwen/Qwen2.5-3B-Instruct,tensor_parallel_size=4",
        output_dir=tmp_path / "eval",
        custom_tasks=tmp_path / "tasks.py",
        max_samples=8,
    )

    assert command[:4] == ("lighteval", "vllm", "model_name=Qwen/Qwen2.5-3B-Instruct,tensor_parallel_size=4", "ifeval")
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
            "Qwen/Qwen2.5-3B-Instruct",
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
