from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


LIGHTEVAL_BACKENDS = ("vllm", "accelerate", "transformers", "sglang", "custom")


@dataclass(frozen=True)
class LightEvalPlan:
    backend: str
    tasks: str
    model_args: str
    output_dir: str
    save_details: bool
    custom_tasks: str | None
    max_samples: int | None
    command: tuple[str, ...]


def lighteval_executable() -> str | None:
    return shutil.which("lighteval")


def model_args_from_options(model: str, tensor_parallel_size: int | None = None) -> str:
    fields = [f"model_name={model}"]
    if tensor_parallel_size is not None:
        fields.append(f"tensor_parallel_size={tensor_parallel_size}")
    return ",".join(fields)


def build_lighteval_command(
    *,
    backend: str,
    tasks: str,
    model_args: str,
    output_dir: Path,
    custom_tasks: Path | None = None,
    max_samples: int | None = None,
    save_details: bool = True,
) -> tuple[str, ...]:
    if backend not in LIGHTEVAL_BACKENDS:
        raise ValueError(f"unsupported LightEval backend {backend!r}; choose one of {LIGHTEVAL_BACKENDS}")
    command = ["lighteval", backend, model_args, tasks, "--output-dir", str(output_dir)]
    if save_details:
        command.append("--save-details")
    if custom_tasks is not None:
        command.extend(["--custom-tasks", str(custom_tasks)])
    if max_samples is not None:
        command.extend(["--max-samples", str(max_samples)])
    return tuple(command)


def build_plan(args: argparse.Namespace) -> LightEvalPlan:
    model_args = args.model_args or model_args_from_options(args.model, args.tensor_parallel_size)
    command = build_lighteval_command(
        backend=args.backend,
        tasks=args.tasks,
        model_args=model_args,
        output_dir=args.out,
        custom_tasks=args.custom_tasks,
        max_samples=args.max_samples,
        save_details=not args.no_save_details,
    )
    return LightEvalPlan(
        backend=args.backend,
        tasks=args.tasks,
        model_args=model_args,
        output_dir=str(args.out),
        save_details=not args.no_save_details,
        custom_tasks=str(args.custom_tasks) if args.custom_tasks else None,
        max_samples=args.max_samples,
        command=command,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run a LightEval confirmation/evaluation job.")
    parser.add_argument("--backend", choices=LIGHTEVAL_BACKENDS, default="vllm")
    parser.add_argument("--tasks", required=True, help="LightEval task string, for example 'ifeval' or 'mytask|0'.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--model-args", default="", help="Raw LightEval model-args string. Overrides --model.")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--out", type=Path, default=Path("results/lighteval"))
    parser.add_argument("--custom-tasks", type=Path, help="Path to a LightEval custom task file.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--no-save-details", action="store_true")
    parser.add_argument("--run", action="store_true", help="Execute LightEval instead of only writing the plan.")
    parser.add_argument("--plan-out", type=Path, help="Optional JSON file for the normalized LightEval command.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    plan = build_plan(args)
    payload = asdict(plan)
    payload["command"] = list(plan.command)
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.plan_out:
        args.plan_out.parent.mkdir(parents=True, exist_ok=True)
        args.plan_out.write_text(text)
    else:
        print(text, end="")
    if not args.run:
        return 0
    if lighteval_executable() is None:
        raise RuntimeError("LightEval executable not found. Install the eval extra with `python -m pip install -e \".[eval]\"`.")
    return subprocess.run(plan.command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
