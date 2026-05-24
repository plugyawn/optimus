from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path

from optimus.defaults import DEFAULT_LIGHTEVAL_POPULATIONS, DEFAULT_MODEL


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


@dataclass(frozen=True)
class LightEvalSweepEntry:
    population: int
    model: str
    output_dir: str
    model_args: str
    command: tuple[str, ...]


@dataclass(frozen=True)
class LightEvalSweepPlan:
    backend: str
    tasks: str
    populations: tuple[int, ...]
    save_details: bool
    custom_tasks: str | None
    max_samples: int | None
    runs: tuple[LightEvalSweepEntry, ...]


def lighteval_executable() -> str | None:
    return shutil.which("lighteval")


def parse_int_tuple(text: str) -> tuple[int, ...]:
    items = [item for chunk in text.split(",") for item in chunk.split()]
    return tuple(int(item) for item in items if item.strip())


def format_model_arg_value(value: str | int | float | bool) -> str:
    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def validate_model_arg(text: str) -> str:
    if "=" not in text:
        raise ValueError(f"LightEval model arg {text!r} must use KEY=VALUE syntax.")
    key, value = text.split("=", 1)
    if not key.strip() or value == "":
        raise ValueError(f"LightEval model arg {text!r} must use non-empty KEY=VALUE syntax.")
    return f"{key.strip()}={value}"


def format_population_template(text: str, values: dict[str, int]) -> str:
    if "{population}" in text or "{pop}" in text:
        return text.format(**values)
    return text


def model_args_from_options(
    model: str,
    tensor_parallel_size: int | None = None,
    *,
    data_parallel_size: int | None = None,
    pipeline_parallel_size: int | None = None,
    dtype: str | None = "bfloat16",
    gpu_memory_utilization: float | None = None,
    max_model_length: int | None = None,
    trust_remote_code: bool | None = True,
    use_chat_template: bool | None = None,
    model_key: str = "model_name",
    extra_model_args: tuple[str, ...] = (),
) -> str:
    fields: list[tuple[str, str | int | float | bool]] = [(model_key, model)]
    if dtype:
        fields.append(("dtype", dtype))
    if tensor_parallel_size is not None:
        fields.append(("tensor_parallel_size", tensor_parallel_size))
    if data_parallel_size is not None:
        fields.append(("data_parallel_size", data_parallel_size))
    if pipeline_parallel_size is not None:
        fields.append(("pipeline_parallel_size", pipeline_parallel_size))
    if gpu_memory_utilization is not None:
        fields.append(("gpu_memory_utilization", gpu_memory_utilization))
    if max_model_length is not None:
        fields.append(("max_model_length", max_model_length))
    if trust_remote_code is not None:
        fields.append(("trust_remote_code", trust_remote_code))
    if use_chat_template is not None:
        fields.append(("use_chat_template", use_chat_template))
    rendered = [f"{key}={format_model_arg_value(value)}" for key, value in fields]
    rendered.extend(validate_model_arg(item) for item in extra_model_args)
    return ",".join(rendered)


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
    model_args = args.model_args or model_args_from_options(
        args.model,
        args.tensor_parallel_size,
        data_parallel_size=args.data_parallel_size,
        pipeline_parallel_size=args.pipeline_parallel_size,
        dtype=args.dtype,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_length=args.max_model_length,
        trust_remote_code=args.trust_remote_code,
        use_chat_template=args.use_chat_template,
        model_key=args.model_key,
        extra_model_args=tuple(args.model_arg),
    )
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


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", choices=LIGHTEVAL_BACKENDS, default="vllm")
    parser.add_argument("--tasks", required=True, help="LightEval task string, for example 'ifeval' or 'mytask|0'.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-args", default="", help="Raw LightEval model-args string. Overrides --model.")
    parser.add_argument("--model-key", default="model_name", help="Model-id key for generated LightEval args.")
    parser.add_argument("--model-arg", action="append", default=[], help="Additional LightEval model arg as KEY=VALUE.")
    parser.add_argument("--tensor-parallel-size", type=int)
    parser.add_argument("--data-parallel-size", type=int)
    parser.add_argument("--pipeline-parallel-size", type=int)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    parser.add_argument("--max-model-length", type=int)
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--use-chat-template", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--custom-tasks", type=Path, help="Path to a LightEval custom task file.")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--no-save-details", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run a LightEval confirmation/evaluation job.")
    add_common_args(parser)
    parser.add_argument("--out", type=Path, default=Path("results/lighteval"))
    parser.add_argument("--run", action="store_true", help="Execute LightEval instead of only writing the plan.")
    parser.add_argument("--plan-out", type=Path, help="Optional JSON file for the normalized LightEval command.")
    return parser


def build_sweep(args: argparse.Namespace) -> LightEvalSweepPlan:
    populations = parse_int_tuple(args.populations)
    runs: list[LightEvalSweepEntry] = []
    model_template = args.model_template or args.model
    for population in populations:
        template_values = {"population": population, "pop": population}
        model = format_population_template(model_template, template_values)
        output_dir = (
            Path(format_population_template(str(args.out_template), template_values))
            if args.out_template
            else args.out_root / f"p{population}"
        )
        model_args = (
            format_population_template(args.model_args, template_values)
            if args.model_args
            else model_args_from_options(
                model,
                args.tensor_parallel_size,
                data_parallel_size=args.data_parallel_size,
                pipeline_parallel_size=args.pipeline_parallel_size,
                dtype=args.dtype,
                gpu_memory_utilization=args.gpu_memory_utilization,
                max_model_length=args.max_model_length,
                trust_remote_code=args.trust_remote_code,
                use_chat_template=args.use_chat_template,
                model_key=args.model_key,
                extra_model_args=tuple(format_population_template(arg, template_values) for arg in args.model_arg),
            )
        )
        command = build_lighteval_command(
            backend=args.backend,
            tasks=args.tasks,
            model_args=model_args,
            output_dir=output_dir,
            custom_tasks=args.custom_tasks,
            max_samples=args.max_samples,
            save_details=not args.no_save_details,
        )
        runs.append(
            LightEvalSweepEntry(
                population=population,
                model=model,
                output_dir=str(output_dir),
                model_args=model_args,
                command=command,
            )
        )
    return LightEvalSweepPlan(
        backend=args.backend,
        tasks=args.tasks,
        populations=populations,
        save_details=not args.no_save_details,
        custom_tasks=str(args.custom_tasks) if args.custom_tasks else None,
        max_samples=args.max_samples,
        runs=tuple(runs),
    )


def build_sweep_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan or run LightEval over population-labelled model artifacts.")
    add_common_args(parser)
    parser.add_argument(
        "--populations",
        default=",".join(str(item) for item in DEFAULT_LIGHTEVAL_POPULATIONS),
        help="Comma or space separated population labels.",
    )
    parser.add_argument(
        "--model-template",
        default="",
        help="Model/path template with optional {population} or {pop}; defaults to --model.",
    )
    parser.add_argument("--out-root", type=Path, default=Path("results/lighteval/population_sweep"))
    parser.add_argument("--out-template", default="", help="Output path template with optional {population} or {pop}.")
    parser.add_argument("--run", action="store_true", help="Execute each LightEval command instead of only writing the plan.")
    parser.add_argument("--continue-on-error", action="store_true", help="Run remaining jobs after a LightEval failure.")
    parser.add_argument("--plan-out", type=Path, help="Optional JSON file for the normalized sweep commands.")
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


def sweep_main(argv: list[str] | None = None) -> int:
    args = build_sweep_parser().parse_args(argv)
    plan = build_sweep(args)
    payload = asdict(plan)
    payload["runs"] = [
        {
            **{key: value for key, value in asdict(run).items() if key != "command"},
            "command": list(run.command),
        }
        for run in plan.runs
    ]
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
    final_returncode = 0
    for run in plan.runs:
        returncode = subprocess.run(run.command, check=False).returncode
        if returncode != 0:
            final_returncode = returncode
            if not args.continue_on_error:
                return returncode
    return final_returncode


if __name__ == "__main__":
    raise SystemExit(main())
