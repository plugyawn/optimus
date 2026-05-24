from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping

from optimus.core.experiments import ExperimentKey, status_record


@dataclass(frozen=True)
class GpuSuiteConfig:
    output_root: Path = Path("results/optimus_gpu_suite")
    systems_output_root: Path = Path("results/report/optimus_systems")
    data: Path = Path("data/countdown_generated_1200_seed20260507.json")
    model: str = "Qwen/Qwen2.5-3B-Instruct"
    populations: tuple[int, ...] = (1024, 4096)
    prompts: int = 64
    holdout_prompts: int = 256
    promote: int = 64
    rank: int = 8
    sigma: float = 0.0075
    seed: int = 2468
    targets: str = "q_proj,v_proj"
    max_new_tokens: int = 32
    chunk_adapters: int = 8
    max_loras: int = 8
    max_cpu_loras: int = 8192
    tensor_parallel_size: int = 8
    bench_adapters: tuple[int, ...] = (8, 16, 32)
    halving_population: int = 1024
    halving_stage_prompts: int = 8
    halving_survivors: int = 64
    run_halving: bool = True


@dataclass(frozen=True)
class RunSpec:
    name: str
    kind: str
    output_path: Path
    command: tuple[str, ...]
    backend: str = "vllm"
    method: str = "lora"
    population: int | None = None
    identity: Mapping[str, Any] = field(default_factory=dict)

    @property
    def key(self) -> ExperimentKey:
        command_text = json.dumps(list(self.command), sort_keys=True)
        extra = dict(self.identity)
        extra["command_sha256"] = hashlib.sha256(command_text.encode("utf-8")).hexdigest()
        return ExperimentKey(
            name=self.name,
            kind=self.kind,
            method=self.method,
            backend=self.backend,
            population=self.population,
            model=str(extra.get("model")) if extra.get("model") is not None else None,
            seed=int(extra["seed"]) if extra.get("seed") is not None else None,
            extra=extra,
        )


def _base_search_args(config: GpuSuiteConfig, output_path: Path, population: int) -> list[str]:
    return [
        "optimus",
        "vllm-search",
        "--out",
        str(output_path),
        "--model",
        config.model,
        "--data",
        str(config.data),
        "--prompts",
        str(config.prompts),
        "--holdout-prompts",
        str(config.holdout_prompts),
        "--population",
        str(population),
        "--promote",
        str(config.promote),
        "--rank",
        str(config.rank),
        "--sigma",
        f"{config.sigma:g}",
        "--seed",
        str(config.seed),
        "--targets",
        config.targets,
        "--max-loras",
        str(config.max_loras),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--chunk-adapters",
        str(config.chunk_adapters),
        "--max-cpu-loras",
        str(config.max_cpu_loras),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--stop-at-answer",
        "--antithetic",
    ]


def search_identity(config: GpuSuiteConfig, population: int) -> dict[str, Any]:
    return {
        "model": config.model,
        "data": str(config.data),
        "family": "isotropic",
        "population": population,
        "screen_prompts": config.prompts,
        "holdout_prompts": config.holdout_prompts,
        "promote": config.promote,
        "rank": config.rank,
        "sigma": config.sigma,
        "seed": config.seed,
        "targets": config.targets,
        "max_new_tokens": config.max_new_tokens,
        "chunk_adapters": config.chunk_adapters,
        "max_loras": config.max_loras,
        "max_cpu_loras": config.max_cpu_loras,
        "tensor_parallel_size": config.tensor_parallel_size,
        "antithetic": True,
    }


def bench_identity(config: GpuSuiteConfig, adapters: int) -> dict[str, Any]:
    return {
        "model": config.model,
        "data": str(config.data),
        "family": "isotropic",
        "adapters": adapters,
        "prompts": config.prompts,
        "rank": config.rank,
        "sigma": config.sigma,
        "seed": config.seed,
        "targets": config.targets,
        "max_new_tokens": config.max_new_tokens,
        "max_loras": adapters,
        "max_cpu_loras": config.max_cpu_loras,
        "tensor_parallel_size": config.tensor_parallel_size,
    }


def halving_identity(config: GpuSuiteConfig) -> dict[str, Any]:
    identity = search_identity(config, config.halving_population)
    identity.update(
        {
            "stage_prompts": config.halving_stage_prompts,
            "survivors": config.halving_survivors,
        }
    )
    return identity


def bench_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    specs = []
    for adapters in config.bench_adapters:
        out = config.output_root / f"bench_a{adapters}_p{config.prompts}"
        specs.append(
            RunSpec(
                name=f"bench_a{adapters}_p{config.prompts}",
                kind="bench",
                output_path=out,
                population=adapters,
                identity=bench_identity(config, adapters),
                command=(
                    "optimus",
                    "vllm-bench",
                    "--out",
                    str(out),
                    "--model",
                    config.model,
                    "--data",
                    str(config.data),
                    "--adapters",
                    str(adapters),
                    "--prompts",
                    str(config.prompts),
                    "--rank",
                    str(config.rank),
                    "--sigma",
                    f"{config.sigma:g}",
                    "--targets",
                    config.targets,
                    "--max-loras",
                    str(adapters),
                    "--tensor-parallel-size",
                    str(config.tensor_parallel_size),
                    "--max-cpu-loras",
                    str(config.max_cpu_loras),
                    "--max-new-tokens",
                    str(config.max_new_tokens),
                    "--stop-at-answer",
                    "--preload",
                    "--mixed-batch",
                    "--skip-sequential",
                    "--no-include-base",
                ),
            )
        )
    return specs


def search_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    specs = []
    for population in config.populations:
        out = config.output_root / f"search_p{population}_chunk{config.chunk_adapters}"
        specs.append(
            RunSpec(
                name=f"search_p{population}_chunk{config.chunk_adapters}",
                kind="search",
                output_path=out,
                population=population,
                identity=search_identity(config, population),
                command=tuple(_base_search_args(config, out, population)),
            )
        )
    return specs


def halving_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    out = (
        config.output_root
        / f"halving_p{config.halving_population}_stage{config.halving_stage_prompts}_surv{config.halving_survivors}"
    )
    reference = config.output_root / f"search_p{config.halving_population}_chunk{config.chunk_adapters}"
    return [
        RunSpec(
            name=out.name,
            kind="halving",
            output_path=out,
            population=config.halving_population,
            identity=halving_identity(config),
            command=(
                "optimus",
                "vllm-halving",
                "--out",
                str(out),
                "--model",
                config.model,
                "--data",
                str(config.data),
                "--prompts",
                str(config.prompts),
                "--stage-prompts",
                str(config.halving_stage_prompts),
                "--holdout-prompts",
                str(config.holdout_prompts),
                "--population",
                str(config.halving_population),
                "--survivors",
                str(config.halving_survivors),
                "--promote",
                str(config.promote),
                "--rank",
                str(config.rank),
                "--sigma",
                f"{config.sigma:g}",
                "--seed",
                str(config.seed),
                "--targets",
                config.targets,
                "--max-loras",
                str(config.max_loras),
                "--tensor-parallel-size",
                str(config.tensor_parallel_size),
                "--chunk-adapters",
                str(config.chunk_adapters),
                "--max-cpu-loras",
                str(config.max_cpu_loras),
                "--max-new-tokens",
                str(config.max_new_tokens),
                "--stop-at-answer",
                "--antithetic",
                "--full-search-reference",
                str(reference),
            ),
        )
    ]


def report_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    report_root = config.output_root.parent
    return [
        RunSpec(
            name="systems_report",
            kind="report",
            output_path=config.systems_output_root,
            backend="local",
            method="report",
            identity={"root": str(report_root), "systems_out": str(config.systems_output_root)},
            command=("optimus", "systems-report", "--root", str(report_root), "--out", str(config.systems_output_root)),
        )
    ]


def gpu_suite_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    halving = halving_specs(config) if config.run_halving else []
    return [*bench_specs(config), *search_specs(config), *halving, *report_specs(config)]


def completion_marker(spec: RunSpec) -> Path:
    if spec.kind == "report":
        return spec.output_path / "report.md"
    return spec.output_path / "summary.json"


def spec_is_complete(spec: RunSpec) -> bool:
    if spec.kind == "report":
        return False
    marker = completion_marker(spec)
    if not marker.exists():
        return False
    try:
        payload = json.loads(marker.read_text())
    except (json.JSONDecodeError, OSError):
        return False
    if str(payload.get("kind", "")).endswith("_failure"):
        return False
    for key, expected in spec.identity.items():
        if key in {"command_sha256", "data"}:
            continue
        if key not in payload:
            return False
        observed = payload[key]
        if isinstance(observed, list):
            observed = ",".join(str(item) for item in observed)
        if str(observed) != str(expected):
            return False
    return True


def execute_specs(
    specs: list[RunSpec],
    *,
    dry_run: bool = False,
    skip_existing: bool = True,
    on_update: Callable[[list[dict[str, Any]]], None] | None = None,
) -> list[dict[str, Any]]:
    rows = []
    for spec in specs:
        marker = completion_marker(spec)
        if skip_existing and spec_is_complete(spec):
            rows.append(
                status_record(
                    key=spec.key,
                    output_path=spec.output_path,
                    command=spec.command,
                    status="skipped",
                    marker=marker,
                )
            )
            if on_update:
                on_update(rows)
            continue
        if dry_run:
            rows.append(
                status_record(
                    key=spec.key,
                    output_path=spec.output_path,
                    command=spec.command,
                    status="dry_run",
                    marker=marker,
                )
            )
            if on_update:
                on_update(rows)
            continue
        started_at = time.time()
        try:
            completed = subprocess.run(spec.command, check=True)
        except subprocess.CalledProcessError as exc:
            rows.append(
                status_record(
                    key=spec.key,
                    output_path=spec.output_path,
                    command=spec.command,
                    status="failed",
                    marker=marker,
                    started_at=started_at,
                    finished_at=time.time(),
                    returncode=exc.returncode,
                    error=str(exc),
                )
            )
            if on_update:
                on_update(rows)
            raise
        rows.append(
            status_record(
                key=spec.key,
                output_path=spec.output_path,
                command=spec.command,
                status="completed",
                marker=marker,
                started_at=started_at,
                finished_at=time.time(),
                returncode=completed.returncode,
            )
        )
        if on_update:
            on_update(rows)
    return rows


def plan_payload(config: GpuSuiteConfig) -> dict:
    return {
        "config": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in asdict(config).items()
        },
        "runs": [
            {
                "name": spec.name,
                "kind": spec.kind,
                "backend": spec.backend,
                "method": spec.method,
                "population": spec.population,
                "output_path": str(spec.output_path),
                "command": list(spec.command),
            }
            for spec in gpu_suite_specs(config)
        ],
    }


def parse_int_tuple(text: str) -> tuple[int, ...]:
    items = [item for chunk in text.split(",") for item in chunk.split()]
    return tuple(int(item) for item in items if item.strip())


def add_config_args(parser: argparse.ArgumentParser, *, include_out: bool = True) -> None:
    if include_out:
        parser.add_argument("--out", type=Path)
    parser.add_argument("--root", type=Path, default=Path("results/optimus_gpu_suite"))
    parser.add_argument("--systems-out", type=Path, default=Path("results/report/optimus_systems"))
    parser.add_argument("--data", type=Path, default=Path("data/countdown_generated_1200_seed20260507.json"))
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--populations", default="1024,4096")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--holdout-prompts", type=int, default=256)
    parser.add_argument("--promote", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.0075)
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--targets", default="q_proj,v_proj")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--chunk-adapters", type=int, default=8)
    parser.add_argument("--max-loras", type=int, default=8)
    parser.add_argument("--max-cpu-loras", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=8)
    parser.add_argument("--bench-adapters", default="8,16,32")
    parser.add_argument("--halving-population", type=int, default=1024)
    parser.add_argument("--halving-stage-prompts", type=int, default=8)
    parser.add_argument("--halving-survivors", type=int, default=64)
    parser.add_argument("--skip-halving", action="store_true")


def config_from_args(args: argparse.Namespace) -> GpuSuiteConfig:
    return GpuSuiteConfig(
        output_root=args.root,
        systems_output_root=args.systems_out,
        data=args.data,
        model=args.model,
        populations=parse_int_tuple(args.populations),
        prompts=args.prompts,
        holdout_prompts=args.holdout_prompts,
        promote=args.promote,
        rank=args.rank,
        sigma=args.sigma,
        seed=args.seed,
        targets=args.targets,
        max_new_tokens=args.max_new_tokens,
        chunk_adapters=args.chunk_adapters,
        max_loras=args.max_loras,
        max_cpu_loras=args.max_cpu_loras,
        tensor_parallel_size=args.tensor_parallel_size,
        bench_adapters=parse_int_tuple(args.bench_adapters),
        halving_population=args.halving_population,
        halving_stage_prompts=args.halving_stage_prompts,
        halving_survivors=args.halving_survivors,
        run_halving=not args.skip_halving,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Write an Optimus P1024/P4096 GPU run plan.")
    add_config_args(parser)
    parser.add_argument("--format", choices=["json", "markdown"], default="json")
    return parser


def markdown_plan(payload: dict) -> str:
    lines = ["# Optimus GPU Run Plan", "", "| kind | name | output |", "| --- | --- | --- |"]
    for run in payload["runs"]:
        lines.append(f"| {run['kind']} | `{run['name']}` | `{run['output_path']}` |")
    lines.extend(["", "## Commands", ""])
    for run in payload["runs"]:
        lines.append(f"### {run['name']}")
        lines.append("")
        lines.append("```bash")
        lines.append(" ".join(run["command"]))
        lines.append("```")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = config_from_args(args)
    payload = plan_payload(config)
    text = (
        json.dumps(payload, indent=2, sort_keys=True) + "\n"
        if args.format == "json"
        else markdown_plan(payload) + "\n"
    )
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
    else:
        print(text, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
