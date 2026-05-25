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
from optimus.defaults import DEFAULT_MODEL, DEFAULT_SEARCH_POPULATIONS, DEFAULT_TARGETS


@dataclass(frozen=True)
class GpuSuiteConfig:
    output_root: Path = Path("results/optimus_gpu_suite")
    systems_output_root: Path = Path("results/report/optimus_systems")
    data: Path = Path("data/countdown_generated_1200_seed20260507.json")
    model: str = DEFAULT_MODEL
    backend: str = "vllm"
    method: str = "lora"
    populations: tuple[int, ...] = DEFAULT_SEARCH_POPULATIONS
    prompts: int = 64
    holdout_prompts: int = 256
    promote: int = 64
    rank: int = 8
    sigma: float = 0.0075
    basis_rank: int = 128
    basis_prompts: int = 32
    target_preset: str = "transformer-linears"
    scale_mode: str = "relative-output-rms"
    rho_grid: str = "0.002,0.005,0.01,0.02"
    sigma_w_grid: str = ""
    budget_policy: str = "per-block-equal"
    basis_kind: str = "activation-svd"
    top_k_grid: str = "1,4,8,16"
    seed: int = 2468
    targets: str = DEFAULT_TARGETS
    max_new_tokens: int = 32
    prompt_variants: str = "default"
    prompt_input: str = "text"
    use_chat_template: bool = False
    max_base_malformed_for_selection: float = 0.05
    max_base_cap_hit_for_selection: float = 0.05
    min_selection_prompt_variants: int = 1
    require_all_prompt_variants_valid: bool = False
    chunk_adapters: int = 32
    max_loras: int = 32
    max_cpu_loras: int = 8192
    tensor_parallel_size: int = 1
    enable_prefix_caching: bool | None = None
    enable_chunked_prefill: bool | None = None
    kv_cache_dtype: str = ""
    vllm_kwargs: tuple[str, ...] = ()
    keep_adapters: bool = False
    bench_adapters: tuple[int, ...] = (8, 16, 32)
    halving_population: int = 1024
    halving_stage_prompts: int = 8
    halving_survivors: int = 64
    run_halving: bool = False


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
    args = [
        "optimus",
        "search",
        "--backend",
        config.backend,
        "--method",
        config.method,
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
        "--seed",
        str(config.seed),
        "--tensor-parallel-size",
        str(config.tensor_parallel_size),
        "--max-new-tokens",
        str(config.max_new_tokens),
        "--prompt-variants",
        config.prompt_variants,
        "--prompt-input",
        config.prompt_input,
        "--max-base-malformed-for-selection",
        f"{config.max_base_malformed_for_selection:g}",
        "--max-base-cap-hit-for-selection",
        f"{config.max_base_cap_hit_for_selection:g}",
        "--min-selection-prompt-variants",
        str(config.min_selection_prompt_variants),
        "--stop-at-answer",
        "--antithetic",
    ] + _search_prompt_args(config) + _vllm_runtime_args(config)
    if config.method == "subspace":
        return args + _subspace_args(config)
    return args + _lora_search_args(config) + _search_artifact_args(config)


def _lora_search_args(config: GpuSuiteConfig) -> list[str]:
    return [
        "--rank",
        str(config.rank),
        "--sigma",
        f"{config.sigma:g}",
        "--targets",
        config.targets,
        "--max-loras",
        str(config.max_loras),
        "--chunk-adapters",
        str(config.chunk_adapters),
        "--max-cpu-loras",
        str(config.max_cpu_loras),
    ]


def _subspace_args(config: GpuSuiteConfig) -> list[str]:
    args = [
        "--basis-rank",
        str(config.basis_rank),
        "--basis-prompts",
        str(config.basis_prompts),
        "--target-preset",
        config.target_preset,
        "--scale-mode",
        config.scale_mode,
        "--budget-policy",
        config.budget_policy,
        "--basis-kind",
        config.basis_kind,
        "--top-k-grid",
        config.top_k_grid,
    ]
    if config.scale_mode == "projected-dense":
        args.extend(["--sigma-w-grid", config.sigma_w_grid or "1e-5,3e-5,1e-4"])
    else:
        args.extend(["--rho-grid", config.rho_grid])
    return args


def _search_prompt_args(config: GpuSuiteConfig) -> list[str]:
    args = []
    if config.use_chat_template:
        args.append("--use-chat-template")
    if config.require_all_prompt_variants_valid:
        args.append("--require-all-prompt-variants-valid")
    return args


def _vllm_runtime_args(config: GpuSuiteConfig) -> list[str]:
    args = []
    if config.enable_prefix_caching is not None:
        args.append("--enable-prefix-caching" if config.enable_prefix_caching else "--no-enable-prefix-caching")
    if config.enable_chunked_prefill is not None:
        args.append("--enable-chunked-prefill" if config.enable_chunked_prefill else "--no-enable-chunked-prefill")
    if config.kv_cache_dtype:
        args.extend(["--kv-cache-dtype", config.kv_cache_dtype])
    for item in config.vllm_kwargs:
        args.extend(["--vllm-kwarg", item])
    return args


def _search_artifact_args(config: GpuSuiteConfig) -> list[str]:
    return ["--keep-adapters"] if config.keep_adapters else []


def search_identity(config: GpuSuiteConfig, population: int) -> dict[str, Any]:
    identity = {
        "model": config.model,
        "data": str(config.data),
        "backend": config.backend,
        "method": config.method,
        "population": population,
        "screen_prompts": config.prompts,
        "holdout_prompts": config.holdout_prompts,
        "promote": config.promote,
        "seed": config.seed,
        "max_new_tokens": config.max_new_tokens,
        "prompt_variants": config.prompt_variants,
        "prompt_input": config.prompt_input,
        "use_chat_template": config.use_chat_template,
        "max_base_malformed_for_selection": config.max_base_malformed_for_selection,
        "max_base_cap_hit_for_selection": config.max_base_cap_hit_for_selection,
        "min_selection_prompt_variants": config.min_selection_prompt_variants,
        "require_all_prompt_variants_valid": config.require_all_prompt_variants_valid,
        "tensor_parallel_size": config.tensor_parallel_size,
        "antithetic": True,
    }
    if config.method == "subspace":
        identity.update(
            {
                "basis_rank": config.basis_rank,
                "basis_prompts": config.basis_prompts,
                "target_preset": config.target_preset,
                "scale_mode": config.scale_mode,
                "rho_grid": config.rho_grid,
                "sigma_w_grid": config.sigma_w_grid,
                "budget_policy": config.budget_policy,
                "basis_kind": config.basis_kind,
                "top_k_grid": config.top_k_grid,
            }
        )
    else:
        identity.update(
            {
                "family": "isotropic",
                "rank": config.rank,
                "sigma": config.sigma,
                "targets": config.targets,
                "chunk_adapters": config.chunk_adapters,
                "max_loras": config.max_loras,
                "max_cpu_loras": config.max_cpu_loras,
            }
        )
    return identity


def bench_identity(config: GpuSuiteConfig, adapters: int) -> dict[str, Any]:
    return {
        "model": config.model,
        "data": str(config.data),
        "backend": config.backend,
        "method": config.method,
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


def bench_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    if config.method == "subspace":
        return []
    specs = []
    for adapters in config.bench_adapters:
        out = config.output_root / f"bench_a{adapters}_p{config.prompts}"
        specs.append(
            RunSpec(
                name=f"bench_a{adapters}_p{config.prompts}",
                kind="bench",
                output_path=out,
                population=adapters,
                backend=config.backend,
                method=config.method,
                identity=bench_identity(config, adapters),
                command=(
                    "optimus",
                    "bench",
                    "--backend",
                    config.backend,
                    "--method",
                    config.method,
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
                    "--prompt-input",
                    config.prompt_input,
                    "--stop-at-answer",
                    "--preload",
                    "--mixed-batch",
                    "--skip-sequential",
                    "--no-include-base",
                    *_vllm_runtime_args(config),
                ),
            )
        )
    return specs


def search_specs(config: GpuSuiteConfig) -> list[RunSpec]:
    specs = []
    for population in config.populations:
        suffix = f"subspace_r{config.basis_rank}" if config.method == "subspace" else f"chunk{config.chunk_adapters}"
        out = config.output_root / f"search_p{population}_{suffix}"
        specs.append(
            RunSpec(
                name=out.name,
                kind="search",
                output_path=out,
                backend=config.backend,
                method=config.method,
                population=population,
                identity=search_identity(config, population),
                command=tuple(_base_search_args(config, out, population)),
            )
        )
    return specs


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
    if config.run_halving:
        raise RuntimeError("staged search is disabled until it has a final public route")
    return [*bench_specs(config), *search_specs(config), *report_specs(config)]


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
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--backend", choices=["vllm"], default="vllm")
    parser.add_argument("--method", choices=["lora", "subspace"], default="lora")
    parser.add_argument("--populations", default="1024,4096")
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--holdout-prompts", type=int, default=256)
    parser.add_argument("--promote", type=int, default=64)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.0075)
    parser.add_argument("--basis-rank", type=int, default=128)
    parser.add_argument("--basis-prompts", type=int, default=32)
    parser.add_argument("--target-preset", default="transformer-linears")
    parser.add_argument("--scale-mode", choices=["relative-output-rms", "projected-dense"], default="relative-output-rms")
    parser.add_argument("--rho-grid", default="0.002,0.005,0.01,0.02")
    parser.add_argument("--sigma-w-grid", default="")
    parser.add_argument("--budget-policy", default="per-block-equal")
    parser.add_argument("--basis-kind", default="activation-svd")
    parser.add_argument("--top-k-grid", default="1,4,8,16")
    parser.add_argument("--seed", type=int, default=2468)
    parser.add_argument("--targets", default=DEFAULT_TARGETS)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--prompt-variants", default="default")
    parser.add_argument("--prompt-input", default="text", choices=["text", "token_ids"])
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument("--max-base-malformed-for-selection", type=float, default=0.05)
    parser.add_argument("--max-base-cap-hit-for-selection", type=float, default=0.05)
    parser.add_argument("--min-selection-prompt-variants", type=int, default=1)
    parser.add_argument("--require-all-prompt-variants-valid", action="store_true")
    parser.add_argument("--chunk-adapters", type=int, default=32)
    parser.add_argument("--max-loras", type=int, default=32)
    parser.add_argument("--max-cpu-loras", type=int, default=8192)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--enable-chunked-prefill", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--kv-cache-dtype", default="")
    parser.add_argument("--vllm-kwarg", action="append", default=[], help="Extra vLLM LLM() kwarg as KEY=VALUE.")
    parser.add_argument("--keep-adapters", action="store_true")
    parser.add_argument("--bench-adapters", default="8,16,32")
    parser.add_argument("--halving-population", type=int, default=1024)
    parser.add_argument("--halving-stage-prompts", type=int, default=8)
    parser.add_argument("--halving-survivors", type=int, default=64)
    parser.add_argument("--run-halving", action="store_true", help="Reserved until a final staged-search route exists.")
    parser.add_argument("--skip-halving", action="store_true", help="Compatibility no-op; staged search is disabled by default.")


def config_from_args(args: argparse.Namespace) -> GpuSuiteConfig:
    return GpuSuiteConfig(
        output_root=args.root,
        systems_output_root=args.systems_out,
        data=args.data,
        model=args.model,
        backend=args.backend,
        method=args.method,
        populations=parse_int_tuple(args.populations),
        prompts=args.prompts,
        holdout_prompts=args.holdout_prompts,
        promote=args.promote,
        rank=args.rank,
        sigma=args.sigma,
        basis_rank=args.basis_rank,
        basis_prompts=args.basis_prompts,
        target_preset=args.target_preset,
        scale_mode=args.scale_mode,
        rho_grid=args.rho_grid,
        sigma_w_grid=args.sigma_w_grid,
        budget_policy=args.budget_policy,
        basis_kind=args.basis_kind,
        top_k_grid=args.top_k_grid,
        seed=args.seed,
        targets=args.targets,
        max_new_tokens=args.max_new_tokens,
        prompt_variants=args.prompt_variants,
        prompt_input=args.prompt_input,
        use_chat_template=args.use_chat_template,
        max_base_malformed_for_selection=args.max_base_malformed_for_selection,
        max_base_cap_hit_for_selection=args.max_base_cap_hit_for_selection,
        min_selection_prompt_variants=args.min_selection_prompt_variants,
        require_all_prompt_variants_valid=args.require_all_prompt_variants_valid,
        chunk_adapters=args.chunk_adapters,
        max_loras=args.max_loras,
        max_cpu_loras=args.max_cpu_loras,
        tensor_parallel_size=args.tensor_parallel_size,
        enable_prefix_caching=args.enable_prefix_caching,
        enable_chunked_prefill=args.enable_chunked_prefill,
        kv_cache_dtype=args.kv_cache_dtype,
        vllm_kwargs=tuple(args.vllm_kwarg),
        keep_adapters=args.keep_adapters,
        bench_adapters=parse_int_tuple(args.bench_adapters),
        halving_population=args.halving_population,
        halving_stage_prompts=args.halving_stage_prompts,
        halving_survivors=args.halving_survivors,
        run_halving=args.run_halving and not args.skip_halving,
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
    try:
        payload = plan_payload(config)
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
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
