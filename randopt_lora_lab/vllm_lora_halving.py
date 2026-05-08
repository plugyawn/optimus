from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
import time
import traceback
from dataclasses import asdict
from importlib import metadata
from pathlib import Path

from .countdown import load_examples, unique_example_count, unique_semantic_example_count
from .vllm_lora_search import (
    base_eval,
    candidate_panel,
    import_vllm_lora_request,
    make_adapter_specs,
    mixed_eval,
    parse_targets,
    write_json,
    write_jsonl,
)
from .vllm_lora_bench import make_sampling_params


def reset_outputs(out: Path) -> None:
    for name in [
        "adapters.jsonl",
        "stage_candidate_summary.jsonl",
        "stage_per_prompt.jsonl",
        "candidate_summary.jsonl",
        "per_prompt.jsonl",
        "holdout_per_prompt.jsonl",
    ]:
        path = out / name
        if path.exists():
            path.unlink()


def run_halving(args) -> dict:
    targets = parse_targets(args.targets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out)
    write_json(out / "args.json", vars(args))

    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    stage = screen[: min(args.stage_prompts, len(screen))]
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic)

    adapter_start = time.time()
    specs = make_adapter_specs(args, out, targets, candidates)
    adapter_build_s = time.time() - adapter_start
    write_jsonl(out / "adapters.jsonl", [asdict(spec) for spec in specs])

    LLM, SamplingParams, LoRARequest = import_vllm_lora_request()
    sampling = make_sampling_params(SamplingParams, args.max_new_tokens, args.stop_at_answer)
    load_start = time.time()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_loras=args.max_loras,
        max_lora_rank=args.rank,
        max_cpu_loras=max(args.max_cpu_loras, len(specs)),
        enforce_eager=args.enforce_eager,
        **({"max_num_batched_tokens": args.max_num_batched_tokens} if args.max_num_batched_tokens else {}),
    )
    load_s = time.time() - load_start

    base_stage_rows, base_stage = base_eval(llm, sampling, stage, args, mode="base_stage")
    base_screen_rows, base_screen = base_eval(llm, sampling, screen, args, mode="base_screen")
    base_holdout_rows, base_holdout = base_eval(llm, sampling, holdout, args, mode="base_holdout")
    write_jsonl(out / "stage_per_prompt.jsonl", base_stage_rows)
    write_jsonl(out / "per_prompt.jsonl", base_screen_rows)
    write_jsonl(out / "holdout_per_prompt.jsonl", base_holdout_rows)

    stage_rows, stage_candidate_rows, stage_aggregate = mixed_eval(
        llm,
        LoRARequest,
        sampling,
        stage,
        specs,
        args,
        mode="stage",
    )
    write_jsonl(out / "stage_per_prompt.jsonl", stage_rows)
    write_jsonl(out / "stage_candidate_summary.jsonl", stage_candidate_rows)

    survivor_n = min(args.survivors, len(stage_candidate_rows))
    survivors = sorted(stage_candidate_rows, key=lambda r: r["exact_mean"], reverse=True)[:survivor_n]
    by_candidate = {spec.candidate: spec for spec in specs}
    survivor_specs = [by_candidate[row["candidate"]] for row in survivors]
    screen_rows, screen_candidate_rows, screen_aggregate = mixed_eval(
        llm,
        LoRARequest,
        sampling,
        screen,
        survivor_specs,
        args,
        mode="screen",
    )
    stage_by_candidate = {row["candidate"]: row for row in stage_candidate_rows}
    for row in screen_candidate_rows:
        row["stage_exact_mean"] = stage_by_candidate[row["candidate"]]["exact_mean"]
    write_jsonl(out / "per_prompt.jsonl", screen_rows)
    write_jsonl(out / "candidate_summary.jsonl", screen_candidate_rows)

    top = sorted(screen_candidate_rows, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(screen_candidate_rows))]
    top_specs = [by_candidate[row["candidate"]] for row in top]
    holdout_rows, holdout_candidate_rows, holdout_aggregate = mixed_eval(
        llm,
        LoRARequest,
        sampling,
        holdout,
        top_specs,
        args,
        mode="holdout",
    )
    write_jsonl(out / "holdout_per_prompt.jsonl", holdout_rows)
    holdout_by_candidate = {row["candidate"]: row for row in holdout_candidate_rows}
    top_holdout = [holdout_by_candidate[row["candidate"]] for row in top if row["candidate"] in holdout_by_candidate]

    effective_prompt_evals = len(specs) * len(stage) + len(survivor_specs) * len(screen) + len(top_specs) * len(holdout)
    full_prompt_evals = len(specs) * len(screen) + len(top_specs) * len(holdout)
    eval_elapsed_s = stage_aggregate["elapsed_s"] + screen_aggregate["elapsed_s"] + holdout_aggregate["elapsed_s"]
    adapters_kept = bool(args.keep_adapters or args.adapter_dir)
    if not adapters_kept:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    summary = {
        "kind": "vllm_lora_halving",
        "model": args.model,
        "family": args.family,
        "population": len(specs),
        "survivors": len(survivor_specs),
        "promote": args.promote,
        "rank": args.rank,
        "sigma": args.sigma,
        "targets": targets,
        "antithetic": args.antithetic,
        "stage_prompts": len(stage),
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "stage_unique_prompts": unique_example_count(stage),
        "screen_unique_prompts": unique_example_count(screen),
        "holdout_unique_prompts": unique_example_count(holdout),
        "stage_unique_semantic_prompts": unique_semantic_example_count(stage),
        "screen_unique_semantic_prompts": unique_semantic_example_count(screen),
        "holdout_unique_semantic_prompts": unique_semantic_example_count(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "max_loras": args.max_loras,
        "chunk_adapters": args.chunk_adapters,
        "enforce_eager": args.enforce_eager,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "adapter_build_s": adapter_build_s,
        "adapters_kept": adapters_kept,
        "load_s": load_s,
        "base_stage_exact": base_stage["exact_mean"],
        "base_screen_exact": base_screen["exact_mean"],
        "base_holdout_exact": base_holdout["exact_mean"],
        "stage_tokens_per_sec": stage_aggregate["tokens_per_sec"],
        "stage_prompts_per_sec": stage_aggregate["prompts_per_sec"],
        "stage_candidate_sec": stage_aggregate["candidate_sec"],
        "screen_tokens_per_sec": screen_aggregate["tokens_per_sec"],
        "screen_prompts_per_sec": screen_aggregate["prompts_per_sec"],
        "screen_candidate_sec": screen_aggregate["candidate_sec"],
        "eval_elapsed_s": eval_elapsed_s,
        "prompt_eval_sec": effective_prompt_evals / max(eval_elapsed_s, 1e-9),
        "effective_prompt_evals": effective_prompt_evals,
        "full_prompt_evals": full_prompt_evals,
        "prompt_eval_savings": 1.0 - (effective_prompt_evals / max(full_prompt_evals, 1)),
        "candidate_sec": len(specs) / max(eval_elapsed_s, 1e-9),
        "top_stage": sorted(stage_candidate_rows, key=lambda r: r["exact_mean"], reverse=True)[: min(args.promote, len(stage_candidate_rows))],
        "top_screen": top,
        "top_holdout": top_holdout,
    }
    write_json(out / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def diagnostic_payload(args, exc: BaseException) -> dict:
    return {
        "kind": "vllm_lora_halving_failure",
        "argv": sys.argv,
        "args": vars(args) if args is not None else None,
        "error_type": type(exc).__name__,
        "error": str(exc),
        "traceback": traceback.format_exc(),
        "python": sys.version,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "versions": {
            "vllm": package_version("vllm"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "safetensors": package_version("safetensors"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Run staged mixed-batch vLLM LoRA RandOpt search.")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=64)
    p.add_argument("--stage-prompts", type=int, default=8)
    p.add_argument("--holdout-prompts", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--population", type=int, default=512)
    p.add_argument("--survivors", type=int, default=64)
    p.add_argument("--promote", type=int, default=0)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.0075)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument(
        "--family",
        default="isotropic",
        choices=[
            "isotropic",
            "factor_gaussian_lora",
            "projected_gaussian_rank_r",
            "randomized_projected_gaussian_rank_r",
            "spectral_projected_gaussian_rank_r",
            "spectral_projected_gaussian_rank_r_c0p5",
            "spectral_projected_gaussian_rank_r_c0p75",
            "spectral_projected_gaussian_rank_r_c1p25",
            "spectral_projected_gaussian_rank_r_c1p5",
            "spectral_projected_gaussian_rank_r_c2",
            "activation_projected_gaussian_rank_r",
            "activation_projected_gaussian_rank_r_c0p5",
            "activation_projected_gaussian_rank_r_c0p75",
            "activation_projected_gaussian_rank_r_c1p25",
            "activation_projected_gaussian_rank_r_c1p5",
            "activation_projected_gaussian_rank_r_c2",
        ],
    )
    p.add_argument("--antithetic", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--stop-at-answer", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--adapter-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-num-batched-tokens", type=int, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-loras", type=int, default=16)
    p.add_argument("--max-cpu-loras", type=int, default=2048)
    p.add_argument("--chunk-adapters", type=int, default=16)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--keep-adapters", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-repeat-data", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    try:
        run_halving(args)
        return 0
    except Exception as exc:
        payload = diagnostic_payload(args, exc)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "diagnostic.json", payload)
        write_json(out / "summary.json", payload)
        shutil.rmtree(out / "adapters", ignore_errors=True)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
