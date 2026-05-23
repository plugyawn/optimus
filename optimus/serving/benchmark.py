from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from optimus.core.candidates import SearchCandidate as Candidate
from optimus.modeling.lora import AdapterSpec, parse_targets, save_seed_adapter
from optimus.modeling.qwen import SUPPORTED_QWEN_LORA_TARGETS, load_qwen_lora_config, qwen_lora_shapes
from optimus.serving.prompting import make_vllm_prompt_inputs
from optimus.serving.runtime import (
    import_vllm_lora_request,
    make_sampling_params,
    package_version,
    score_mixed_rows,
    score_rows,
    timed_generate,
    write_json,
    write_jsonl,
)

from optimus.tasks.countdown import load_examples, prompts as make_prompts


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TARGETS = "q_proj,v_proj"
SUPPORTED_QWEN_TARGETS = SUPPORTED_QWEN_LORA_TARGETS


def reset_jsonl_outputs(out: Path) -> None:
    for name in ["adapters.jsonl", "adapter_rows.jsonl", "per_prompt.jsonl"]:
        path = out / name
        if path.exists():
            path.unlink()


def make_adapter_specs(args, out: Path, targets: list[str]) -> list[AdapterSpec]:
    config = load_qwen_lora_config(args.model, local_files_only=args.local_files_only)

    adapter_root = Path(args.adapter_dir) if args.adapter_dir else out / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)
    seeds = [rng.randrange(1, 2**31 - 1) for _ in range(args.adapters)]
    specs = []
    for idx, seed in enumerate(seeds):
        candidate = Candidate(args.family, seed, args.sigma, 1)
        name = f"randopt_seed{seed}_s{args.sigma:g}"
        path = adapter_root / f"{idx:04d}_{name}"
        save_seed_adapter(
            path,
            model=args.model,
            candidate=candidate,
            rank=args.rank,
            targets=targets,
            config=config,
            tensor_dtype=args.adapter_dtype,
        )
        specs.append(
            AdapterSpec(
                index=idx,
                name=name,
                lora_int_id=idx + 1,
                path=str(path.resolve()),
                candidate=candidate.key,
                seed=seed,
                sigma=args.sigma,
                sign=1,
            )
        )
    return specs


def run_benchmark(args) -> dict:
    targets = parse_targets(args.targets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_jsonl_outputs(out)
    write_json(out / "args.json", vars(args))

    examples = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    prompt_texts = make_prompts(examples)

    adapter_start = time.time()
    specs = make_adapter_specs(args, out, targets)
    adapter_build_s = time.time() - adapter_start
    write_jsonl(out / "adapters.jsonl", [asdict(spec) for spec in specs])

    if args.prepare_only:
        summary = {
            "kind": "vllm_lora_bench_prepare_only",
            "model": args.model,
            "adapters": len(specs),
            "rank": args.rank,
            "targets": targets,
            "adapter_build_s": adapter_build_s,
        }
        write_json(out / "summary.json", summary)
        return summary

    LLM, SamplingParams, LoRARequest = import_vllm_lora_request()
    sampling = make_sampling_params(SamplingParams, args.max_new_tokens, args.stop_at_answer)
    load_start = time.time()
    llm = LLM(
        model=args.model,
        dtype=args.dtype,
        trust_remote_code=True,
        tensor_parallel_size=args.tensor_parallel_size,
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
    prompt_inputs = make_vllm_prompt_inputs(prompt_texts, llm.get_tokenizer(), args.prompt_input)

    requests = [
        LoRARequest(spec.name, spec.lora_int_id, spec.path)
        for spec in specs
    ]
    preload_rows = []
    if args.preload:
        preload_sampling = SamplingParams(max_tokens=1, temperature=0.0)
        for spec, req in zip(specs, requests):
            start = time.time()
            llm.generate([prompt_inputs[0]], preload_sampling, lora_request=req, use_tqdm=False)
            preload_rows.append(
                {
                    "mode": "preload",
                    "candidate": spec.candidate,
                    "adapter": spec.name,
                    "elapsed_s": time.time() - start,
                }
            )
        write_jsonl(out / "adapter_rows.jsonl", preload_rows)

    per_prompt_rows = []
    adapter_rows = []
    if args.include_base:
        outputs, elapsed_s = timed_generate(llm, prompt_inputs, sampling)
        rows, metrics = score_rows(
            mode="base",
            candidate="base",
            examples=examples,
            outputs=outputs,
            max_new_tokens=args.max_new_tokens,
        )
        per_prompt_rows.extend(rows)
        adapter_rows.append(
            {
                "mode": "base",
                "adapter_index": None,
                "adapter": None,
                "candidate": "base",
                "repeat": 0,
                "elapsed_s": elapsed_s,
                "tokens_per_sec": metrics["output_tokens"] / max(elapsed_s, 1e-9),
                "prompts_per_sec": len(prompt_texts) / max(elapsed_s, 1e-9),
                **metrics,
            }
        )

    if not args.skip_sequential:
        for repeat in range(args.repeats):
            for spec, req in zip(specs, requests):
                outputs, elapsed_s = timed_generate(llm, prompt_inputs, sampling, lora_request=req)
                rows, metrics = score_rows(
                    mode="sequential",
                    candidate=spec.candidate,
                    examples=examples,
                    outputs=outputs,
                    max_new_tokens=args.max_new_tokens,
                )
                per_prompt_rows.extend(rows)
                adapter_rows.append(
                    {
                        "mode": "sequential",
                        "adapter_index": spec.index,
                        "adapter": spec.name,
                        "candidate": spec.candidate,
                        "repeat": repeat,
                        "elapsed_s": elapsed_s,
                        "tokens_per_sec": metrics["output_tokens"] / max(elapsed_s, 1e-9),
                        "prompts_per_sec": len(prompt_texts) / max(elapsed_s, 1e-9),
                        **metrics,
                    }
                )

    mixed_rows = []
    if args.mixed_batch:
        mixed_prompts = []
        mixed_requests = []
        for req in requests:
            mixed_prompts.extend(prompt_inputs)
            mixed_requests.extend([req] * len(prompt_texts))
        outputs, elapsed_s = timed_generate(llm, mixed_prompts, sampling, lora_request=mixed_requests)
        rows, metrics = score_mixed_rows(
            examples=examples,
            outputs=outputs,
            specs=specs,
            prompts_per_adapter=len(prompt_texts),
            max_new_tokens=args.max_new_tokens,
        )
        mixed_rows.extend(rows)
        per_prompt_rows.extend(rows)
        adapter_rows.append(
            {
                "mode": "mixed",
                "adapter_index": None,
                "adapter": "mixed",
                "candidate": "mixed",
                "repeat": 0,
                "elapsed_s": elapsed_s,
                "tokens_per_sec": metrics["output_tokens"] / max(elapsed_s, 1e-9),
                "prompts_per_sec": len(mixed_prompts) / max(elapsed_s, 1e-9),
                "by_candidate": metrics.pop("by_candidate"),
                **metrics,
            }
        )

    if per_prompt_rows:
        write_jsonl(out / "per_prompt.jsonl", per_prompt_rows)
    if adapter_rows:
        write_jsonl(out / "adapter_rows.jsonl", adapter_rows)

    lora_rows = [r for r in adapter_rows if r["mode"] == "sequential"]
    lora_tokens = sum(float(r["output_tokens"]) for r in lora_rows)
    lora_elapsed = sum(float(r["elapsed_s"]) for r in lora_rows)
    best = max(lora_rows, key=lambda r: r["tokens_per_sec"]) if lora_rows else {}
    mixed = next((r for r in adapter_rows if r["mode"] == "mixed"), {})
    summary = {
        "kind": "vllm_lora_bench",
        "model": args.model,
        "vllm_version": package_version("vllm"),
        "transformers_version": package_version("transformers"),
        "torch_version": package_version("torch"),
        "adapters": len(specs),
        "rank": args.rank,
        "sigma": args.sigma,
        "family": args.family,
        "targets": targets,
        "prompts": len(prompt_texts),
        "prompt_input": args.prompt_input,
        "repeats": args.repeats,
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "enforce_eager": args.enforce_eager,
        "tensor_parallel_size": args.tensor_parallel_size,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "adapter_build_s": adapter_build_s,
        "load_s": load_s,
        "preload": args.preload,
        "preload_s": sum(float(r["elapsed_s"]) for r in preload_rows),
        "skip_sequential": args.skip_sequential,
        "lora_elapsed_s": lora_elapsed,
        "lora_output_tokens": lora_tokens,
        "lora_tokens_per_sec": None if not lora_rows else lora_tokens / max(lora_elapsed, 1e-9),
        "lora_prompts_per_sec": None
        if not lora_rows
        else (len(prompt_texts) * len(specs) * args.repeats) / max(lora_elapsed, 1e-9),
        "best_adapter_tokens_per_sec": best.get("tokens_per_sec"),
        "best_adapter_prompts_per_sec": best.get("prompts_per_sec"),
        "adapter_rows": len(lora_rows),
        "base_tokens_per_sec": next(
            (r["tokens_per_sec"] for r in adapter_rows if r["mode"] == "base"),
            None,
        ),
        "mixed_batch": args.mixed_batch,
        "mixed_tokens_per_sec": mixed.get("tokens_per_sec"),
        "mixed_prompts_per_sec": mixed.get("prompts_per_sec"),
        "mixed_exact_mean": mixed.get("exact_mean"),
    }
    write_json(out / "summary.json", summary)
    return summary


def diagnostic_payload(args, exc: BaseException) -> dict:
    return {
        "kind": "vllm_lora_bench_failure",
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
            "peft": package_version("peft"),
            "safetensors": package_version("safetensors"),
        },
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Benchmark vLLM dynamic LoRA adapter serving on fixed Countdown prompts."
    )
    p.add_argument("--out", required=True)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--adapters", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.02)
    p.add_argument("--targets", default=DEFAULT_TARGETS)
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
            "activation_generalized_projected_gaussian_rank_r",
            "activation_generalized_projected_gaussian_rank_r_c0p5",
            "activation_generalized_projected_gaussian_rank_r_c0p75",
            "activation_generalized_projected_gaussian_rank_r_c1p25",
            "activation_generalized_projected_gaussian_rank_r_c1p5",
            "activation_generalized_projected_gaussian_rank_r_c2",
            "activation_generalized_spectral_lora",
            "activation_generalized_spectral_lora_c0p5",
            "activation_generalized_spectral_lora_c0p75",
            "activation_generalized_spectral_lora_c1p25",
            "activation_generalized_spectral_lora_c1p5",
            "activation_generalized_spectral_lora_c2",
            "activation_generalized_spectral_lora_sv",
            "activation_generalized_spectral_lora_sv_c0p75",
            "activation_generalized_spectral_lora_sv_c1p25",
            "activation_generalized_spectral_lora_sv_c1p5",
            "activation_generalized_spectral_lora_sv_c2",
            "sparse_low_rank_lora",
            "sparse_low_rank_lora_d0p125",
            "sparse_low_rank_lora_d0p25",
            "sparse_low_rank_lora_d0p5",
        ],
    )
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--prompt-input", default="text", choices=["text", "token_ids"])
    p.add_argument("--stop-at-answer", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--adapter-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--tensor-parallel-size", type=int, default=1)
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-num-batched-tokens", type=int, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-loras", type=int, default=8)
    p.add_argument("--max-cpu-loras", type=int, default=64)
    p.add_argument("--repeats", type=int, default=1)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--preload", action="store_true", help="Warm-load every adapter before timed rows.")
    p.add_argument("--mixed-batch", action="store_true", help="Evaluate all adapters in one vLLM request batch.")
    p.add_argument("--skip-sequential", action="store_true", help="Skip slow one-adapter-at-a-time rows.")
    p.add_argument("--include-base", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--prepare-only", action="store_true", help="Only write deterministic adapter files.")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-repeat-data", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out = Path(args.out)
    try:
        summary = run_benchmark(args)
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        payload = diagnostic_payload(args, exc)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "diagnostic.json", payload)
        write_json(out / "summary.json", payload)
        print(json.dumps(payload, indent=2, sort_keys=True), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
