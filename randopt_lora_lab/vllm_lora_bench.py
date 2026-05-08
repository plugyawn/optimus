from __future__ import annotations

import argparse
import json
import os
import platform
import random
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Iterable

from .countdown import CountdownExample, load_examples, prompts as make_prompts, score_completion
from .lora_space import Candidate, lora_noise_tensors
from .vllm_prompting import make_vllm_prompt_inputs


DEFAULT_MODEL = "Qwen/Qwen2.5-3B-Instruct"
DEFAULT_TARGETS = "q_proj,v_proj"
SUPPORTED_QWEN_TARGETS = {
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
}


@dataclass(frozen=True)
class AdapterSpec:
    index: int
    name: str
    lora_int_id: int
    path: str
    candidate: str
    seed: int
    sigma: float
    sign: int


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def reset_jsonl_outputs(out: Path) -> None:
    for name in ["adapters.jsonl", "adapter_rows.jsonl", "per_prompt.jsonl"]:
        path = out / name
        if path.exists():
            path.unlink()


def parse_targets(text: str) -> list[str]:
    targets = [x.strip() for x in text.split(",") if x.strip()]
    if not targets:
        raise ValueError("--targets must contain at least one module suffix")
    unknown = sorted(set(targets) - SUPPORTED_QWEN_TARGETS)
    if unknown:
        raise ValueError(
            "Direct adapter generation currently supports Qwen2-style targets "
            f"{sorted(SUPPORTED_QWEN_TARGETS)}, got unsupported targets {unknown}."
        )
    return targets


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def qwen_lora_shapes(config, targets: list[str]) -> list[tuple[str, int, int]]:
    hidden = int(config.hidden_size)
    intermediate = int(config.intermediate_size)
    layers = int(config.num_hidden_layers)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    kv_out = kv_heads * head_dim

    dims = {
        "q_proj": ("self_attn", hidden, hidden),
        "k_proj": ("self_attn", hidden, kv_out),
        "v_proj": ("self_attn", hidden, kv_out),
        "o_proj": ("self_attn", hidden, hidden),
        "gate_proj": ("mlp", hidden, intermediate),
        "up_proj": ("mlp", hidden, intermediate),
        "down_proj": ("mlp", intermediate, hidden),
    }
    shapes = []
    for layer_idx in range(layers):
        for target in targets:
            block, in_features, out_features = dims[target]
            module = f"model.layers.{layer_idx}.{block}.{target}"
            shapes.append((module, in_features, out_features))
    return shapes


def adapter_config(model: str, rank: int, targets: list[str]) -> dict:
    return {
        "base_model_name_or_path": model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }


def save_seed_adapter(
    adapter_dir: Path,
    *,
    model: str,
    candidate: Candidate,
    rank: int,
    targets: list[str],
    config,
    tensor_dtype: str,
    family_state: dict | None = None,
) -> None:
    import torch
    from safetensors.torch import save_file

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[tensor_dtype]
    tensors = {}
    for module, in_features, out_features in qwen_lora_shapes(config, targets):
        a, b = lora_noise_tensors(
            module,
            (rank, in_features),
            (out_features, rank),
            candidate,
            rank,
            family_state=family_state,
            state_key=module,
        )
        prefix = f"base_model.model.{module}"
        tensors[f"{prefix}.lora_A.weight"] = a.to(dtype).contiguous()
        tensors[f"{prefix}.lora_B.weight"] = b.to(dtype).contiguous()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    write_json(adapter_dir / "adapter_config.json", adapter_config(model, rank, targets))
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})


def make_adapter_specs(args, out: Path, targets: list[str]) -> list[AdapterSpec]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model_type = str(getattr(config, "model_type", ""))
    if not model_type.startswith("qwen2"):
        raise ValueError(
            f"{args.model} has model_type={model_type!r}; direct shape generation is only validated for Qwen2."
        )

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


def import_vllm_lora_request():
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as exc:
        raise RuntimeError(
            "vLLM with LoRA support is required for serving. Install vllm in the run environment."
        ) from exc
    return LLM, SamplingParams, LoRARequest


def extract_output(item) -> tuple[str, int]:
    if not item.outputs:
        return "", 0
    output = item.outputs[0]
    return output.text, len(output.token_ids or [])


def score_rows(
    *,
    mode: str,
    candidate: str,
    examples: list[CountdownExample],
    outputs,
    max_new_tokens: int,
) -> tuple[list[dict], dict]:
    rows = []
    exact = []
    malformed = []
    cap_hits = []
    answer_closed = []
    output_tokens = 0
    for ex, item in zip(examples, outputs):
        text, tokens = extract_output(item)
        output_tokens += tokens
        score = score_completion(text, ex)
        cap_hit = float(tokens >= max_new_tokens)
        closed = float("</answer>" in text)
        exact.append(float(score["exact"]))
        malformed.append(float(score["malformed"]))
        cap_hits.append(cap_hit)
        answer_closed.append(closed)
        rows.append(
            {
                "mode": mode,
                "candidate": candidate,
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "text": text,
                "output_tokens": tokens,
                "cap_hit": cap_hit,
                "answer_closed": closed,
                **score,
            }
        )
    metrics = {
        "exact_mean": sum(exact) / max(len(exact), 1),
        "malformed_mean": sum(malformed) / max(len(malformed), 1),
        "cap_hit_mean": sum(cap_hits) / max(len(cap_hits), 1),
        "answer_closed_mean": sum(answer_closed) / max(len(answer_closed), 1),
        "output_tokens": output_tokens,
    }
    return rows, metrics


def score_mixed_rows(
    *,
    examples: list[CountdownExample],
    outputs,
    specs: list[AdapterSpec],
    prompts_per_adapter: int,
    max_new_tokens: int,
) -> tuple[list[dict], dict]:
    rows = []
    exact = []
    malformed = []
    cap_hits = []
    answer_closed = []
    output_tokens = 0
    by_candidate: dict[str, dict] = {}
    for idx, item in enumerate(outputs):
        adapter_index = idx // prompts_per_adapter
        example_index = idx % prompts_per_adapter
        spec = specs[adapter_index]
        ex = examples[example_index]
        text, tokens = extract_output(item)
        output_tokens += tokens
        score = score_completion(text, ex)
        cap_hit = float(tokens >= max_new_tokens)
        closed = float("</answer>" in text)
        exact.append(float(score["exact"]))
        malformed.append(float(score["malformed"]))
        cap_hits.append(cap_hit)
        answer_closed.append(closed)
        bucket = by_candidate.setdefault(
            spec.candidate,
            {"exact": [], "malformed": [], "cap_hit": [], "answer_closed": [], "output_tokens": 0},
        )
        bucket["exact"].append(float(score["exact"]))
        bucket["malformed"].append(float(score["malformed"]))
        bucket["cap_hit"].append(cap_hit)
        bucket["answer_closed"].append(closed)
        bucket["output_tokens"] += tokens
        rows.append(
            {
                "mode": "mixed",
                "candidate": spec.candidate,
                "adapter_index": spec.index,
                "adapter": spec.name,
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "text": text,
                "output_tokens": tokens,
                "cap_hit": cap_hit,
                "answer_closed": closed,
                **score,
            }
        )
    metrics = {
        "exact_mean": sum(exact) / max(len(exact), 1),
        "malformed_mean": sum(malformed) / max(len(malformed), 1),
        "cap_hit_mean": sum(cap_hits) / max(len(cap_hits), 1),
        "answer_closed_mean": sum(answer_closed) / max(len(answer_closed), 1),
        "output_tokens": output_tokens,
        "by_candidate": {
            candidate: {
                "exact_mean": sum(values["exact"]) / max(len(values["exact"]), 1),
                "malformed_mean": sum(values["malformed"]) / max(len(values["malformed"]), 1),
                "cap_hit_mean": sum(values["cap_hit"]) / max(len(values["cap_hit"]), 1),
                "answer_closed_mean": sum(values["answer_closed"]) / max(len(values["answer_closed"]), 1),
                "output_tokens": values["output_tokens"],
            }
            for candidate, values in by_candidate.items()
        },
    }
    return rows, metrics


def timed_generate(llm, prompts, sampling, *, lora_request=None) -> tuple[list, float]:
    start = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=lora_request, use_tqdm=False)
    return outputs, time.time() - start


def make_sampling_params(SamplingParams, max_tokens: int, stop_at_answer: bool):
    kwargs = {"max_tokens": max_tokens, "temperature": 0.0}
    if stop_at_answer:
        kwargs["stop"] = ["</answer>"]
        kwargs["include_stop_str_in_output"] = True
    try:
        return SamplingParams(**kwargs)
    except TypeError:
        kwargs.pop("include_stop_str_in_output", None)
        return SamplingParams(**kwargs)


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
