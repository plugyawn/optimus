from __future__ import annotations

import json
import os
import platform
import sys
import time
from collections.abc import Iterable
from importlib import metadata
from pathlib import Path

from optimus.modeling.lora import AdapterSpec
from optimus.tasks.countdown import CountdownExample, score_completion


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        for row in rows:
            f.write(json.dumps(row, sort_keys=True) + "\n")


def package_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def runtime_environment() -> dict:
    payload = {
        "python": sys.version,
        "platform": platform.platform(),
        "versions": {
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
            "peft": package_version("peft"),
            "vllm": package_version("vllm"),
            "lighteval": package_version("lighteval"),
            "flashinfer-python": package_version("flashinfer-python"),
            "flash-attn": package_version("flash-attn"),
            "triton": package_version("triton"),
            "safetensors": package_version("safetensors"),
        },
        "vllm_env": {
            "VLLM_ATTENTION_BACKEND": os.environ.get("VLLM_ATTENTION_BACKEND"),
            "VLLM_ENABLE_V1_MULTIPROCESSING": os.environ.get("VLLM_ENABLE_V1_MULTIPROCESSING"),
            "VLLM_WORKER_MULTIPROC_METHOD": os.environ.get("VLLM_WORKER_MULTIPROC_METHOD"),
        },
    }
    try:
        import torch

        cuda_available = bool(torch.cuda.is_available())
        gpus = []
        if cuda_available:
            for idx in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(idx)
                gpus.append(
                    {
                        "index": idx,
                        "name": props.name,
                        "total_memory_bytes": int(props.total_memory),
                        "major": int(props.major),
                        "minor": int(props.minor),
                        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(idx)),
                        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved(idx)),
                    }
                )
        payload["cuda"] = {
            "available": cuda_available,
            "device_count": len(gpus),
            "torch_cuda_version": torch.version.cuda,
            "gpus": gpus,
        }
    except Exception as exc:
        payload["cuda"] = {"available": False, "error": f"{type(exc).__name__}: {exc}"}
    return payload


def configure_vllm_logging() -> None:
    os.environ.setdefault("VLLM_LOGGING_LEVEL", "ERROR")


def parse_backend_value(text: str):
    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in {"none", "null"}:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def parse_backend_kwargs(items: Iterable[str]) -> dict:
    kwargs = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"backend kwarg {item!r} must use KEY=VALUE syntax.")
        key, value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"backend kwarg {item!r} has an empty key.")
        kwargs[key] = parse_backend_value(value)
    return kwargs


def optional_vllm_kwargs(args) -> dict:
    kwargs = {}
    if getattr(args, "max_num_batched_tokens", 0):
        kwargs["max_num_batched_tokens"] = args.max_num_batched_tokens
    if getattr(args, "enable_prefix_caching", None) is not None:
        kwargs["enable_prefix_caching"] = args.enable_prefix_caching
    if getattr(args, "enable_chunked_prefill", None) is not None:
        kwargs["enable_chunked_prefill"] = args.enable_chunked_prefill
    if getattr(args, "kv_cache_dtype", ""):
        kwargs["kv_cache_dtype"] = args.kv_cache_dtype
    kwargs.update(parse_backend_kwargs(getattr(args, "vllm_kwarg", []) or []))
    return kwargs


def import_vllm_lora_request():
    configure_vllm_logging()
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except Exception as exc:
        raise RuntimeError(
            "vLLM with LoRA support is required for serving. Install vllm in the run environment."
        ) from exc
    return LLM, SamplingParams, LoRARequest


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


__all__ = [
    "extract_output",
    "configure_vllm_logging",
    "import_vllm_lora_request",
    "make_sampling_params",
    "package_version",
    "runtime_environment",
    "score_mixed_rows",
    "score_rows",
    "timed_generate",
    "write_json",
    "write_jsonl",
]
