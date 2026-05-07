from __future__ import annotations

import argparse
import json
import math
import shutil
import time
from dataclasses import asdict
from pathlib import Path

from .backend_contract import backend_contract, vllm_tokenizer_contract
from .backends import TransformersLoraBackend
from .countdown import load_examples, prompts as make_prompts
from .lora_space import Candidate
from .vllm_lora_bench import AdapterSpec, import_vllm_lora_request, parse_targets, save_seed_adapter, write_json, write_jsonl


def parse_candidate_key(key: str) -> Candidate:
    parts = key.split(":")
    if len(parts) != 4:
        raise ValueError(f"cannot parse candidate key: {key!r}")
    return Candidate(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


def safe_name(label: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in label)


def logprob_value(value) -> tuple[float, str]:
    if hasattr(value, "logprob"):
        return float(value.logprob), str(getattr(value, "decoded_token", ""))
    if isinstance(value, dict):
        return float(value.get("logprob", float("nan"))), str(value.get("decoded_token", ""))
    return float(value), ""


def normalize_vllm_logprobs(logprobs) -> list[dict]:
    if not logprobs:
        return []
    rows = []
    for token_id, value in dict(logprobs).items():
        lp, token = logprob_value(value)
        rows.append({"token_id": int(token_id), "logprob": lp, "token": token})
    return sorted(rows, key=lambda row: row["logprob"], reverse=True)


def compare_topk(left: list[dict], right: list[dict]) -> dict:
    left_ids = [row["token_id"] for row in left]
    right_ids = [row["token_id"] for row in right]
    common = sorted(set(left_ids) & set(right_ids))
    left_lp = {row["token_id"]: row["logprob"] for row in left}
    right_lp = {row["token_id"]: row["logprob"] for row in right}
    deltas = [abs(left_lp[token_id] - right_lp[token_id]) for token_id in common if math.isfinite(left_lp[token_id]) and math.isfinite(right_lp[token_id])]
    return {
        "top1_equal": bool(left_ids and right_ids and left_ids[0] == right_ids[0]),
        "topk_overlap": len(common),
        "topk_union": len(set(left_ids) | set(right_ids)),
        "max_common_abs_logprob_delta": max(deltas) if deltas else None,
        "mean_common_abs_logprob_delta": sum(deltas) / len(deltas) if deltas else None,
    }


def hf_topk(backend: TransformersLoraBackend, prompt: str, top_k: int) -> list[dict]:
    import torch

    logits = backend.logits_signature([prompt])[0]
    logprobs = torch.log_softmax(logits.float(), dim=-1)
    values, indices = torch.topk(logprobs, k=top_k)
    rows = []
    for token_id, value in zip(indices.tolist(), values.tolist()):
        rows.append(
            {
                "token_id": int(token_id),
                "logprob": float(value),
                "token": backend.tokenizer.decode([int(token_id)]),
            }
        )
    return rows


def make_adapter_specs(args, out: Path, targets: list[str], candidates: list[tuple[str, Candidate]]) -> dict[str, AdapterSpec]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True, local_files_only=args.local_files_only)
    adapter_root = out / "adapters"
    specs = {}
    for idx, (label, candidate) in enumerate(candidates):
        path = adapter_root / f"{idx:04d}_{safe_name(label)}"
        save_seed_adapter(
            path,
            model=args.model,
            candidate=candidate,
            rank=args.rank,
            targets=targets,
            config=config,
            tensor_dtype=args.adapter_dtype,
        )
        specs[label] = AdapterSpec(
            index=idx,
            name=safe_name(label),
            lora_int_id=idx + 1,
            path=str(path.resolve()),
            candidate=candidate.key,
            seed=candidate.seed,
            sigma=candidate.sigma,
            sign=candidate.sign,
        )
    return specs


def vllm_topk(llm, SamplingParams, LoRARequest, prompt: str, spec: AdapterSpec | None, top_k: int) -> tuple[list[dict], int | None, str]:
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=top_k)
    request = None if spec is None else LoRARequest(spec.name, spec.lora_int_id, spec.path)
    outputs = llm.generate([prompt], sampling, lora_request=request, use_tqdm=False)
    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids or [])
    generated_id = int(token_ids[0]) if token_ids else None
    generated_text = output.text
    logprobs = output.logprobs[0] if getattr(output, "logprobs", None) else {}
    return normalize_vllm_logprobs(logprobs), generated_id, generated_text


def summarize(rows: list[dict]) -> dict:
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        by_condition[condition] = {
            "n": len(subset),
            "top1_equal_rate": sum(row["top1_equal"] for row in subset) / max(len(subset), 1),
            "mean_topk_overlap": sum(row["topk_overlap"] for row in subset) / max(len(subset), 1),
            "mean_text_equal": sum(row["hf_top1_token"] == row["vllm_generated_token"] for row in subset) / max(len(subset), 1),
            "max_common_abs_logprob_delta": max(
                [row["max_common_abs_logprob_delta"] for row in subset if row["max_common_abs_logprob_delta"] is not None],
                default=None,
            ),
        }
    return {
        "kind": "backend_next_token_probe",
        "rows": len(rows),
        "conditions": by_condition,
        "overall_top1_equal_rate": sum(row["top1_equal"] for row in rows) / max(len(rows), 1),
    }


def run(args) -> dict:
    targets = parse_targets(args.targets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["rows.jsonl", "adapters.jsonl", "prompt_contract.json"]:
        path = out / name
        if path.exists():
            path.unlink()
    write_json(out / "args.json", vars(args))
    examples = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    prompt_texts = make_prompts(examples)
    candidate_items = [(key, parse_candidate_key(key)) for key in args.candidate]
    adapter_items = []
    if args.include_zero:
        adapter_items.append(("zero", Candidate(args.zero_family, 0, 0.0, 1)))
    adapter_items.extend(candidate_items)
    adapter_specs = make_adapter_specs(args, out, targets, adapter_items)
    write_jsonl(out / "adapters.jsonl", [asdict(spec) for spec in adapter_specs.values()])

    hf = TransformersLoraBackend(
        args.model,
        rank=args.rank,
        target_suffixes=tuple(targets),
        max_new_tokens=1,
        batch_size=1,
        dtype=args.hf_dtype,
        stop_at_answer=False,
    )
    LLM, SamplingParams, LoRARequest = import_vllm_lora_request()
    contract = backend_contract(hf.tokenizer, prompt_texts, args, SamplingParams)
    write_json(out / "prompt_contract.json", contract)
    load_start = time.time()
    llm = LLM(
        model=args.model,
        dtype=args.vllm_dtype,
        trust_remote_code=True,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=args.max_model_len,
        enable_lora=True,
        max_loras=max(1, len(adapter_specs)),
        max_lora_rank=args.rank,
        max_cpu_loras=max(16, len(adapter_specs)),
        enforce_eager=args.enforce_eager,
    )
    load_s = time.time() - load_start
    contract["vllm_tokenizer"] = vllm_tokenizer_contract(llm, prompt_texts)
    write_json(out / "prompt_contract.json", contract)

    conditions: list[tuple[str, Candidate | None, AdapterSpec | None]] = [("base", None, None)]
    if args.include_zero:
        conditions.append(("zero", Candidate(args.zero_family, 0, 0.0, 1), adapter_specs["zero"]))
    for label, candidate in candidate_items:
        conditions.append((label, candidate, adapter_specs[label]))

    rows = []
    for condition, candidate, spec in conditions:
        if candidate is None:
            hf.clear_candidate()
        else:
            hf.set_candidate(candidate)
        for ex, prompt in zip(examples, prompt_texts):
            left = hf_topk(hf, prompt, args.top_k)
            right, generated_id, generated_text = vllm_topk(llm, SamplingParams, LoRARequest, prompt, spec, args.top_k)
            cmp = compare_topk(left, right)
            row = {
                "condition": condition,
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "hf_top1_token_id": left[0]["token_id"] if left else None,
                "hf_top1_token": left[0]["token"] if left else "",
                "vllm_generated_token_id": generated_id,
                "vllm_generated_token": generated_text,
                "hf_topk": left,
                "vllm_topk": right,
                **cmp,
            }
            rows.append(row)
            write_jsonl(out / "rows.jsonl", [row])
    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "rank": args.rank,
            "targets": targets,
            "top_k": args.top_k,
            "prompts": len(examples),
            "load_s": load_s,
            "candidate_conditions": [label for label, _ in candidate_items],
        }
    )
    write_json(out / "summary.json", summary)
    if not args.keep_adapters:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PEFT and vLLM next-token top-k distributions.")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--include-zero", action="store_true")
    p.add_argument("--zero-family", default="factor_gaussian_lora")
    p.add_argument("--hf-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--vllm-dtype", default="bfloat16")
    p.add_argument("--adapter-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--keep-adapters", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-repeat-data", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidate and not args.include_zero:
        raise ValueError("provide at least one --candidate or --include-zero")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
