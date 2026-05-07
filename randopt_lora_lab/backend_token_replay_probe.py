from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .backend_contract import backend_contract, tokenizer_input_ids, vllm_tokenizer_contract
from .backend_next_token_probe import compare_topk, make_adapter_specs, normalize_vllm_logprobs, parse_candidate_key
from .backends import TransformersLoraBackend
from .countdown import load_examples, prompts as make_prompts
from .lora_space import Candidate
from .vllm_lora_bench import AdapterSpec, import_vllm_lora_request, parse_targets, write_json, write_jsonl


def parse_csv(value: str) -> list[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def contains_suffix(values: list[int], suffix: list[int]) -> bool:
    if not suffix or len(values) < len(suffix):
        return False
    return values[-len(suffix) :] == suffix


def candidate_conditions(args, specs: dict[str, AdapterSpec]) -> list[tuple[str, Candidate | None, AdapterSpec | None]]:
    conditions: list[tuple[str, Candidate | None, AdapterSpec | None]] = [("base", None, None)]
    if args.include_zero:
        conditions.append(("zero", Candidate(args.zero_family, 0, 0.0, 1), specs["zero"]))
    for key in args.candidate:
        conditions.append((key, parse_candidate_key(key), specs[key]))
    return conditions


def next_prefix_id(prefix_mode: str, hf_token_id: int | None, vllm_token_id: int | None, top1_equal: bool) -> int | None:
    if prefix_mode == "hf":
        return hf_token_id
    if prefix_mode == "vllm":
        return vllm_token_id
    if prefix_mode == "match":
        return hf_token_id if top1_equal else None
    raise ValueError(f"unknown prefix mode: {prefix_mode}")


def hf_topk_token_ids(backend: TransformersLoraBackend, prompt_token_ids: list[int], top_k: int) -> list[dict]:
    import torch

    device = backend.model.device
    input_ids = torch.tensor([prompt_token_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    out = backend.model(input_ids=input_ids, attention_mask=attention_mask)
    logits = out.logits[0, -1, :].float()
    logprobs = torch.log_softmax(logits, dim=-1)
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


def _vllm_generate_from_token_ids(llm, prompt_token_ids: list[int], sampling, *, lora_request=None):
    try:
        return llm.generate([list(prompt_token_ids)], sampling, lora_request=lora_request, use_tqdm=False)
    except Exception as first_exc:
        try:
            from vllm.inputs import TokensPrompt

            prompt = TokensPrompt(prompt_token_ids=list(prompt_token_ids))
            return llm.generate([prompt], sampling, lora_request=lora_request, use_tqdm=False)
        except Exception:
            raise first_exc


def vllm_topk_token_ids(llm, SamplingParams, LoRARequest, prompt_token_ids: list[int], spec: AdapterSpec | None, top_k: int) -> tuple[list[dict], int | None, str]:
    sampling = SamplingParams(max_tokens=1, temperature=0.0, logprobs=top_k)
    request = None if spec is None else LoRARequest(spec.name, spec.lora_int_id, spec.path)
    outputs = _vllm_generate_from_token_ids(llm, prompt_token_ids, sampling, lora_request=request)
    output = outputs[0].outputs[0]
    token_ids = list(output.token_ids or [])
    generated_id = int(token_ids[0]) if token_ids else None
    generated_text = output.text
    logprobs = output.logprobs[0] if getattr(output, "logprobs", None) else {}
    return normalize_vllm_logprobs(logprobs), generated_id, generated_text


def summarize(rows: list[dict]) -> dict:
    groups = {}
    for row in rows:
        key = (row["condition"], row["prefix_mode"])
        groups.setdefault(key, []).append(row)
    conditions = {}
    for (condition, prefix_mode), subset in sorted(groups.items()):
        first_mismatches = []
        for example_id in sorted({row["example_id"] for row in subset}):
            path_rows = [row for row in subset if row["example_id"] == example_id]
            mismatch_steps = [row["step"] for row in path_rows if not row["top1_equal"]]
            first_mismatches.append(min(mismatch_steps) if mismatch_steps else None)
        finite_mismatches = [x for x in first_mismatches if x is not None]
        max_deltas = [row["max_common_abs_logprob_delta"] for row in subset if row["max_common_abs_logprob_delta"] is not None]
        conditions[f"{condition}|{prefix_mode}"] = {
            "condition": condition,
            "prefix_mode": prefix_mode,
            "rows": len(subset),
            "examples": len({row["example_id"] for row in subset}),
            "top1_equal_rate": mean(row["top1_equal"] for row in subset),
            "mean_topk_overlap": mean(row["topk_overlap"] for row in subset),
            "mean_generated_token_id_equal": mean(row["hf_top1_token_id"] == row["vllm_generated_token_id"] for row in subset),
            "first_mismatch_rate": len(finite_mismatches) / max(len(first_mismatches), 1),
            "mean_first_mismatch_step": mean(finite_mismatches) if finite_mismatches else None,
            "max_common_abs_logprob_delta": max(max_deltas) if max_deltas else None,
        }
    return {
        "kind": "backend_token_replay_probe",
        "rows": len(rows),
        "conditions": conditions,
        "overall_top1_equal_rate": mean(row["top1_equal"] for row in rows) if rows else None,
    }


def run(args) -> dict:
    targets = parse_targets(args.targets)
    prefix_modes = parse_csv(args.prefix_modes)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["rows.jsonl", "adapters.jsonl", "prompt_contract.json"]:
        path = out / name
        if path.exists():
            path.unlink()
    write_json(out / "args.json", vars(args))

    examples = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    prompt_texts = make_prompts(examples)

    adapter_items = []
    if args.include_zero:
        adapter_items.append(("zero", Candidate(args.zero_family, 0, 0.0, 1)))
    adapter_items.extend((key, parse_candidate_key(key)) for key in args.candidate)
    specs = make_adapter_specs(args, out, targets, adapter_items)
    write_jsonl(out / "adapters.jsonl", [asdict(spec) for spec in specs.values()])

    hf = TransformersLoraBackend(
        args.model,
        rank=args.rank,
        target_suffixes=tuple(targets),
        max_new_tokens=1,
        batch_size=1,
        dtype=args.hf_dtype,
        stop_at_answer=False,
    )
    prompt_token_rows = [tokenizer_input_ids(hf.tokenizer, prompt, add_special_tokens=True) for prompt in prompt_texts]
    answer_stop_ids = [int(token_id) for token_id in hf.answer_stop_ids]
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
        max_loras=max(1, len(specs)),
        max_lora_rank=args.rank,
        max_cpu_loras=max(16, len(specs)),
        enforce_eager=args.enforce_eager,
    )
    load_s = time.time() - load_start
    contract["vllm_tokenizer"] = vllm_tokenizer_contract(llm, prompt_texts)
    write_json(out / "prompt_contract.json", contract)

    rows = []
    for condition, candidate, spec in candidate_conditions(args, specs):
        if candidate is None:
            hf.clear_candidate()
        else:
            hf.set_candidate(candidate)
        for prefix_mode in prefix_modes:
            for ex, base_prompt_ids in zip(examples, prompt_token_rows):
                prefix_ids = list(base_prompt_ids)
                generated_ids: list[int] = []
                for step in range(args.max_steps):
                    left = hf_topk_token_ids(hf, prefix_ids, args.top_k)
                    right, generated_id, generated_text = vllm_topk_token_ids(llm, SamplingParams, LoRARequest, prefix_ids, spec, args.top_k)
                    cmp = compare_topk(left, right)
                    hf_token_id = left[0]["token_id"] if left else None
                    row = {
                        "condition": condition,
                        "prefix_mode": prefix_mode,
                        "example_id": ex.id,
                        "numbers": list(ex.numbers),
                        "target": ex.target,
                        "step": step,
                        "prefix_token_count": len(prefix_ids),
                        "prefix_token_ids_tail": prefix_ids[-args.prefix_tail_tokens :],
                        "generated_token_ids_so_far": list(generated_ids),
                        "hf_top1_token_id": hf_token_id,
                        "hf_top1_token": left[0]["token"] if left else "",
                        "vllm_generated_token_id": generated_id,
                        "vllm_generated_token": generated_text,
                        "hf_topk": left,
                        "vllm_topk": right,
                        **cmp,
                    }
                    rows.append(row)
                    write_jsonl(out / "rows.jsonl", [row])
                    token_id = next_prefix_id(prefix_mode, hf_token_id, generated_id, bool(cmp["top1_equal"]))
                    if token_id is None:
                        break
                    prefix_ids.append(int(token_id))
                    generated_ids.append(int(token_id))
                    if args.stop_at_answer and contains_suffix(generated_ids, answer_stop_ids):
                        break

    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "rank": args.rank,
            "targets": targets,
            "top_k": args.top_k,
            "prompts": len(examples),
            "max_steps": args.max_steps,
            "prefix_modes": prefix_modes,
            "prefix_input": "token_ids",
            "load_s": load_s,
            "candidate_conditions": list(args.candidate),
            "include_zero": args.include_zero,
        }
    )
    write_json(out / "summary.json", summary)
    if not args.keep_adapters:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PEFT and vLLM next-token parity along shared token-ID prefixes.")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=4)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--max-steps", type=int, default=16)
    p.add_argument("--prefix-modes", default="hf")
    p.add_argument("--include-zero", action="store_true")
    p.add_argument("--zero-family", default="factor_gaussian_lora")
    p.add_argument("--hf-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--vllm-dtype", default="bfloat16")
    p.add_argument("--adapter-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--stop-at-answer", action="store_true")
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--keep-adapters", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-repeat-data", action="store_true")
    p.add_argument("--prefix-tail-tokens", type=int, default=32)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if not args.candidate and not args.include_zero:
        raise ValueError("provide at least one --candidate or --include-zero")
    run(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
