from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict
from pathlib import Path
from statistics import mean

from .backend_next_token_probe import make_adapter_specs, parse_candidate_key
from .backends import TransformersLoraBackend
from .countdown import load_examples, prompts as make_prompts, score_completion
from .lora_space import Candidate
from .vllm_lora_bench import AdapterSpec, import_vllm_lora_request, make_sampling_params, parse_targets, write_json, write_jsonl


def vllm_generate(llm, SamplingParams, LoRARequest, prompts: list[str], args, spec: AdapterSpec | None):
    sampling = make_sampling_params(SamplingParams, args.max_new_tokens, args.stop_at_answer)
    request = None if spec is None else LoRARequest(spec.name, spec.lora_int_id, spec.path)
    start = time.time()
    outputs = llm.generate(prompts, sampling, lora_request=request, use_tqdm=False)
    elapsed_s = time.time() - start
    texts = []
    token_counts = []
    token_id_rows = []
    finish_reasons = []
    for output in outputs:
        completion = output.outputs[0]
        token_ids = list(completion.token_ids or [])
        texts.append(completion.text)
        token_counts.append(len(token_ids))
        token_id_rows.append([int(token_id) for token_id in token_ids])
        finish_reasons.append(getattr(completion, "finish_reason", ""))
    return texts, token_counts, token_id_rows, finish_reasons, elapsed_s


def candidate_conditions(args, specs: dict[str, AdapterSpec]) -> list[tuple[str, Candidate | None, AdapterSpec | None]]:
    conditions: list[tuple[str, Candidate | None, AdapterSpec | None]] = [("base", None, None)]
    if args.include_zero:
        conditions.append(("zero", Candidate(args.zero_family, 0, 0.0, 1), specs["zero"]))
    for key in args.candidate:
        conditions.append((key, parse_candidate_key(key), specs[key]))
    return conditions


def first_divergence(left: list[int], right: list[int]) -> int | None:
    for idx, (left_id, right_id) in enumerate(zip(left, right)):
        if left_id != right_id:
            return idx
    if len(left) != len(right):
        return min(len(left), len(right))
    return None


def compare_one(
    condition: str,
    ex,
    hf_text: str,
    vllm_text: str,
    hf_tokens: int,
    vllm_tokens: int,
    hf_token_ids: list[int],
    vllm_token_ids: list[int],
    vllm_finish_reason: str,
    args,
) -> dict:
    hf_score = score_completion(hf_text, ex)
    vllm_score = score_completion(vllm_text, ex)
    hf_cap_hit = float(hf_tokens >= args.max_new_tokens)
    vllm_cap_hit = float(vllm_tokens >= args.max_new_tokens or vllm_finish_reason == "length")
    divergence = first_divergence(hf_token_ids, vllm_token_ids)
    return {
        "condition": condition,
        "example_id": ex.id,
        "numbers": list(ex.numbers),
        "target": ex.target,
        "hf_text": hf_text,
        "vllm_text": vllm_text,
        "text_equal": hf_text == vllm_text,
        "hf_answer": hf_score["answer"],
        "vllm_answer": vllm_score["answer"],
        "answer_equal": hf_score["answer"] == vllm_score["answer"],
        "hf_exact": float(hf_score["exact"]),
        "vllm_exact": float(vllm_score["exact"]),
        "exact_equal": float(hf_score["exact"]) == float(vllm_score["exact"]),
        "hf_malformed": float(hf_score["malformed"]),
        "vllm_malformed": float(vllm_score["malformed"]),
        "malformed_equal": bool(hf_score["malformed"]) == bool(vllm_score["malformed"]),
        "hf_output_tokens": int(hf_tokens),
        "vllm_output_tokens": int(vllm_tokens),
        "hf_token_ids": hf_token_ids,
        "vllm_token_ids": vllm_token_ids,
        "token_ids_equal": hf_token_ids == vllm_token_ids,
        "first_token_divergence": divergence,
        "shared_prefix_tokens": len(hf_token_ids) if divergence is None else divergence,
        "output_token_delta": int(vllm_tokens) - int(hf_tokens),
        "hf_cap_hit": hf_cap_hit,
        "vllm_cap_hit": vllm_cap_hit,
        "cap_hit_equal": hf_cap_hit == vllm_cap_hit,
        "vllm_finish_reason": vllm_finish_reason,
    }


def summarize(rows: list[dict]) -> dict:
    by_condition = {}
    for condition in sorted({row["condition"] for row in rows}):
        subset = [row for row in rows if row["condition"] == condition]
        exact_deltas = [row["vllm_exact"] - row["hf_exact"] for row in subset]
        finite_divergences = [row["first_token_divergence"] for row in subset if row.get("first_token_divergence") is not None]
        by_condition[condition] = {
            "n": len(subset),
            "text_equal_rate": mean(row["text_equal"] for row in subset),
            "answer_equal_rate": mean(row["answer_equal"] for row in subset),
            "exact_equal_rate": mean(row["exact_equal"] for row in subset),
            "hf_exact_mean": mean(row["hf_exact"] for row in subset),
            "vllm_exact_mean": mean(row["vllm_exact"] for row in subset),
            "mean_exact_delta": mean(exact_deltas),
            "max_abs_exact_delta": max(abs(delta) for delta in exact_deltas),
            "hf_malformed_mean": mean(row["hf_malformed"] for row in subset),
            "vllm_malformed_mean": mean(row["vllm_malformed"] for row in subset),
            "hf_cap_hit_mean": mean(row["hf_cap_hit"] for row in subset),
            "vllm_cap_hit_mean": mean(row["vllm_cap_hit"] for row in subset),
            "mean_abs_output_token_delta": mean(abs(row["output_token_delta"]) for row in subset),
            "token_ids_equal_rate": mean(row.get("token_ids_equal", False) for row in subset),
            "first_token_divergence_rate": len(finite_divergences) / max(len(subset), 1),
            "mean_first_token_divergence": mean(finite_divergences) if finite_divergences else None,
        }
    return {
        "kind": "backend_rollout_probe",
        "rows": len(rows),
        "conditions": by_condition,
        "overall_text_equal_rate": mean(row["text_equal"] for row in rows) if rows else None,
        "overall_answer_equal_rate": mean(row["answer_equal"] for row in rows) if rows else None,
        "overall_exact_equal_rate": mean(row["exact_equal"] for row in rows) if rows else None,
    }


def run(args) -> dict:
    targets = parse_targets(args.targets)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["rows.jsonl", "adapters.jsonl"]:
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
        max_new_tokens=args.max_new_tokens,
        batch_size=args.hf_batch_size,
        dtype=args.hf_dtype,
        stop_at_answer=args.stop_at_answer,
    )
    LLM, SamplingParams, LoRARequest = import_vllm_lora_request()
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

    rows = []
    for condition, candidate, spec in candidate_conditions(args, specs):
        if candidate is None:
            hf.clear_candidate()
        else:
            hf.set_candidate(candidate)
        hf_result = hf.generate(prompt_texts)
        vllm_texts, vllm_token_counts, vllm_token_id_rows, vllm_finish_reasons, vllm_elapsed_s = vllm_generate(
            llm,
            SamplingParams,
            LoRARequest,
            prompt_texts,
            args,
            spec,
        )
        for idx, ex in enumerate(examples):
            row = compare_one(
                condition,
                ex,
                hf_result.texts[idx],
                vllm_texts[idx],
                hf_result.token_counts[idx],
                vllm_token_counts[idx],
                hf_result.token_ids[idx] if hf_result.token_ids else [],
                vllm_token_id_rows[idx],
                vllm_finish_reasons[idx],
                args,
            )
            row["hf_elapsed_s"] = hf_result.elapsed_s
            row["vllm_elapsed_s"] = vllm_elapsed_s
            rows.append(row)
            write_jsonl(out / "rows.jsonl", [row])

    summary = summarize(rows)
    summary.update(
        {
            "model": args.model,
            "rank": args.rank,
            "targets": targets,
            "prompts": len(examples),
            "max_new_tokens": args.max_new_tokens,
            "stop_at_answer": args.stop_at_answer,
            "load_s": load_s,
            "candidate_conditions": list(args.candidate),
        }
    )
    write_json(out / "summary.json", summary)
    if not args.keep_adapters:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Compare PEFT and vLLM short greedy rollouts.")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=8)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--candidate", action="append", default=[])
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--include-zero", action="store_true")
    p.add_argument("--zero-family", default="factor_gaussian_lora")
    p.add_argument("--hf-dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument("--hf-batch-size", type=int, default=1)
    p.add_argument("--vllm-dtype", default="bfloat16")
    p.add_argument("--adapter-dtype", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--stop-at-answer", action="store_true")
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
