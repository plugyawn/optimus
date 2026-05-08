from __future__ import annotations

import argparse
import gc
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

import numpy as np

from .backend_contract import backend_contract, vllm_tokenizer_contract
from .countdown import load_examples, unique_example_count, unique_semantic_example_count
from .experiments import anzo_anchor_prompts, ensemble_ks_from_values, majority_vote_evaluation, parse_float_list, parse_ratio_list
from .prompt_variants import make_variant_prompts
from .selection_score import (
    combine_candidate_conditions,
    enrich_condition_rows,
    filter_condition_rows_by_variants,
    parse_prompt_variants,
    protocol_valid_variants,
)
from .vllm_lora_bench import (
    AdapterSpec,
    Candidate,
    import_vllm_lora_request,
    make_sampling_params,
    parse_targets,
    qwen_lora_shapes,
    save_seed_adapter,
    score_mixed_rows,
    score_rows,
    write_json,
    write_jsonl,
)
from .vllm_prompting import make_vllm_prompt_inputs


def candidate_panel(
    family: str,
    population: int,
    sigma: float,
    seed: int,
    antithetic: bool,
    sigma_values: list[float] | None = None,
) -> list[Candidate]:
    rng = np.random.default_rng(seed)
    sigmas = sigma_values or [sigma]
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=population if not antithetic else population // 2)]
    sampled_sigmas = [float(x) for x in rng.choice(sigmas, size=len(seeds), replace=True)]
    out = []
    for candidate_seed, sampled_sigma in zip(seeds, sampled_sigmas):
        out.append(Candidate(family, candidate_seed, sampled_sigma, 1))
        if antithetic:
            out.append(Candidate(family, candidate_seed, sampled_sigma, -1))
    return out[:population]


def parse_candidate_key(key: str) -> Candidate:
    parts = key.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid candidate key: {key}")
    return Candidate(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


def read_candidate_file(path: str) -> list[Candidate]:
    candidates = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = line
            key = item.get("candidate") if isinstance(item, dict) else str(item)
            if not key:
                raise ValueError(f"{path}:{line_no} missing candidate")
            candidates.append(parse_candidate_key(str(key)))
    return candidates


def safe_name(candidate: Candidate) -> str:
    sign = "pos" if candidate.sign > 0 else "neg"
    return f"randopt_seed{candidate.seed}_s{candidate.sigma:g}_{sign}"


def make_adapter_specs(
    args,
    out: Path,
    targets: list[str],
    candidates: list[Candidate],
    *,
    family_state: dict | None = None,
) -> list[AdapterSpec]:
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained(
        args.model,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model_type = str(getattr(config, "model_type", ""))
    if not model_type.startswith("qwen2"):
        raise ValueError(f"{args.model} has model_type={model_type!r}; direct adapter generation is only validated for Qwen2.")

    # Force shape validation before spending time writing many adapter files.
    qwen_lora_shapes(config, targets)
    adapter_root = Path(args.adapter_dir) if args.adapter_dir else out / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)
    specs = []
    for idx, candidate in enumerate(candidates):
        name = safe_name(candidate)
        path = adapter_root / f"{idx:05d}_{name}"
        save_seed_adapter(
            path,
            model=args.model,
            candidate=candidate,
            rank=args.rank,
            targets=targets,
            config=config,
            tensor_dtype=args.adapter_dtype,
            family_state=family_state,
        )
        specs.append(
            AdapterSpec(
                index=idx,
                name=name,
                lora_int_id=idx + 1,
                path=str(path.resolve()),
                candidate=candidate.key,
                seed=candidate.seed,
                sigma=candidate.sigma,
                sign=candidate.sign,
            )
        )
    return specs


def build_activation_family_state(args, out: Path, screen, prompt_variants: list[str]) -> dict | None:
    if not (
        args.family.startswith("activation_spectral_lora")
        or args.family.startswith("activation_projected_gaussian_rank_r")
        or args.family.startswith("activation_generalized_projected_gaussian_rank_r")
        or args.family.startswith("activation_generalized_spectral_lora")
    ):
        return None

    import torch

    from .backends import TransformersLoraBackend

    state_start = time.time()
    backend = TransformersLoraBackend(
        args.model,
        rank=args.rank,
        target_suffixes=tuple(args.targets.split(",")),
        max_new_tokens=args.max_new_tokens,
        batch_size=args.activation_state_batch_size,
        dtype="bf16" if args.dtype in {"bfloat16", "bf16"} else "fp16",
        stop_at_answer=args.stop_at_answer,
    )
    target_examples = screen[: min(args.activation_state_prompts, len(screen))]
    target_prompts = []
    for variant in prompt_variants:
        target_prompts.extend(
            make_variant_prompts(
                target_examples,
                variant,
                tokenizer=backend.tokenizer,
                use_chat_template=args.use_chat_template,
            )
        )
    if args.family.startswith("activation_spectral_lora_sv"):
        build_state = backend.build_activation_spectral_state
    elif args.family.startswith("activation_generalized_projected_gaussian_rank_r") or args.family.startswith(
        "activation_generalized_spectral_lora"
    ):
        build_state = backend.build_activation_generalized_state
    else:
        build_state = backend.build_anzo_state
    family_state = build_state(
        target_prompts,
        anzo_anchor_prompts(),
        subtract_anchor=not args.activation_state_no_anchor_subtract,
    )
    state_s = time.time() - state_start
    torch.save(family_state, out / "family_state.pt")
    write_json(
        out / "family_state_summary.json",
        {
            "kind": "activation_spectral_family_state",
            "family": args.family,
            "activation_state_prompts": len(target_examples),
            "activation_state_prompt_variants": prompt_variants,
            "activation_state_target_prompt_count": len(target_prompts),
            "activation_state_anchor_prompt_count": len(anzo_anchor_prompts()),
            "activation_state_no_anchor_subtract": args.activation_state_no_anchor_subtract,
            "modules": len(family_state),
            "rank": args.rank,
            "elapsed_s": state_s,
        },
    )
    backend.clear_candidate()
    del backend
    gc.collect()
    torch.cuda.empty_cache()
    return family_state


def reset_outputs(out: Path) -> None:
    for name in [
        "adapters.jsonl",
        "candidate_summary.jsonl",
        "candidate_condition_summary.jsonl",
        "holdout_candidate_summary.jsonl",
        "holdout_candidate_condition_summary.jsonl",
        "per_prompt.jsonl",
        "holdout_per_prompt.jsonl",
        "ensemble_per_prompt.jsonl",
    ]:
        path = out / name
        if path.exists():
            path.unlink()


def selection_variants_or_raise(base_by_variant: dict[str, dict], args, *, split: str) -> list[str]:
    valid = protocol_valid_variants(
        base_by_variant,
        max_malformed=args.max_base_malformed_for_selection,
        max_cap_hit=args.max_base_cap_hit_for_selection,
    )
    if len(valid) < args.min_selection_prompt_variants:
        invalid = sorted(set(base_by_variant) - set(valid))
        raise RuntimeError(
            f"{split} has only {len(valid)} base-valid prompt variants {valid}; "
            f"need at least {args.min_selection_prompt_variants}. "
            f"Stress/invalid variants: {invalid}."
        )
    return valid


def mixed_eval(
    llm,
    LoRARequest,
    sampling,
    examples,
    specs: list[AdapterSpec],
    args,
    *,
    mode: str,
    prompt_variant: str = "default",
) -> tuple[list[dict], list[dict], dict]:
    prompt_texts = make_variant_prompts(
        examples,
        prompt_variant,
        tokenizer=llm.get_tokenizer(),
        use_chat_template=args.use_chat_template,
    )
    prompt_inputs = make_vllm_prompt_inputs(prompt_texts, llm.get_tokenizer(), args.prompt_input)
    per_prompt_rows = []
    candidate_rows = []
    total_elapsed = 0.0
    total_tokens = 0
    chunk_size = max(1, min(args.chunk_adapters, args.max_loras))
    for chunk_start in range(0, len(specs), chunk_size):
        chunk = specs[chunk_start : chunk_start + chunk_size]
        prompts = []
        requests = []
        for spec in chunk:
            req = LoRARequest(spec.name, spec.lora_int_id, spec.path)
            prompts.extend(prompt_inputs)
            requests.extend([req] * len(prompt_texts))
        start = time.time()
        outputs = llm.generate(prompts, sampling, lora_request=requests, use_tqdm=False)
        elapsed = time.time() - start
        rows, metrics = score_mixed_rows(
            examples=examples,
            outputs=outputs,
            specs=chunk,
            prompts_per_adapter=len(prompt_texts),
            max_new_tokens=args.max_new_tokens,
        )
        per_prompt_rows.extend(dict(row, mode=mode, prompt_variant=prompt_variant) for row in rows)
        total_elapsed += elapsed
        total_tokens += int(metrics["output_tokens"])
        by_candidate = metrics.pop("by_candidate")
        for spec in chunk:
            row = dict(by_candidate[spec.candidate])
            row.update(
                {
                    "candidate": spec.candidate,
                    "adapter_index": spec.index,
                    "adapter": spec.name,
                    "mode": mode,
                    "prompt_variant": prompt_variant,
                    "seed": spec.seed,
                    "sigma": spec.sigma,
                    "sign": spec.sign,
                    "elapsed_s": elapsed / max(len(chunk), 1),
                    "prompts": len(prompt_texts),
                }
            )
            candidate_rows.append(row)
    aggregate = {
        "elapsed_s": total_elapsed,
        "output_tokens": total_tokens,
        "tokens_per_sec": total_tokens / max(total_elapsed, 1e-9),
        "prompts_per_sec": (len(specs) * len(prompt_texts)) / max(total_elapsed, 1e-9),
        "candidate_sec": len(specs) / max(total_elapsed, 1e-9),
    }
    return per_prompt_rows, candidate_rows, aggregate


def base_eval(llm, sampling, examples, args, *, mode: str, prompt_variant: str = "default") -> tuple[list[dict], dict]:
    prompt_texts = make_variant_prompts(
        examples,
        prompt_variant,
        tokenizer=llm.get_tokenizer(),
        use_chat_template=args.use_chat_template,
    )
    prompt_inputs = make_vllm_prompt_inputs(prompt_texts, llm.get_tokenizer(), args.prompt_input)
    start = time.time()
    outputs = llm.generate(prompt_inputs, sampling, use_tqdm=False)
    elapsed = time.time() - start
    rows, metrics = score_rows(
        mode=mode,
        candidate="base",
        examples=examples,
        outputs=outputs,
        max_new_tokens=args.max_new_tokens,
    )
    rows = [dict(row, prompt_variant=prompt_variant) for row in rows]
    metrics["prompt_variant"] = prompt_variant
    metrics["elapsed_s"] = elapsed
    metrics["tokens_per_sec"] = metrics["output_tokens"] / max(elapsed, 1e-9)
    metrics["prompts_per_sec"] = len(prompt_texts) / max(elapsed, 1e-9)
    return rows, metrics


def write_prompt_contracts(out: Path, llm, SamplingParams, args, screen, holdout, prompt_variants: list[str]) -> None:
    tokenizer = llm.get_tokenizer()
    contracts = {}
    for split, examples in [("screen", screen), ("holdout", holdout)]:
        for variant in prompt_variants:
            prompt_texts = make_variant_prompts(
                examples,
                variant,
                tokenizer=tokenizer,
                use_chat_template=args.use_chat_template,
            )
            contract = backend_contract(tokenizer, prompt_texts, args, SamplingParams)
            contract["vllm_tokenizer"] = vllm_tokenizer_contract(llm, prompt_texts)
            contracts[f"{split}:{variant}"] = contract
    write_json(
        out / "prompt_contract.json",
        {
            "kind": "vllm_lora_search_prompt_contracts",
            "model": args.model,
            "dtype": args.dtype,
            "prompt_input": args.prompt_input,
            "prompt_variants": prompt_variants,
            "contracts": contracts,
        },
    )


def run_search(args) -> dict:
    targets = parse_targets(args.targets)
    prompt_variants = parse_prompt_variants(args.prompt_variants)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out)
    write_json(out / "args.json", vars(args))

    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    sigma_values = parse_float_list(args.sigma_values) if args.sigma_values else [args.sigma]
    if args.candidate_file:
        candidates = read_candidate_file(args.candidate_file)
    else:
        candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic, sigma_values)

    family_state = build_activation_family_state(args, out, screen, prompt_variants)
    adapter_start = time.time()
    specs = make_adapter_specs(args, out, targets, candidates, family_state=family_state)
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
    write_prompt_contracts(out, llm, SamplingParams, args, screen, holdout, prompt_variants)

    base_screen_rows = []
    base_holdout_rows = []
    base_screen_by_variant = {}
    base_holdout_by_variant = {}
    for variant in prompt_variants:
        rows, metrics = base_eval(llm, sampling, screen, args, mode="base_screen", prompt_variant=variant)
        base_screen_rows.extend(rows)
        base_screen_by_variant[variant] = metrics
        rows, metrics = base_eval(llm, sampling, holdout, args, mode="base_holdout", prompt_variant=variant)
        base_holdout_rows.extend(rows)
        base_holdout_by_variant[variant] = metrics
    screen_selection_variants = selection_variants_or_raise(base_screen_by_variant, args, split="screen")
    holdout_selection_variants = selection_variants_or_raise(base_holdout_by_variant, args, split="holdout")
    screen_stress_variants = sorted(set(prompt_variants) - set(screen_selection_variants))
    holdout_stress_variants = sorted(set(prompt_variants) - set(holdout_selection_variants))
    write_jsonl(out / "per_prompt.jsonl", base_screen_rows)
    write_jsonl(out / "holdout_per_prompt.jsonl", base_holdout_rows)

    screen_rows = []
    screen_condition_rows = []
    screen_aggregate = {"elapsed_s": 0.0, "output_tokens": 0}
    for variant in prompt_variants:
        rows, condition_rows, aggregate = mixed_eval(
            llm,
            LoRARequest,
            sampling,
            screen,
            specs,
            args,
            mode="screen",
            prompt_variant=variant,
        )
        screen_rows.extend(rows)
        screen_condition_rows.extend(condition_rows)
        screen_aggregate["elapsed_s"] += aggregate["elapsed_s"]
        screen_aggregate["output_tokens"] += aggregate["output_tokens"]
    write_jsonl(out / "per_prompt.jsonl", screen_rows)
    screen_selection_condition_rows = filter_condition_rows_by_variants(
        screen_condition_rows,
        screen_selection_variants,
    )
    screen_condition_rows = enrich_condition_rows(
        screen_condition_rows,
        base_screen_by_variant,
        malformed_penalty=args.malformed_penalty,
        cap_hit_penalty=args.cap_hit_penalty,
    )
    screen_selection_condition_rows = enrich_condition_rows(
        screen_selection_condition_rows,
        base_screen_by_variant,
        malformed_penalty=args.malformed_penalty,
        cap_hit_penalty=args.cap_hit_penalty,
    )
    candidate_rows = combine_candidate_conditions(
        screen_selection_condition_rows,
        base_screen_by_variant,
        score_mode=args.score_mode,
        malformed_penalty=args.malformed_penalty,
        cap_hit_penalty=args.cap_hit_penalty,
    )
    write_jsonl(out / "candidate_condition_summary.jsonl", screen_condition_rows)
    write_jsonl(out / "candidate_summary.jsonl", candidate_rows)

    ensemble_ks = ensemble_ks_from_values(len(specs), args.ensemble_ks, args.ensemble_ratios)
    promote_n = max(args.promote, max(ensemble_ks, default=0))
    top = sorted(candidate_rows, key=lambda r: r["selection_score"], reverse=True)[: min(promote_n, len(candidate_rows))]
    top_specs = {spec.candidate: spec for spec in specs}
    holdout_specs = [top_specs[row["candidate"]] for row in top]
    holdout_rows = []
    holdout_condition_rows = []
    holdout_aggregate = {"elapsed_s": 0.0, "output_tokens": 0}
    for variant in prompt_variants:
        rows, condition_rows, aggregate = mixed_eval(
            llm,
            LoRARequest,
            sampling,
            holdout,
            holdout_specs,
            args,
            mode="holdout",
            prompt_variant=variant,
        )
        holdout_rows.extend(rows)
        holdout_condition_rows.extend(condition_rows)
        holdout_aggregate["elapsed_s"] += aggregate["elapsed_s"]
        holdout_aggregate["output_tokens"] += aggregate["output_tokens"]
    holdout_selection_condition_rows = filter_condition_rows_by_variants(
        holdout_condition_rows,
        holdout_selection_variants,
    )
    holdout_condition_rows = enrich_condition_rows(
        holdout_condition_rows,
        base_holdout_by_variant,
        malformed_penalty=args.malformed_penalty,
        cap_hit_penalty=args.cap_hit_penalty,
    )
    holdout_selection_condition_rows = enrich_condition_rows(
        holdout_selection_condition_rows,
        base_holdout_by_variant,
        malformed_penalty=args.malformed_penalty,
        cap_hit_penalty=args.cap_hit_penalty,
    )
    holdout_candidate_rows = combine_candidate_conditions(
        holdout_selection_condition_rows,
        base_holdout_by_variant,
        score_mode=args.score_mode,
        malformed_penalty=args.malformed_penalty,
        cap_hit_penalty=args.cap_hit_penalty,
    )
    write_jsonl(out / "holdout_per_prompt.jsonl", holdout_rows)
    write_jsonl(out / "holdout_candidate_condition_summary.jsonl", holdout_condition_rows)
    write_jsonl(out / "holdout_candidate_summary.jsonl", holdout_candidate_rows)

    by_candidate_holdout = {row["candidate"]: row for row in holdout_candidate_rows}
    top_holdout = [by_candidate_holdout[row["candidate"]] for row in top if row["candidate"] in by_candidate_holdout]
    ensemble_holdout = []
    strict_ensemble_holdout = []
    if ensemble_ks:
        by_k: dict[int, list[dict]] = {k: [] for k in ensemble_ks}
        strict_by_k: dict[int, list[dict]] = {k: [] for k in ensemble_ks}
        ensemble_per_prompt = []
        candidate_order = [str(row["candidate"]) for row in top]
        for variant in holdout_selection_variants:
            variant_rows = [row for row in holdout_rows if str(row.get("prompt_variant", "default")) == variant]
            variant_summary, variant_per_prompt = majority_vote_evaluation(candidate_order, variant_rows, holdout, ensemble_ks)
            strict_variant_summary, strict_variant_per_prompt = majority_vote_evaluation(
                candidate_order,
                variant_rows,
                holdout,
                ensemble_ks,
                strict_rows=True,
            )
            for row in variant_summary:
                by_k[int(row["k"])].append(dict(row, prompt_variant=variant))
            for row in strict_variant_summary:
                strict_by_k[int(row["k"])].append(dict(row, prompt_variant=variant))
            ensemble_per_prompt.extend(dict(row, prompt_variant=variant, vote_filter="numeric") for row in variant_per_prompt)
            ensemble_per_prompt.extend(dict(row, prompt_variant=variant, vote_filter="strict_numeric") for row in strict_variant_per_prompt)
        for k in ensemble_ks:
            rows = by_k[k]
            ensemble_holdout.append(
                {
                    "k": k,
                    "prompt_variants": sorted(str(row["prompt_variant"]) for row in rows),
                    "condition_count": len(rows),
                    "exact_mean": float(np.mean([float(row["exact_mean"]) for row in rows])) if rows else 0.0,
                    "min_exact_mean": min((float(row["exact_mean"]) for row in rows), default=0.0),
                    "coverage_mean": float(np.mean([float(row["coverage_mean"]) for row in rows])) if rows else 0.0,
                    "valid_votes_per_prompt": float(np.mean([float(row["valid_votes_per_prompt"]) for row in rows])) if rows else 0.0,
                    "conditions": rows,
                }
            )
            strict_rows = strict_by_k[k]
            strict_ensemble_holdout.append(
                {
                    "k": k,
                    "prompt_variants": sorted(str(row["prompt_variant"]) for row in strict_rows),
                    "condition_count": len(strict_rows),
                    "exact_mean": float(np.mean([float(row["exact_mean"]) for row in strict_rows])) if strict_rows else 0.0,
                    "min_exact_mean": min((float(row["exact_mean"]) for row in strict_rows), default=0.0),
                    "coverage_mean": float(np.mean([float(row["coverage_mean"]) for row in strict_rows])) if strict_rows else 0.0,
                    "valid_votes_per_prompt": float(np.mean([float(row["valid_votes_per_prompt"]) for row in strict_rows])) if strict_rows else 0.0,
                    "conditions": strict_rows,
                }
            )
        write_jsonl(out / "ensemble_per_prompt.jsonl", ensemble_per_prompt)
    total_eval_s = screen_aggregate["elapsed_s"] + holdout_aggregate["elapsed_s"]
    screen_tokens_per_sec = screen_aggregate["output_tokens"] / max(screen_aggregate["elapsed_s"], 1e-9)
    screen_prompts_per_sec = (len(specs) * len(screen) * len(prompt_variants)) / max(screen_aggregate["elapsed_s"], 1e-9)
    screen_candidate_sec = len(specs) / max(screen_aggregate["elapsed_s"], 1e-9)
    holdout_tokens_per_sec = holdout_aggregate["output_tokens"] / max(holdout_aggregate["elapsed_s"], 1e-9)
    adapters_kept = bool(args.keep_adapters or args.adapter_dir)
    if not adapters_kept:
        shutil.rmtree(out / "adapters", ignore_errors=True)
    summary = {
        "kind": "vllm_lora_search",
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "candidate_file": args.candidate_file,
        "candidate_families": sorted({candidate.family for candidate in candidates}),
        "family_state": "family_state.pt" if family_state is not None else None,
        "population": len(specs),
        "rank": args.rank,
        "sigma": args.sigma,
        "sigma_values": sigma_values,
        "seed": args.seed,
        "targets": targets,
        "antithetic": args.antithetic,
        "prompt_variants": prompt_variants,
        "prompt_input": args.prompt_input,
        "use_chat_template": args.use_chat_template,
        "screen_selection_prompt_variants": screen_selection_variants,
        "screen_stress_prompt_variants": screen_stress_variants,
        "holdout_selection_prompt_variants": holdout_selection_variants,
        "holdout_stress_prompt_variants": holdout_stress_variants,
        "candidate_score_metric": "exact_answer",
        "ensemble_vote_metric": "valid_numeric_majority_vote",
        "strict_ensemble_vote_metric": "strict_parser_then_valid_numeric_majority_vote",
        "score_mode": args.score_mode,
        "malformed_penalty": args.malformed_penalty,
        "cap_hit_penalty": args.cap_hit_penalty,
        "max_base_malformed_for_selection": args.max_base_malformed_for_selection,
        "max_base_cap_hit_for_selection": args.max_base_cap_hit_for_selection,
        "min_selection_prompt_variants": args.min_selection_prompt_variants,
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "screen_unique_prompts": unique_example_count(screen),
        "holdout_unique_prompts": unique_example_count(holdout),
        "screen_unique_semantic_prompts": unique_semantic_example_count(screen),
        "holdout_unique_semantic_prompts": unique_semantic_example_count(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "promote": args.promote,
        "ensemble_ks": ensemble_ks,
        "ensemble_ratios": parse_ratio_list(args.ensemble_ratios) if args.ensemble_ratios else [],
        "max_loras": args.max_loras,
        "chunk_adapters": args.chunk_adapters,
        "enforce_eager": args.enforce_eager,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "max_new_tokens": args.max_new_tokens,
        "stop_at_answer": args.stop_at_answer,
        "dtype": args.dtype,
        "adapter_dtype": args.adapter_dtype,
        "allow_repeat_data": args.allow_repeat_data,
        "adapter_build_s": adapter_build_s,
        "adapters_kept": adapters_kept,
        "load_s": load_s,
        "base_screen_exact": sum(row["exact_mean"] for row in base_screen_by_variant.values()) / len(base_screen_by_variant),
        "base_holdout_exact": sum(row["exact_mean"] for row in base_holdout_by_variant.values()) / len(base_holdout_by_variant),
        "base_screen_by_prompt": base_screen_by_variant,
        "base_holdout_by_prompt": base_holdout_by_variant,
        "screen_tokens_per_sec": screen_tokens_per_sec,
        "screen_prompts_per_sec": screen_prompts_per_sec,
        "screen_candidate_sec": screen_candidate_sec,
        "holdout_tokens_per_sec": holdout_tokens_per_sec,
        "eval_elapsed_s": total_eval_s,
        "candidate_sec": len(specs) / max(total_eval_s, 1e-9),
        "best_tokens_per_sec": screen_tokens_per_sec,
        "best_prompts_per_sec": screen_prompts_per_sec,
        "top_screen": top,
        "top_holdout": top_holdout,
        "ensemble_holdout": ensemble_holdout,
        "best_ensemble_holdout_exact": max((row["exact_mean"] for row in ensemble_holdout), default=None),
        "strict_ensemble_holdout": strict_ensemble_holdout,
        "best_strict_ensemble_holdout_exact": max((row["exact_mean"] for row in strict_ensemble_holdout), default=None),
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
        "kind": "vllm_lora_search_failure",
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
    p = argparse.ArgumentParser(description="Run mixed-batch vLLM LoRA RandOpt search.")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=32)
    p.add_argument("--holdout-prompts", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--population", type=int, default=128)
    p.add_argument("--promote", type=int, default=8)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--sigma-values", default="")
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--ensemble-ks", default="")
    p.add_argument("--ensemble-ratios", default="")
    p.add_argument("--prompt-variants", default="default")
    p.add_argument("--prompt-input", default="text", choices=["text", "token_ids"])
    p.add_argument("--use-chat-template", action="store_true")
    p.add_argument("--score-mode", default="exact", choices=["exact", "robust_mean", "robust_min"])
    p.add_argument("--malformed-penalty", type=float, default=1.0)
    p.add_argument("--cap-hit-penalty", type=float, default=1.0)
    p.add_argument("--max-base-malformed-for-selection", type=float, default=0.05)
    p.add_argument("--max-base-cap-hit-for-selection", type=float, default=0.05)
    p.add_argument("--min-selection-prompt-variants", type=int, default=1)
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
            "activation_spectral_lora",
            "activation_spectral_lora_c0p5",
            "activation_spectral_lora_c0p75",
            "activation_spectral_lora_c1p25",
            "activation_spectral_lora_c1p5",
            "activation_spectral_lora_c2",
            "activation_spectral_lora_c3",
            "activation_spectral_lora_c4",
            "activation_spectral_lora_tscale_q2_v1",
            "activation_spectral_lora_tscale_q2_v1p045",
            "activation_spectral_lora_tscale_q2_k1_v1_o2",
            "activation_spectral_lora_tscale_q2_k1p045_v1p045_o2",
            "activation_spectral_lora_sv",
            "activation_spectral_lora_sv_c0p75",
            "activation_spectral_lora_sv_c1p25",
            "activation_spectral_lora_sv_c1p5",
            "activation_spectral_lora_sv_c2",
            "sparse_low_rank_lora",
            "sparse_low_rank_lora_d0p125",
            "sparse_low_rank_lora_d0p25",
            "sparse_low_rank_lora_d0p5",
        ],
    )
    p.add_argument("--candidate-file", default="", help="Optional JSONL/list of exact candidate keys to evaluate.")
    p.add_argument("--antithetic", action="store_true")
    p.add_argument("--max-new-tokens", type=int, default=32)
    p.add_argument("--stop-at-answer", action="store_true")
    p.add_argument("--dtype", default="bfloat16")
    p.add_argument("--adapter-dtype", default="bfloat16", choices=["float16", "bfloat16", "float32"])
    p.add_argument("--gpu-memory-utilization", type=float, default=0.82)
    p.add_argument("--max-model-len", type=int, default=1024)
    p.add_argument("--max-num-batched-tokens", type=int, default=0)
    p.add_argument("--enforce-eager", action="store_true")
    p.add_argument("--max-loras", type=int, default=32)
    p.add_argument("--max-cpu-loras", type=int, default=1024)
    p.add_argument("--chunk-adapters", type=int, default=32)
    p.add_argument("--adapter-dir", default=None)
    p.add_argument("--keep-adapters", action="store_true")
    p.add_argument("--activation-state-prompts", type=int, default=16)
    p.add_argument("--activation-state-batch-size", type=int, default=16)
    p.add_argument("--activation-state-no-anchor-subtract", action="store_true")
    p.add_argument("--local-files-only", action="store_true")
    p.add_argument("--allow-repeat-data", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    try:
        run_search(args)
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
