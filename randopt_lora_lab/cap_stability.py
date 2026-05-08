from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from .aggregate_lora import build_aggregate_state, top_candidate_rows
from .countdown import CountdownExample, load_examples, score_completion
from .experiments import anzo_anchor_prompts, make_backend, parse_candidate_key, reset_outputs, tag_rows, write_jsonl
from .lora_space import Candidate
from .prompt_variants import (
    PromptFn,
    compact_tagged_prompt,
    direct_tagged_prompt,
    prompt_fn,
    reordered_tagged_prompt,
    tight_tagged_prompt,
    xml_tagged_prompt,
)


def infer_activation_state_prompt_variants(source_run: Path, override: str) -> list[str]:
    if override:
        return [item.strip() for item in override.split(",") if item.strip()]
    family_state_summary = source_run / "family_state_summary.json"
    if family_state_summary.exists():
        summary = json.loads(family_state_summary.read_text())
        variants = summary.get("activation_state_prompt_variants")
        if variants:
            return [str(item) for item in variants]
    summary_path = source_run / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        variants = summary.get("activation_state_prompt_variants") or summary.get("prompt_variants")
        if variants:
            return [str(item) for item in variants]
        variant = summary.get("prompt_variant")
        if variant:
            return [str(variant)]
    return ["default"]


def build_family_states(
    backend,
    candidates: list[Candidate],
    screen: list[CountdownExample],
    *,
    source_run: Path,
    activation_state_prompt_variants: list[str],
) -> tuple[dict[str, dict], dict]:
    states = {}
    saved_state = source_run / "family_state.pt"
    activation_families = {
        candidate.family
        for candidate in candidates
        if candidate.family.startswith("activation_spectral_lora")
        or candidate.family.startswith("activation_projected_gaussian_rank_r")
        or candidate.family.startswith("activation_generalized_projected_gaussian_rank_r")
    }
    if saved_state.exists() and activation_families:
        loaded = torch.load(saved_state, map_location="cpu")
        return {family: loaded for family in activation_families}, {
            "activation_state_source": str(saved_state),
            "activation_state_prompt_variants": activation_state_prompt_variants,
            "activation_state_modules": len(loaded),
        }
    target_examples = screen[: min(16, len(screen))]
    target_prompts = [prompt_fn(variant)(ex) for variant in activation_state_prompt_variants for ex in target_examples]
    for family in sorted({candidate.family for candidate in candidates}):
        if family.startswith("activation_spectral_lora_sv"):
            states[family] = backend.build_activation_spectral_state(target_prompts, anzo_anchor_prompts())
        elif family.startswith("activation_generalized_projected_gaussian_rank_r"):
            states[family] = backend.build_activation_generalized_state(target_prompts, anzo_anchor_prompts())
        elif family.startswith("activation_spectral_lora") or family.startswith("activation_projected_gaussian_rank_r"):
            states[family] = backend.build_anzo_state(target_prompts, anzo_anchor_prompts())
    return states, {
        "activation_state_source": "rebuilt",
        "activation_state_prompt_variants": activation_state_prompt_variants,
        "activation_state_target_prompt_count": len(target_prompts),
        "activation_state_modules": len(next(iter(states.values()), {})),
    }


def evaluate_with_prompt_fn(
    backend,
    candidate: Candidate | None,
    examples: list[CountdownExample],
    make_prompt: PromptFn,
    *,
    family_state: dict | None = None,
    label: str | None = None,
) -> dict:
    prompts = [make_prompt(ex) for ex in examples]
    mutation_start = time.time()
    if candidate is None:
        backend.clear_candidate()
        key = label or "base"
    else:
        backend.set_candidate(candidate, family_state)
        key = label or candidate.key
    mutation_s = time.time() - mutation_start
    result = backend.generate(prompts)
    rows = []
    exact = []
    malformed = []
    cap_hits = []
    answer_closed = []
    output_token_counts = []
    raw_token_counts = result.raw_token_counts or result.token_counts
    for ex, text, output_tokens, raw_output_tokens in zip(examples, result.texts, result.token_counts, raw_token_counts):
        score = score_completion(text, ex)
        cap_hit = float(raw_output_tokens >= backend.max_new_tokens)
        closed = float("</answer>" in text)
        exact.append(score["exact"])
        malformed.append(float(score["malformed"]))
        cap_hits.append(cap_hit)
        answer_closed.append(closed)
        output_token_counts.append(int(output_tokens))
        rows.append(
            {
                "candidate": key,
                "example_id": ex.id,
                "numbers": list(ex.numbers),
                "target": ex.target,
                "text": text,
                "output_tokens": int(output_tokens),
                "raw_output_tokens": int(raw_output_tokens),
                "hidden_after_answer_tokens": max(int(raw_output_tokens) - int(output_tokens), 0),
                "cap_hit": cap_hit,
                "answer_closed": closed,
                **score,
            }
        )
    return {
        "candidate": key,
        "exact_mean": float(np.mean(exact)),
        "malformed_mean": float(np.mean(malformed)),
        "cap_hit_mean": float(np.mean(cap_hits)),
        "answer_closed_mean": float(np.mean(answer_closed)),
        "output_tokens": int(result.output_tokens),
        "raw_output_tokens": int(result.raw_output_tokens if result.raw_output_tokens is not None else result.output_tokens),
        "output_token_mean": float(np.mean(output_token_counts)),
        "output_token_p95": float(np.quantile(output_token_counts, 0.95)),
        "elapsed_s": float(result.elapsed_s),
        "mutation_s": float(mutation_s),
        "rows": rows,
    }


def metric_row(ev: dict, *, cap: int, prompt_variant: str, split: str, candidate_kind: str) -> dict:
    return {
        key: value
        for key, value in {
            "candidate": ev["candidate"],
            "candidate_kind": candidate_kind,
            "split": split,
            "prompt_variant": prompt_variant,
            "max_new_tokens": cap,
            "exact_mean": ev["exact_mean"],
            "malformed_mean": ev["malformed_mean"],
            "cap_hit_mean": ev["cap_hit_mean"],
            "answer_closed_mean": ev["answer_closed_mean"],
            "output_tokens": ev["output_tokens"],
            "raw_output_tokens": ev.get("raw_output_tokens"),
            "output_token_mean": ev["output_token_mean"],
            "output_token_p95": ev["output_token_p95"],
            "elapsed_s": ev["elapsed_s"],
            "mutation_s": ev["mutation_s"],
        }.items()
    }


def run_conditions(
    out: Path,
    backend,
    candidates: list[tuple[str, Candidate | None, dict | None]],
    examples: list[CountdownExample],
    *,
    split: str,
    caps: list[int],
    prompt_variants: list[str],
) -> list[dict]:
    summaries = []
    for cap in caps:
        backend.max_new_tokens = cap
        for variant in prompt_variants:
            make_prompt = prompt_fn(variant)
            for candidate_kind, candidate, family_state in candidates:
                ev = evaluate_with_prompt_fn(
                    backend,
                    candidate,
                    examples,
                    make_prompt,
                    family_state=family_state,
                    label=candidate_kind if candidate is None else None,
                )
                row = metric_row(ev, cap=cap, prompt_variant=variant, split=split, candidate_kind=candidate_kind)
                summaries.append(row)
                write_jsonl(
                    out / "per_prompt.jsonl",
                    tag_rows(
                        ev["rows"],
                        candidate_kind=candidate_kind,
                        split=split,
                        prompt_variant=variant,
                        max_new_tokens=cap,
                    ),
                )
                print(
                    f"{split} cap={cap} prompt={variant} {candidate_kind} "
                    f"exact={ev['exact_mean']:.4f} cap_hit={ev['cap_hit_mean']:.4f} "
                    f"malformed={ev['malformed_mean']:.4f}",
                    flush=True,
                )
    return summaries


def run_audit(args) -> None:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    reset_outputs(out, ["per_prompt.jsonl", "summary.json", "summary_rows.jsonl"])
    caps = [int(x) for x in args.max_new_tokens_grid.split(",") if x]
    variants = [x for x in args.prompt_variants.split(",") if x]
    elite_rows = top_candidate_rows(Path(args.source_run), args.top_k)
    if not elite_rows:
        raise ValueError(f"no elite candidates found in {args.source_run}")

    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    holdout = load_examples(
        args.data,
        args.holdout_prompts,
        args.seed + 999,
        allow_repeat=args.allow_repeat_data,
        exclude_ids={ex.id for ex in screen},
    )
    splits = [("holdout", holdout)]
    if args.include_screen:
        splits.insert(0, ("screen", screen))

    common = vars(args).copy()
    common["max_new_tokens"] = max(caps)
    common["stop_at_answer"] = args.stop_at_answer

    elite_args = argparse.Namespace(**common)
    elite_args.rank = args.base_rank
    elite_backend = make_backend(elite_args)
    elite_candidate_objs = [parse_candidate_key(row["candidate"]) for row in elite_rows]
    activation_state_prompt_variants = infer_activation_state_prompt_variants(Path(args.source_run), args.activation_state_prompt_variants)
    family_states, family_state_summary = build_family_states(
        elite_backend,
        elite_candidate_objs,
        screen,
        source_run=Path(args.source_run),
        activation_state_prompt_variants=activation_state_prompt_variants,
    )
    elite_candidates = [("base", None, None)]
    elite_candidates.extend(
        (f"elite_{idx}", candidate, family_states.get(candidate.family))
        for idx, candidate in enumerate(elite_candidate_objs)
    )

    summaries = []
    start = time.time()
    for split, examples in splits:
        summaries.extend(
            run_conditions(
                out,
                elite_backend,
                elite_candidates,
                examples,
                split=split,
                caps=caps,
                prompt_variants=variants,
            )
        )
    elite_backend.clear_candidate()
    del elite_backend
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    aggregate_summary = None
    if not args.skip_aggregate:
        aggregate_args = argparse.Namespace(**common)
        aggregate_args.rank = args.base_rank * len(elite_rows)
        aggregate_backend = make_backend(aggregate_args)
        aggregate_state, aggregate_summary = build_aggregate_state(
            aggregate_backend.model,
            elite_rows,
            args.base_rank,
            args.weight_mode,
        )
        aggregate_candidate = Candidate("elite_aggregate_lora", args.seed, 1.0, 1)
        aggregate_candidates = [("aggregate", aggregate_candidate, aggregate_state)]
        for split, examples in splits:
            summaries.extend(
                run_conditions(
                    out,
                    aggregate_backend,
                    aggregate_candidates,
                    examples,
                    split=split,
                    caps=caps,
                    prompt_variants=variants,
                )
            )

    write_jsonl(out / "summary_rows.jsonl", summaries)
    summary = {
        "kind": "cap_stability_audit",
        "source_run": args.source_run,
        "data": args.data,
        "top_k": len(elite_rows),
        "base_rank": args.base_rank,
        "aggregate_rank": args.base_rank * len(elite_rows),
        "weight_mode": args.weight_mode,
        "caps": caps,
        "prompt_variants": variants,
        **family_state_summary,
        "include_screen": bool(args.include_screen),
        "skip_aggregate": bool(args.skip_aggregate),
        "screen_prompts": len(screen),
        "holdout_prompts": len(holdout),
        "screen_holdout_overlap": len({ex.id for ex in screen} & {ex.id for ex in holdout}),
        "aggregate": aggregate_summary,
        "rows": summaries,
        "elapsed_s": time.time() - start,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit token-cap and prompt stability for LoRA elites and aggregates.")
    parser.add_argument("--source-run", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default=None)
    parser.add_argument("--prompts", type=int, default=64)
    parser.add_argument("--holdout-prompts", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--base-rank", type=int, default=32)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--weight-mode", choices=["uniform", "score", "centered"], default="score")
    parser.add_argument("--targets", default="q_proj,v_proj")
    parser.add_argument("--max-new-tokens-grid", default="32,64,128,256")
    parser.add_argument("--prompt-variants", default="default,reordered,xml")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--stop-at-answer", action="store_true")
    parser.add_argument("--include-screen", action="store_true")
    parser.add_argument("--skip-aggregate", action="store_true")
    parser.add_argument("--activation-state-prompt-variants", default="")
    parser.add_argument("--allow-repeat-data", action="store_true")
    parser.add_argument("--perturbation-backend", choices=["lora"], default="lora")
    parser.add_argument("--dense-snapshot-device", default="model")
    args = parser.parse_args(argv)
    run_audit(args)


if __name__ == "__main__":
    main()
