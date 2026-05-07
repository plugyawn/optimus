from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from .countdown import load_examples
from .experiments import candidate_panel, make_backend, make_prompts_for_backend, maybe_build_family_state, parse_float_list, write_jsonl


def kl_from_logits(p_logits: torch.Tensor, q_logits: torch.Tensor) -> torch.Tensor:
    """Return KL(softmax(p_logits) || softmax(q_logits)) per row."""
    p_logprob = torch.log_softmax(p_logits.float(), dim=-1)
    q_logprob = torch.log_softmax(q_logits.float(), dim=-1)
    p_prob = torch.exp(p_logprob)
    return torch.sum(p_prob * (p_logprob - q_logprob), dim=-1)


def drift_metrics(base_logits: torch.Tensor, candidate_logits: torch.Tensor) -> dict:
    delta = candidate_logits.float() - base_logits.float()
    l2 = torch.linalg.vector_norm(delta, ord=2, dim=-1)
    base_to_candidate = kl_from_logits(base_logits, candidate_logits)
    candidate_to_base = kl_from_logits(candidate_logits, base_logits)
    top1_equal = torch.argmax(base_logits, dim=-1) == torch.argmax(candidate_logits, dim=-1)
    return {
        "logit_l2_mean": float(l2.mean().item()),
        "logit_l2_max": float(l2.max().item()),
        "kl_base_to_candidate_mean": float(base_to_candidate.mean().item()),
        "kl_base_to_candidate_max": float(base_to_candidate.max().item()),
        "kl_candidate_to_base_mean": float(candidate_to_base.mean().item()),
        "kl_candidate_to_base_max": float(candidate_to_base.max().item()),
        "top1_equal_rate": float(top1_equal.float().mean().item()),
        "prompts": int(base_logits.shape[0]),
    }


def summarize(rows: list[dict], *, max_mean_kl: float | None, min_top1_equal: float | None) -> dict:
    out = {"rows": len(rows)}
    if rows:
        for key in [
            "logit_l2_mean",
            "kl_base_to_candidate_mean",
            "kl_candidate_to_base_mean",
            "top1_equal_rate",
        ]:
            values = [float(row[key]) for row in rows]
            out[f"{key}_mean"] = sum(values) / len(values)
            out[f"{key}_min"] = min(values)
            out[f"{key}_max"] = max(values)
    gates = {}
    if max_mean_kl is not None:
        worst_kl = max((float(row["kl_base_to_candidate_mean"]) for row in rows), default=float("inf"))
        gates["max_mean_kl"] = worst_kl <= max_mean_kl
        out["max_mean_kl_threshold"] = max_mean_kl
        out["worst_kl_base_to_candidate_mean"] = worst_kl
    if min_top1_equal is not None:
        worst_top1 = min((float(row["top1_equal_rate"]) for row in rows), default=0.0)
        gates["min_top1_equal"] = worst_top1 >= min_top1_equal
        out["min_top1_equal_threshold"] = min_top1_equal
        out["worst_top1_equal_rate"] = worst_top1
    out["gates"] = gates
    out["pass"] = bool(gates) and all(gates.values())
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Measure task-conditioned logit drift for perturbation candidates.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--data", default=None)
    parser.add_argument("--prompts", type=int, default=32)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--sigma-values", default="")
    parser.add_argument("--targets", default="q_proj,v_proj")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--perturbation-backend", choices=["lora", "dense"], default="lora")
    parser.add_argument("--dense-snapshot-device", choices=["model", "cpu"], default="model")
    parser.add_argument("--dense-noise-mode", choices=["canonical", "paper"], default="canonical")
    parser.add_argument("--stop-at-answer", action="store_true")
    parser.add_argument("--prompt-variant", default="default")
    parser.add_argument("--use-chat-template", action="store_true")
    parser.add_argument(
        "--family",
        default="factor_gaussian_lora",
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
            "sparse_low_rank_lora",
            "sparse_low_rank_lora_d0p125",
            "sparse_low_rank_lora_d0p25",
            "sparse_low_rank_lora_d0p5",
            "dense_gaussian",
            "anzo",
            "target_svd",
            "random_ortho",
            "anzo_random_target",
        ],
    )
    parser.add_argument("--antithetic", action="store_true")
    parser.add_argument("--allow-repeat-data", action="store_true")
    parser.add_argument("--max-mean-kl", type=float, default=None)
    parser.add_argument("--min-top1-equal", type=float, default=None)
    args = parser.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["candidate_drift.jsonl", "summary.json"]:
        path = out / name
        if path.exists():
            path.unlink()
    backend = make_backend(args)
    examples = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    prompts = make_prompts_for_backend(backend, args, examples)
    backend.clear_candidate()
    base_logits = backend.logits_signature(prompts)
    family_state = maybe_build_family_state(args, backend, examples)
    sigma_values = parse_float_list(args.sigma_values) if args.sigma_values else [args.sigma]
    candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic, sigma_values)
    rows = []
    for idx, candidate in enumerate(candidates):
        backend.set_candidate(candidate, family_state)
        logits = backend.logits_signature(prompts)
        row = {
            "candidate": candidate.key,
            "seed": candidate.seed,
            "sigma": candidate.sigma,
            "sign": candidate.sign,
            **drift_metrics(base_logits, logits),
        }
        rows.append(row)
        write_jsonl(out / "candidate_drift.jsonl", [row])
        print(f"{idx + 1}/{len(candidates)} {candidate.key} kl={row['kl_base_to_candidate_mean']:.6g}", flush=True)
    backend.clear_candidate()
    summary = {
        "kind": "logit_drift",
        "model": args.model,
        "data": args.data,
        "family": args.family,
        "population": len(candidates),
        "rank": args.rank,
        "sigma": args.sigma,
        "sigma_values": sigma_values,
        "targets": args.targets,
        "perturbation_backend": args.perturbation_backend,
        "dense_noise_mode": args.dense_noise_mode,
        "prompt_variant": args.prompt_variant,
        "use_chat_template": args.use_chat_template,
        "prompts": len(examples),
        **summarize(rows, max_mean_kl=args.max_mean_kl, min_top1_equal=args.min_top1_equal),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
