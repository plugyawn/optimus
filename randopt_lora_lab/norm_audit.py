from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from .countdown import load_examples
from .experiments import candidate_panel, make_backend, maybe_build_family_state, write_jsonl
from .lora_space import Candidate, lora_noise_tensors


def lora_modules(model):
    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and module.lora_A:
            adapter = next(iter(module.lora_A.keys()))
            yield name, module.lora_A[adapter].weight, module.lora_B[adapter].weight


def candidate_norms(
    model,
    candidate: Candidate,
    rank: int,
    family_state: dict | None,
    *,
    exact_ba: bool,
) -> dict:
    a_sq = 0.0
    b_sq = 0.0
    ba_sq = 0.0
    upper_sq = 0.0
    modules = 0
    for name, a_weight, b_weight in lora_modules(model):
        a, b = lora_noise_tensors(
            name,
            tuple(a_weight.shape),
            tuple(b_weight.shape),
            candidate,
            rank,
            family_state=family_state,
            state_key=name,
        )
        module_a_sq = float(a.float().pow(2).sum().item())
        module_b_sq = float(b.float().pow(2).sum().item())
        a_sq += module_a_sq
        b_sq += module_b_sq
        upper_sq += module_a_sq * module_b_sq
        if exact_ba:
            ba = b.float() @ a.float()
            ba_sq += float(ba.pow(2).sum().item())
        modules += 1
    return {
        "candidate": candidate.key,
        "family": candidate.family,
        "seed": candidate.seed,
        "sigma": candidate.sigma,
        "sign": candidate.sign,
        "modules": modules,
        "a_frob": math.sqrt(a_sq),
        "b_frob": math.sqrt(b_sq),
        "ba_frob": math.sqrt(ba_sq) if exact_ba else None,
        "ba_frob_upper": math.sqrt(upper_sq),
    }


def summarize(rows: list[dict]) -> dict:
    out = {"rows": len(rows)}
    for key in ["a_frob", "b_frob", "ba_frob", "ba_frob_upper"]:
        values = [float(row[key]) for row in rows if row.get(key) is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        var = sum((x - mean) ** 2 for x in values) / len(values)
        out[f"{key}_mean"] = mean
        out[f"{key}_std"] = math.sqrt(var)
        out[f"{key}_min"] = min(values)
        out[f"{key}_max"] = max(values)
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Audit LoRA perturbation norms for RandOpt families.")
    p.add_argument("--out", required=True)
    p.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    p.add_argument("--data", default=None)
    p.add_argument("--prompts", type=int, default=32)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--population", type=int, default=64)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--sigma", type=float, default=0.01)
    p.add_argument("--targets", default="q_proj,v_proj")
    p.add_argument("--max-new-tokens", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    p.add_argument(
        "--family",
        default="isotropic",
        choices=[
            "isotropic",
            "anzo",
            "target_svd",
            "random_ortho",
            "anzo_random_target",
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
            "activation_spectral_lora_tscale_q2_v1",
            "activation_spectral_lora_tscale_q2_v1p045",
            "activation_spectral_lora_tscale_q1p886_v0p985",
            "activation_spectral_lora_tscale_q2_k1_v1_o2",
            "activation_spectral_lora_tscale_q2_k1p045_v1p045_o2",
            "activation_spectral_lora_tscale_q1p333_k0p697_v0p697_o1p333",
            "activation_spectral_lora_sv",
            "activation_spectral_lora_sv_c0p75",
            "activation_spectral_lora_sv_c1p25",
            "activation_spectral_lora_sv_c1p5",
            "activation_spectral_lora_sv_c2",
        ],
    )
    p.add_argument("--antithetic", action="store_true")
    p.add_argument("--allow-repeat-data", action="store_true")
    p.add_argument("--exact-ba-candidates", type=int, default=0)
    args = p.parse_args(argv)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in ["candidate_norms.jsonl", "summary.json"]:
        path = out / name
        if path.exists():
            path.unlink()
    backend = make_backend(args)
    screen = load_examples(args.data, args.prompts, args.seed, allow_repeat=args.allow_repeat_data)
    family_state = maybe_build_family_state(args, backend, screen)
    candidates = candidate_panel(args.family, args.population, args.sigma, args.seed, args.antithetic)
    rows = []
    for idx, candidate in enumerate(candidates):
        row = candidate_norms(
            backend.model,
            candidate,
            args.rank,
            family_state,
            exact_ba=idx < args.exact_ba_candidates,
        )
        rows.append(row)
        write_jsonl(out / "candidate_norms.jsonl", [row])
        print(json.dumps(row, sort_keys=True), flush=True)
    summary = {
        "kind": "norm_audit",
        "family": args.family,
        "population": len(candidates),
        "sigma": args.sigma,
        "rank": args.rank,
        "targets": args.targets,
        "exact_ba_candidates": args.exact_ba_candidates,
        **summarize(rows),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
