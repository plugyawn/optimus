from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import torch

from optimus.modeling.dense import dense_noise_tensor
from optimus.modeling.geometry import MatrixSpec, best_rank_projection, lora_update
from optimus.modeling.noise import Candidate, lora_noise_tensors


def matrix_update_for_family(spec: MatrixSpec, candidate: Candidate, family: str) -> torch.Tensor:
    if family == "dense_gaussian":
        return dense_noise_tensor(spec.name, spec.shape, candidate).double()
    if family in {"factor_gaussian_lora", "randomized_projected_gaussian_rank_r"} or family.startswith(
        "spectral_projected_gaussian_rank_r"
    ):
        lora_candidate = Candidate(family, candidate.seed, candidate.sigma, candidate.sign)
        a, b = lora_noise_tensors(
            spec.name,
            (spec.rank, spec.in_features),
            (spec.out_features, spec.rank),
            lora_candidate,
            spec.rank,
        )
        return lora_update(a.double(), b.double())
    if family == "projected_gaussian_rank_r":
        dense_candidate = Candidate("dense_gaussian", candidate.seed, candidate.sigma, candidate.sign)
        dense = dense_noise_tensor(spec.name, spec.shape, dense_candidate).double()
        return best_rank_projection(dense, spec.rank)
    raise ValueError(f"unsupported family: {family}")


def effective_rank(delta: torch.Tensor, energy_threshold: float = 0.99) -> int:
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    if energy_threshold < 0.0 or energy_threshold > 1.0:
        raise ValueError("energy_threshold must be in [0, 1]")
    if delta.numel() == 0:
        return 0
    _, s, _ = torch.linalg.svd(delta.double(), full_matrices=False)
    energy = s * s
    total = energy.sum()
    if float(total.item()) == 0.0:
        return 0
    cumulative = torch.cumsum(energy, dim=0) / total
    hits = torch.nonzero(cumulative >= energy_threshold)
    return int(hits[0].item() + 1) if hits.numel() else int(s.numel())


def matrix_geometry(
    name: str,
    delta: torch.Tensor,
    *,
    sparsity_threshold: float = 1e-5,
    energy_threshold: float = 0.99,
) -> dict:
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    abs_delta = delta.abs()
    params = delta.numel()
    near_zero = int((abs_delta <= sparsity_threshold).sum().item())
    frob_sq = float((delta.double() * delta.double()).sum().item())
    dense_rank = min(delta.shape)
    erank = effective_rank(delta, energy_threshold=energy_threshold)
    return {
        "name": name,
        "shape": list(delta.shape),
        "params": params,
        "sparsity_threshold": sparsity_threshold,
        "l0_sparsity": near_zero / params,
        "nonzero_fraction": 1.0 - (near_zero / params),
        "frob_norm": math.sqrt(frob_sq),
        "frob_sq": frob_sq,
        "mean_abs": float(abs_delta.mean().item()),
        "max_abs": float(abs_delta.max().item()) if params else 0.0,
        "dense_rank": dense_rank,
        "effective_rank_threshold": energy_threshold,
        "effective_rank": erank,
        "effective_rank_fraction": erank / dense_rank if dense_rank else 0.0,
        "numerical_rank": int(torch.linalg.matrix_rank(delta.double()).item()),
    }


def summarize_geometry(rows: list[dict]) -> dict:
    total_params = sum(row["params"] for row in rows)
    total_near_zero = sum(round(row["l0_sparsity"] * row["params"]) for row in rows)
    total_frob_sq = sum(row["frob_sq"] for row in rows)
    weighted_effective_rank = sum(row["effective_rank"] * row["params"] for row in rows)
    weighted_dense_rank = sum(row["dense_rank"] * row["params"] for row in rows)
    return {
        "matrix_count": len(rows),
        "total_params": total_params,
        "total_l0_sparsity": total_near_zero / total_params if total_params else 0.0,
        "total_nonzero_fraction": 1.0 - (total_near_zero / total_params if total_params else 0.0),
        "total_frob_norm": math.sqrt(total_frob_sq),
        "mean_effective_rank": sum(row["effective_rank"] for row in rows) / len(rows) if rows else 0.0,
        "weighted_effective_rank_fraction": (
            weighted_effective_rank / weighted_dense_rank if weighted_dense_rank else 0.0
        ),
        "mean_effective_rank_fraction": (
            sum(row["effective_rank_fraction"] for row in rows) / len(rows) if rows else 0.0
        ),
    }


def family_geometry(
    specs: list[MatrixSpec],
    candidate: Candidate,
    families: list[str],
    *,
    sparsity_threshold: float = 1e-5,
    energy_threshold: float = 0.99,
) -> dict:
    out = {}
    for family in families:
        rows = []
        for spec in specs:
            delta = matrix_update_for_family(spec, candidate, family)
            row = matrix_geometry(
                spec.name,
                delta,
                sparsity_threshold=sparsity_threshold,
                energy_threshold=energy_threshold,
            )
            row["rank_cap"] = spec.lora_rank_cap
            rows.append(row)
        out[family] = {
            "summary": summarize_geometry(rows),
            "matrices": rows,
        }
    return out


def parse_shapes(text: str, rank: int) -> list[MatrixSpec]:
    specs = []
    for idx, item in enumerate(part for part in text.split(",") if part):
        out_text, in_text = item.lower().split("x", 1)
        out_features = int(out_text)
        in_features = int(in_text)
        specs.append(MatrixSpec(f"shape_{idx}_{out_features}x{in_features}", out_features, in_features, rank))
    return specs


def render_markdown(payload: dict) -> str:
    lines = [
        "# Update Geometry Audit",
        "",
        f"- seed: `{payload['candidate']['seed']}`",
        f"- sigma: `{payload['candidate']['sigma']}`",
        f"- sign: `{payload['candidate']['sign']}`",
        f"- sparsity threshold: `{payload['sparsity_threshold']}`",
        f"- effective-rank energy threshold: `{payload['energy_threshold']}`",
        "",
        "| family | total sparsity | total Frobenius norm | mean effective rank | weighted effective-rank fraction |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for family, data in payload["families"].items():
        summary = data["summary"]
        lines.append(
            f"| {family} | {summary['total_l0_sparsity']:.6f} | "
            f"{summary['total_frob_norm']:.6f} | {summary['mean_effective_rank']:.3f} | "
            f"{summary['weighted_effective_rank_fraction']:.6f} |"
        )
    lines.append("")
    lines.append("Dense and factor-Gaussian LoRA can be Frobenius-scale matched while having very different rank/correlation geometry.")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit perturbation family sparsity and effective rank.")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--families",
        default=(
            "dense_gaussian,factor_gaussian_lora,projected_gaussian_rank_r,"
            "randomized_projected_gaussian_rank_r,spectral_projected_gaussian_rank_r"
        ),
    )
    parser.add_argument("--shapes", default="128x128,256x128")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--sign", type=int, default=1)
    parser.add_argument("--sparsity-threshold", type=float, default=1e-5)
    parser.add_argument("--energy-threshold", type=float, default=0.99)
    args = parser.parse_args(argv)

    candidate = Candidate("geometry", args.seed, args.sigma, args.sign)
    specs = parse_shapes(args.shapes, args.rank)
    payload = {
        "candidate": {
            "seed": candidate.seed,
            "sigma": candidate.sigma,
            "sign": candidate.sign,
        },
        "rank": args.rank,
        "shapes": [list(spec.shape) for spec in specs],
        "sparsity_threshold": args.sparsity_threshold,
        "energy_threshold": args.energy_threshold,
        "families": family_geometry(
            specs,
            candidate,
            [family for family in args.families.split(",") if family],
            sparsity_threshold=args.sparsity_threshold,
            energy_threshold=args.energy_threshold,
        ),
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    (args.out / "report.md").write_text(render_markdown(payload))
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
