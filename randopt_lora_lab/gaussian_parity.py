from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import torch


@dataclass(frozen=True)
class MatrixSpec:
    name: str
    out_features: int
    in_features: int
    rank: int

    @property
    def shape(self) -> tuple[int, int]:
        return (self.out_features, self.in_features)

    @property
    def dense_params(self) -> int:
        return self.out_features * self.in_features

    @property
    def lora_params(self) -> int:
        return self.rank * (self.out_features + self.in_features)

    @property
    def dense_rank_almost_sure(self) -> int:
        return min(self.out_features, self.in_features)

    @property
    def lora_rank_cap(self) -> int:
        return min(self.rank, self.dense_rank_almost_sure)

    @property
    def param_fraction(self) -> float:
        return self.lora_params / self.dense_params

    @property
    def rank_fraction(self) -> float:
        return self.lora_rank_cap / self.dense_rank_almost_sure


def qwen25_3b_qv_specs(rank: int = 8, layers: int = 36) -> list[MatrixSpec]:
    """Return the q_proj/v_proj matrix shapes used by Qwen2.5-3B-style attention.

    PyTorch linear weights are shaped [out_features, in_features]. For grouped
    query attention, v_proj has only num_key_value_heads * head_dim outputs.
    """

    hidden_size = 2048
    num_attention_heads = 16
    num_key_value_heads = 2
    head_dim = hidden_size // num_attention_heads
    kv_out = num_key_value_heads * head_dim
    specs: list[MatrixSpec] = []
    for layer in range(layers):
        specs.append(MatrixSpec(f"model.layers.{layer}.self_attn.q_proj", hidden_size, hidden_size, rank))
        specs.append(MatrixSpec(f"model.layers.{layer}.self_attn.v_proj", kv_out, hidden_size, rank))
    return specs


def summarize_specs(specs: list[MatrixSpec]) -> dict:
    dense_params = sum(spec.dense_params for spec in specs)
    lora_params = sum(spec.lora_params for spec in specs)
    weighted_dense_rank = sum(spec.dense_rank_almost_sure for spec in specs)
    weighted_lora_rank = sum(spec.lora_rank_cap for spec in specs)
    return {
        "matrices": [asdict(spec) | {
            "dense_params": spec.dense_params,
            "lora_params": spec.lora_params,
            "dense_rank_almost_sure": spec.dense_rank_almost_sure,
            "lora_rank_cap": spec.lora_rank_cap,
            "param_fraction": spec.param_fraction,
            "rank_fraction": spec.rank_fraction,
        } for spec in specs],
        "total_dense_params": dense_params,
        "total_lora_params": lora_params,
        "total_param_fraction": lora_params / dense_params,
        "summed_dense_rank_almost_sure": weighted_dense_rank,
        "summed_lora_rank_cap": weighted_lora_rank,
        "summed_rank_fraction": weighted_lora_rank / weighted_dense_rank,
    }


def expected_update_stats(specs: list[MatrixSpec], sigma: float) -> dict:
    """Expected scale for dense Gaussian and current factor-Gaussian LoRA.

    The current LoRA materializer samples A = sigma * N(0, 1) and
    B = N(0, 1) / sqrt(rank). For one update entry,

        Delta W_ij = sum_k B_ik A_kj

    has variance sigma^2. Therefore expected Frobenius norm matches a dense iid
    Gaussian with per-entry std sigma, even though the LoRA update is low-rank
    and has correlated entries.
    """

    rows = []
    total_dense_frob_sq = 0.0
    total_factor_lora_frob_sq = 0.0
    for spec in specs:
        dense_frob_sq = spec.dense_params * sigma * sigma
        factor_lora_frob_sq = dense_frob_sq
        rows.append({
            "name": spec.name,
            "shape": list(spec.shape),
            "rank": spec.rank,
            "dense_expected_frob_rms": math.sqrt(dense_frob_sq),
            "factor_lora_expected_frob_rms": math.sqrt(factor_lora_frob_sq),
            "expected_frob_ratio_factor_lora_over_dense": 1.0,
            "dense_rank_almost_sure": spec.dense_rank_almost_sure,
            "factor_lora_rank_cap": spec.lora_rank_cap,
        })
        total_dense_frob_sq += dense_frob_sq
        total_factor_lora_frob_sq += factor_lora_frob_sq
    return {
        "sigma": sigma,
        "per_matrix": rows,
        "total_dense_expected_frob_rms": math.sqrt(total_dense_frob_sq),
        "total_factor_lora_expected_frob_rms": math.sqrt(total_factor_lora_frob_sq),
        "total_expected_frob_ratio_factor_lora_over_dense": 1.0,
    }


def dense_gaussian_matrix(shape: tuple[int, int], seed: int, sigma: float = 1.0) -> torch.Tensor:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    return sigma * torch.randn(shape, generator=gen, dtype=torch.float64)


def lora_update(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Return the dense matrix represented by LoRA factors.

    `a` is [rank, in_features] and `b` is [out_features, rank].
    """

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("LoRA factors must both be matrices")
    if b.shape[1] != a.shape[0]:
        raise ValueError(f"incompatible LoRA factor shapes: A={tuple(a.shape)} B={tuple(b.shape)}")
    return b @ a


def best_rank_projection(delta: torch.Tensor, rank: int) -> torch.Tensor:
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if rank == 0:
        return torch.zeros_like(delta)
    u, s, vh = torch.linalg.svd(delta.double(), full_matrices=False)
    k = min(rank, s.numel())
    return (u[:, :k] * s[:k].unsqueeze(0)) @ vh[:k, :]


def low_rank_factors_from_dense(delta: torch.Tensor, rank: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Factor the best rank-r SVD projection as LoRA A/B tensors.

    Returns A [rank, in_features] and B [out_features, rank], padding with zeros
    if `rank` exceeds the matrix's numerical rank capacity.
    """

    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    out_features, in_features = delta.shape
    if rank == 0:
        return torch.zeros((0, in_features), dtype=delta.dtype), torch.zeros((out_features, 0), dtype=delta.dtype)
    u, s, vh = torch.linalg.svd(delta.double(), full_matrices=False)
    k = min(rank, s.numel())
    root_s = torch.sqrt(s[:k])
    b = u[:, :k] * root_s.unsqueeze(0)
    a = root_s.unsqueeze(1) * vh[:k, :]
    if k < rank:
        a = torch.cat([a, torch.zeros((rank - k, in_features), dtype=a.dtype)], dim=0)
        b = torch.cat([b, torch.zeros((out_features, rank - k), dtype=b.dtype)], dim=1)
    return a.to(delta.dtype), b.to(delta.dtype)


def randomized_low_rank_factors_from_dense(
    delta: torch.Tensor,
    rank: int,
    *,
    oversample: int = 8,
    n_iter: int = 1,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Approximate a rank-r projection using a randomized range finder.

    This is the fast bridge candidate for dense-Gaussian-like LoRA directions:
    it avoids a full SVD of the dense matrix and only factors a small sketched
    matrix. The result is not the exact best rank-r projection.
    """

    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if oversample < 0:
        raise ValueError("oversample must be nonnegative")
    if n_iter < 0:
        raise ValueError("n_iter must be nonnegative")
    out_features, in_features = delta.shape
    if rank == 0:
        return torch.zeros((0, in_features), dtype=delta.dtype), torch.zeros((out_features, 0), dtype=delta.dtype)
    max_rank = min(out_features, in_features)
    if rank >= max_rank:
        return low_rank_factors_from_dense(delta, rank)

    sketch_rank = min(max_rank, rank + oversample)
    work = delta.float()
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) % (2**63 - 1))
    omega = torch.randn((in_features, sketch_rank), generator=gen, dtype=torch.float32)
    y = work @ omega
    for _ in range(n_iter):
        y = work @ (work.T @ y)
    q, _ = torch.linalg.qr(y, mode="reduced")
    small = q.T @ work
    u_hat, s, vh = torch.linalg.svd(small.double(), full_matrices=False)
    k = min(rank, s.numel())
    u = q.double() @ u_hat[:, :k]
    root_s = torch.sqrt(s[:k])
    b = u * root_s.unsqueeze(0)
    a = root_s.unsqueeze(1) * vh[:k, :]
    if k < rank:
        a = torch.cat([a, torch.zeros((rank - k, in_features), dtype=a.dtype)], dim=0)
        b = torch.cat([b, torch.zeros((out_features, rank - k), dtype=b.dtype)], dim=1)
    return a.to(delta.dtype), b.to(delta.dtype)


def projection_stats(delta: torch.Tensor, ranks: list[int]) -> list[dict]:
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    _, s, _ = torch.linalg.svd(delta.double(), full_matrices=False)
    total_energy = float((s * s).sum().item())
    rows = []
    for rank in ranks:
        if rank < 0:
            raise ValueError("rank must be nonnegative")
        k = min(rank, s.numel())
        captured = float((s[:k] * s[:k]).sum().item())
        captured_fraction = 1.0 if total_energy == 0.0 else captured / total_energy
        rows.append({
            "rank": rank,
            "captured_frob_fraction": captured_fraction,
            "relative_frob_error": max(0.0, 1.0 - captured_fraction) ** 0.5,
        })
    return rows


def required_rank_for_energy(delta: torch.Tensor, thresholds: list[float]) -> dict[float, int]:
    if delta.ndim != 2:
        raise ValueError("delta must be a matrix")
    if any(threshold < 0.0 or threshold > 1.0 for threshold in thresholds):
        raise ValueError("thresholds must be in [0, 1]")
    _, s, _ = torch.linalg.svd(delta.double(), full_matrices=False)
    energy = s * s
    total = energy.sum()
    if float(total.item()) == 0.0:
        return {threshold: 0 for threshold in thresholds}
    cumulative = torch.cumsum(energy, dim=0) / total
    out: dict[float, int] = {}
    for threshold in thresholds:
        hits = torch.nonzero(cumulative >= threshold)
        out[threshold] = int(hits[0].item() + 1) if hits.numel() else int(s.numel())
    return out


def _parse_ranks(text: str) -> list[int]:
    return [int(item) for item in text.split(",") if item]


def _parse_shape(text: str) -> tuple[int, int]:
    left, right = text.lower().split("x", 1)
    return int(left), int(right)


def _render_markdown(payload: dict) -> str:
    lines = [
        "# Gaussian vs LoRA Parity Audit",
        "",
        "## Capacity Summary",
        "",
        f"- Total dense parameters: `{payload['qwen25_3b_qv']['total_dense_params']}`",
        f"- Total LoRA parameters: `{payload['qwen25_3b_qv']['total_lora_params']}`",
        f"- LoRA parameter fraction: `{payload['qwen25_3b_qv']['total_param_fraction']:.6f}`",
        f"- Summed dense rank almost surely: `{payload['qwen25_3b_qv']['summed_dense_rank_almost_sure']}`",
        f"- Summed LoRA rank cap: `{payload['qwen25_3b_qv']['summed_lora_rank_cap']}`",
        f"- Summed rank fraction: `{payload['qwen25_3b_qv']['summed_rank_fraction']:.6f}`",
        f"- Expected Frobenius RMS ratio at sigma={payload['expected_update_stats']['sigma']}: `{payload['expected_update_stats']['total_expected_frob_ratio_factor_lora_over_dense']:.6f}`",
        "",
        "The current factor-Gaussian LoRA scaling matches dense Gaussian expected Frobenius norm per matrix, so Frobenius norm alone is not evidence of dense-Gaussian parity.",
        "A low-rank LoRA perturbation still cannot exactly represent an arbitrary dense Gaussian perturbation unless the dense perturbation's rank is at most the LoRA rank.",
        "",
        "## Empirical Projection Samples",
        "",
    ]
    for sample in payload["projection_samples"]:
        lines.append(f"### Shape `{sample['shape'][0]}x{sample['shape'][1]}`")
        lines.append("")
        lines.append("| rank | captured Frobenius energy | relative Frobenius error |")
        lines.append("| ---: | ---: | ---: |")
        for row in sample["projection_stats"]:
            lines.append(f"| {row['rank']} | {row['captured_frob_fraction']:.6f} | {row['relative_frob_error']:.6f} |")
        lines.append("")
        required = ", ".join(f"{float(k):.2f}: r={v}" for k, v in sample["required_rank_for_energy"].items())
        lines.append(f"Required rank by energy threshold: {required}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Audit dense Gaussian vs low-rank LoRA parity limits.")
    parser.add_argument("--out", type=Path, required=True, help="Output directory for summary.json and report.md")
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--sigma", type=float, default=0.01)
    parser.add_argument("--ranks", type=str, default="1,2,4,8,16,32,64")
    parser.add_argument("--shapes", type=str, default="128x128,256x128")
    parser.add_argument("--thresholds", type=str, default="0.5,0.9,0.99")
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args(argv)

    ranks = _parse_ranks(args.ranks)
    thresholds = [float(item) for item in args.thresholds.split(",") if item]
    samples = []
    for offset, shape_text in enumerate(args.shapes.split(",")):
        shape = _parse_shape(shape_text)
        delta = dense_gaussian_matrix(shape, seed=args.seed + offset)
        samples.append({
            "shape": list(shape),
            "projection_stats": projection_stats(delta, ranks),
            "required_rank_for_energy": {str(k): v for k, v in required_rank_for_energy(delta, thresholds).items()},
        })

    payload = {
        "rank": args.rank,
        "qwen25_3b_qv": summarize_specs(qwen25_3b_qv_specs(rank=args.rank)),
        "expected_update_stats": expected_update_stats(qwen25_3b_qv_specs(rank=args.rank), args.sigma),
        "projection_samples": samples,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    (args.out / "report.md").write_text(_render_markdown(payload))


if __name__ == "__main__":
    main()
