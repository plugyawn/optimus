from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .gaussian_parity import (
    low_rank_factors_from_dense,
    randomized_low_rank_factors_from_dense,
    spectral_projected_gaussian_factors,
)


@dataclass(frozen=True)
class Candidate:
    family: str
    seed: int
    sigma: float
    sign: int = 1

    @property
    def key(self) -> str:
        return f"{self.family}:seed{self.seed}:s{self.sigma:g}:sign{self.sign}"


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def canonical_module_name(name: str) -> str:
    """Return the bare transformer module path shared by PEFT and vLLM adapters."""

    marker = "model.layers."
    idx = name.find(marker)
    return name[idx:] if idx >= 0 else name


def sparse_lora_density(family: str) -> float:
    if family == "sparse_low_rank_lora":
        return 0.25
    match = re.fullmatch(r"sparse_low_rank_lora_d([0-9]+(?:p[0-9]+)?)", family)
    if not match:
        raise ValueError(f"not a sparse low-rank LoRA family: {family}")
    density = float(match.group(1).replace("p", "."))
    if not 0.0 < density <= 1.0:
        raise ValueError(f"sparse low-rank LoRA density must be in (0, 1], got {density}")
    return density


def spectral_projected_scale(family: str) -> float:
    if family == "spectral_projected_gaussian_rank_r":
        return 1.0
    match = re.fullmatch(r"spectral_projected_gaussian_rank_r_c([0-9]+(?:p[0-9]+)?)", family)
    if not match:
        raise ValueError(f"not a spectral projected Gaussian family: {family}")
    scale = float(match.group(1).replace("p", "."))
    if scale <= 0.0:
        raise ValueError(f"spectral projected scale must be positive, got {scale}")
    return scale


def lora_noise_tensors(
    module_name: str,
    a_shape: torch.Size | tuple[int, ...],
    b_shape: torch.Size | tuple[int, ...],
    candidate: Candidate,
    rank: int,
    *,
    family_state: dict | None = None,
    state_key: str | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Materialize canonical CPU LoRA tensors for one candidate/module.

    Both PEFT in-process mutation and vLLM safetensor generation must call this
    path. Candidate identity otherwise drifts because CPU/CUDA RNGs and PEFT/vLLM
    module names are not equivalent.
    """

    family_state = family_state or {}
    canonical_name = canonical_module_name(module_name)
    lookup_key = state_key or module_name
    if lookup_key in family_state and isinstance(family_state[lookup_key], dict):
        spec = family_state[lookup_key]
        if "fixed_a" in spec and "fixed_b" in spec:
            return spec["fixed_a"].cpu().float().contiguous(), spec["fixed_b"].cpu().float().contiguous()
    if candidate.family == "projected_gaussian_rank_r":
        gen = torch.Generator(device="cpu")
        gen.manual_seed((candidate.seed + stable_int(canonical_name + ":dense_gaussian")) % (2**63 - 1))
        dense = torch.randn((int(b_shape[0]), int(a_shape[1])), generator=gen, dtype=torch.float32)
        dense.mul_(candidate.sign * candidate.sigma)
        return low_rank_factors_from_dense(dense, rank)
    if candidate.family == "randomized_projected_gaussian_rank_r":
        gen = torch.Generator(device="cpu")
        gen.manual_seed((candidate.seed + stable_int(canonical_name + ":dense_gaussian")) % (2**63 - 1))
        dense = torch.randn((int(b_shape[0]), int(a_shape[1])), generator=gen, dtype=torch.float32)
        dense.mul_(candidate.sign * candidate.sigma)
        sketch_seed = (candidate.seed + stable_int(canonical_name + ":randomized_projection")) % (2**63 - 1)
        return randomized_low_rank_factors_from_dense(dense, rank, oversample=8, n_iter=1, seed=sketch_seed)
    if candidate.family.startswith("spectral_projected_gaussian_rank_r"):
        spectral_seed = (candidate.seed + stable_int(canonical_name + ":spectral_projection")) % (2**63 - 1)
        scale = spectral_projected_scale(candidate.family)
        return spectral_projected_gaussian_factors(
            int(b_shape[0]),
            int(a_shape[1]),
            rank,
            sigma=candidate.sigma * scale,
            sign=candidate.sign,
            seed=spectral_seed,
            dtype=torch.float32,
        )
    gen = torch.Generator(device="cpu")
    gen.manual_seed((candidate.seed + stable_int(canonical_name)) % (2**63 - 1))
    a_noise = torch.randn(tuple(a_shape), generator=gen, dtype=torch.float32)
    b_noise = torch.randn(tuple(b_shape), generator=gen, dtype=torch.float32)
    sparse_density = None
    if candidate.family.startswith("sparse_low_rank_lora"):
        sparse_density = sparse_lora_density(candidate.family)
        a_mask = (torch.rand(tuple(a_shape), generator=gen, dtype=torch.float32) < sparse_density).float()
        b_mask = (torch.rand(tuple(b_shape), generator=gen, dtype=torch.float32) < sparse_density).float()
        scale = math.sqrt(sparse_density)
        a_noise = a_noise * a_mask / scale
        b_noise = b_noise * b_mask / scale
    if candidate.family in {"anzo", "target_svd", "random_ortho", "anzo_random_target"} and lookup_key in family_state:
        basis = family_state[lookup_key].cpu().float()
        rows = min(a_noise.shape[0], basis.shape[0])
        cols = min(a_noise.shape[1], basis.shape[1])
        a_noise[:rows, :cols] = basis[:rows, :cols]
    elif lookup_key in family_state and isinstance(family_state[lookup_key], dict):
        spec = family_state[lookup_key]
        mode = spec.get("mode", "elite_basis")
        if "col_scale" in spec:
            col_scale = spec["col_scale"].cpu().float()
            a_noise.mul_(col_scale[: a_noise.shape[1]].clamp(0.05, 20.0).view(1, -1))
        if "basis" in spec:
            basis = spec["basis"].cpu().float()
            if basis.ndim == 2 and basis.numel() > 0:
                cols = min(a_noise.shape[1], basis.shape[1])
                basis = basis[:, :cols]
                if mode == "activation_overwrite":
                    rows = min(a_noise.shape[0], basis.shape[0])
                    a_noise[:rows, :cols] = basis[:rows, :cols]
                else:
                    gen_basis = torch.Generator(device="cpu")
                    gen_basis.manual_seed((candidate.seed + stable_int(canonical_name + ":adaptive_basis")) % (2**63 - 1))
                    coeff = torch.randn((a_noise.shape[0], basis.shape[0]), generator=gen_basis, dtype=torch.float32)
                    basis_noise = (coeff @ basis) / math.sqrt(max(1, basis.shape[0]))
                    basis_noise = basis_noise / basis_noise.std(unbiased=False).clamp_min(1e-6)
                    residual_scale = float(spec.get("residual_scale", 0.5))
                    basis_scale = float(spec.get("basis_scale", 1.0))
                    a_noise[:, :cols] = residual_scale * a_noise[:, :cols] + basis_scale * basis_noise
    a = candidate.sign * candidate.sigma * a_noise
    b = b_noise / math.sqrt(rank)
    return a.contiguous(), b.contiguous()


def lora_module_names(model, suffixes: tuple[str, ...]) -> list[str]:
    names = []
    for name, module in model.named_modules():
        if any(name.endswith(s) for s in suffixes):
            if hasattr(module, "weight") and getattr(module.weight, "ndim", 0) == 2:
                names.append(name)
    return names


def fill_lora_gaussian(model, candidate: Candidate, rank: int, family_state: dict | None = None) -> None:
    family_state = family_state or {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not module.lora_A:
            continue
        adapter = next(iter(module.lora_A.keys()))
        a = module.lora_A[adapter].weight
        b = module.lora_B[adapter].weight
        a_value, b_value = lora_noise_tensors(
            name,
            a.shape,
            b.shape,
            candidate,
            rank,
            family_state=family_state,
            state_key=name,
        )
        with torch.no_grad():
            a.copy_(a_value.to(device=a.device, dtype=a.dtype))
            b.copy_(b_value.to(device=b.device, dtype=b.dtype))


def zero_lora(model) -> None:
    with torch.no_grad():
        for module in model.modules():
            if hasattr(module, "lora_A") and module.lora_A:
                for layer in module.lora_A.values():
                    layer.weight.zero_()
                for layer in module.lora_B.values():
                    layer.weight.zero_()


def build_anzo_state(
    model,
    tokenizer,
    target_prompts: list[str],
    anchor_prompts: list[str],
    rank: int,
    *,
    subtract_anchor: bool = True,
) -> dict:
    """Build a cheap target-active/anchor-quiet input basis for LoRA A matrices.

    For each LoRA target module, collect input activations on target and anchor
    prompts, then use the top right singular vectors of target residual energy:

        X_target - projection_on_anchor_subspace(X_target)

    The returned tensors have shape [rank, in_features], matching LoRA A.
    """

    device = model.device
    handles = []
    acts: dict[str, dict[str, list[torch.Tensor]]] = {}

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float()
            x = x.reshape(-1, x.shape[-1]).cpu()
            if x.shape[0] > 512:
                stride = max(1, x.shape[0] // 512)
                x = x[::stride][:512]
            phase = getattr(model, "_anzo_phase", "target")
            acts.setdefault(name, {}).setdefault(phase, []).append(x)

        return hook

    for name, module in model.named_modules():
        if hasattr(module, "lora_A") and module.lora_A:
            handles.append(module.register_forward_hook(make_hook(name)))

    @torch.no_grad()
    def run(prompts: list[str], phase: str):
        model._anzo_phase = phase
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=256).to(device)
        _ = model(**inputs)

    try:
        run(anchor_prompts, "anchor")
        run(target_prompts, "target")
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_anzo_phase"):
            delattr(model, "_anzo_phase")

    state = {}
    for name, by_phase in acts.items():
        target = torch.cat(by_phase.get("target", []), dim=0)
        anchor = torch.cat(by_phase.get("anchor", []), dim=0)
        target = target - target.mean(dim=0, keepdim=True)
        anchor = anchor - anchor.mean(dim=0, keepdim=True)
        # Anchor subspace from a small SVD. Keep it modest so target signal can survive.
        q = min(rank * 4, anchor.shape[0], anchor.shape[1])
        if subtract_anchor and q > 0:
            _, _, vh_a = torch.linalg.svd(anchor, full_matrices=False)
            anchor_basis = vh_a[:q].T
            target = target - (target @ anchor_basis) @ anchor_basis.T
        _, _, vh_t = torch.linalg.svd(target, full_matrices=False)
        basis = vh_t[:rank]
        if basis.shape[0] < rank:
            basis = F.pad(basis, (0, 0, 0, rank - basis.shape[0]))
        state[name] = basis.contiguous()
    return state


def build_random_orthonormal_state(model, rank: int, seed: int) -> dict:
    state = {}
    for name, module in model.named_modules():
        if not hasattr(module, "lora_A") or not module.lora_A:
            continue
        adapter = next(iter(module.lora_A.keys()))
        weight = module.lora_A[adapter].weight
        gen = torch.Generator(device="cpu")
        gen.manual_seed((seed + stable_int(canonical_module_name(name) + ":random_ortho")) % (2**63 - 1))
        mat = torch.randn((int(weight.shape[1]), rank), generator=gen, dtype=torch.float32)
        q, _ = torch.linalg.qr(mat, mode="reduced")
        state[name] = q.T.contiguous()
    return state
