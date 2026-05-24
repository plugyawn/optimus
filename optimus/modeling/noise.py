from __future__ import annotations

import math
import re

import torch
import torch.nn.functional as F
from optimus.core.perturbations import PerturbationSpec as Candidate
from optimus.core.perturbations import canonical_module_name, stable_int

from optimus.modeling.geometry import (
    low_rank_factors_from_dense,
    randomized_low_rank_factors_from_dense,
    spectral_projected_gaussian_factors,
)


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


def activation_spectral_scale(family: str) -> float:
    if family in {"activation_spectral_lora", "activation_spectral_lora_sv"}:
        return 1.0
    match = re.fullmatch(r"activation_spectral_lora(?:_sv)?_c([0-9]+(?:p[0-9]+)?)", family)
    if not match:
        raise ValueError(f"not an activation spectral LoRA family: {family}")
    scale = float(match.group(1).replace("p", "."))
    if scale <= 0.0:
        raise ValueError(f"activation spectral scale must be positive, got {scale}")
    return scale


def activation_spectral_uses_singular_values(family: str) -> bool:
    return family.startswith("activation_spectral_lora_sv")


def activation_generalized_spectral_scale(family: str) -> float:
    if family in {"activation_generalized_spectral_lora", "activation_generalized_spectral_lora_sv"}:
        return 1.0
    match = re.fullmatch(r"activation_generalized_spectral_lora(?:_sv)?_c([0-9]+(?:p[0-9]+)?)", family)
    if not match:
        raise ValueError(f"not an activation generalized spectral LoRA family: {family}")
    scale = float(match.group(1).replace("p", "."))
    if scale <= 0.0:
        raise ValueError(f"activation generalized spectral scale must be positive, got {scale}")
    return scale


def activation_generalized_spectral_uses_singular_values(family: str) -> bool:
    return family.startswith("activation_generalized_spectral_lora_sv")


def _scale_value(text: str) -> float:
    scale = float(text.replace("p", "."))
    if scale <= 0.0:
        raise ValueError(f"activation spectral scale must be positive, got {scale}")
    return scale


def activation_target_key(module_name: str) -> str:
    base = canonical_module_name(module_name).split(".")[-1]
    return {
        "q_proj": "q",
        "k_proj": "k",
        "v_proj": "v",
        "o_proj": "o",
        "gate_proj": "gate",
        "up_proj": "up",
        "down_proj": "down",
    }.get(base, base)


def activation_basis_target_scales(family: str) -> dict[str, float] | None:
    match = re.fullmatch(
        r"(?:activation_spectral_lora|activation_generalized_spectral_lora)(?:_sv)?_tscale_([a-z0-9p_]+)",
        family,
    )
    if not match:
        return None
    out: dict[str, float] = {}
    for token in match.group(1).split("_"):
        token_match = re.fullmatch(r"(q|k|v|o|gate|up|down)([0-9]+(?:p[0-9]+)?)", token)
        if not token_match:
            raise ValueError(f"bad activation target-scale token {token!r} in {family}")
        out[token_match.group(1)] = _scale_value(token_match.group(2))
    if not out:
        raise ValueError(f"target-scaled activation spectral family has no target scales: {family}")
    return out


def activation_basis_spectral_scale(family: str, module_name: str | None = None) -> float:
    target_scales = activation_basis_target_scales(family)
    if target_scales is not None:
        if module_name is None:
            raise ValueError(f"{family} requires a module name to choose its target scale")
        target = activation_target_key(module_name)
        if target not in target_scales:
            raise ValueError(f"{family} has no target scale for {target} ({module_name})")
        return target_scales[target]
    if family.startswith("activation_generalized_spectral_lora"):
        return activation_generalized_spectral_scale(family)
    return activation_spectral_scale(family)


def activation_basis_spectral_uses_singular_values(family: str) -> bool:
    if family.startswith("activation_generalized_spectral_lora"):
        return activation_generalized_spectral_uses_singular_values(family)
    return activation_spectral_uses_singular_values(family)


def activation_projected_scale(family: str) -> float:
    if family == "activation_projected_gaussian_rank_r":
        return 1.0
    match = re.fullmatch(r"activation_projected_gaussian_rank_r_c([0-9]+(?:p[0-9]+)?)", family)
    if not match:
        raise ValueError(f"not an activation projected Gaussian family: {family}")
    scale = float(match.group(1).replace("p", "."))
    if scale <= 0.0:
        raise ValueError(f"activation projected scale must be positive, got {scale}")
    return scale


def activation_generalized_projected_scale(family: str) -> float:
    if family == "activation_generalized_projected_gaussian_rank_r":
        return 1.0
    match = re.fullmatch(r"activation_generalized_projected_gaussian_rank_r_c([0-9]+(?:p[0-9]+)?)", family)
    if not match:
        raise ValueError(f"not an activation generalized projected Gaussian family: {family}")
    scale = float(match.group(1).replace("p", "."))
    if scale <= 0.0:
        raise ValueError(f"activation generalized projected scale must be positive, got {scale}")
    return scale


def activation_right_projected_scale(family: str) -> float:
    if family.startswith("activation_generalized_projected_gaussian_rank_r"):
        return activation_generalized_projected_scale(family)
    return activation_projected_scale(family)


def _cholesky_with_jitter(matrix: torch.Tensor) -> torch.Tensor:
    eye = torch.eye(matrix.shape[0], dtype=matrix.dtype, device=matrix.device)
    jitter = 0.0
    for _ in range(6):
        try:
            return torch.linalg.cholesky(matrix + jitter * eye)
        except torch.linalg.LinAlgError:
            jitter = 1e-6 if jitter == 0.0 else jitter * 10.0
    return torch.linalg.cholesky(matrix + jitter * eye)


def generalized_activation_basis(
    target: torch.Tensor,
    anchor: torch.Tensor,
    rank: int,
    *,
    oversample: int = 4,
    eps_scale: float = 0.05,
    use_anchor: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return target-vs-anchor generalized directions in activation space.

    The expensive full generalized eigenproblem is restricted to the top target
    activation subspace.  Inside that small subspace, directions maximize
    target covariance subject to anchor covariance plus a ridge.  This is a
    cheap power-method-like basis for "target active, anchor quiet" directions
    without letting the anchor nullspace dominate.
    """

    if target.ndim != 2 or anchor.ndim != 2:
        raise ValueError("target and anchor activations must be 2D")
    if target.shape[1] != anchor.shape[1]:
        raise ValueError(f"activation width mismatch: target={target.shape[1]} anchor={anchor.shape[1]}")
    width = int(target.shape[1])
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if rank == 0 or width == 0 or target.shape[0] == 0:
        return torch.zeros((rank, width), dtype=torch.float32), torch.zeros((rank,), dtype=torch.float32)

    target = target.cpu().float()
    anchor = anchor.cpu().float()
    target = target - target.mean(dim=0, keepdim=True)
    anchor = anchor - anchor.mean(dim=0, keepdim=True)
    subspace_rank = min(max(rank, rank * oversample), target.shape[0], width)
    if subspace_rank <= 0:
        return torch.zeros((rank, width), dtype=torch.float32), torch.zeros((rank,), dtype=torch.float32)

    _, _, vh_t = torch.linalg.svd(target, full_matrices=False)
    target_basis = vh_t[:subspace_rank].contiguous()
    target_small = target @ target_basis.T
    denom_t = max(int(target_small.shape[0]), 1)
    cov_t = (target_small.T @ target_small) / float(denom_t)
    if use_anchor:
        anchor_small = anchor @ target_basis.T
        denom_a = max(int(anchor_small.shape[0]), 1)
        cov_a = (anchor_small.T @ anchor_small) / float(denom_a)
        ridge = float(torch.diag(cov_a).mean().clamp_min(1e-6).item()) * float(eps_scale)
        cov_a = cov_a + ridge * torch.eye(cov_a.shape[0], dtype=cov_a.dtype)
    else:
        cov_a = torch.eye(cov_t.shape[0], dtype=cov_t.dtype)
    chol = _cholesky_with_jitter(cov_a.double())
    left = torch.linalg.solve_triangular(chol, cov_t.double(), upper=False)
    whitened = torch.linalg.solve_triangular(chol, left.T, upper=False).T
    whitened = 0.5 * (whitened + whitened.T)
    eigvals, eigvecs = torch.linalg.eigh(whitened)
    order = torch.argsort(eigvals, descending=True)
    take = min(rank, int(order.numel()))
    generalized_vecs = torch.linalg.solve_triangular(chol.T, eigvecs[:, order[:take]], upper=True)
    small_vecs = generalized_vecs.T.float()
    basis = small_vecs @ target_basis
    basis = F.normalize(basis, dim=1, eps=1e-6)
    scores = eigvals[order[:take]].float()
    if take < rank:
        basis = F.pad(basis, (0, 0, 0, rank - take))
        scores = F.pad(scores, (0, rank - take))
    return basis.contiguous(), scores.contiguous()


def activation_projected_gaussian_factors(
    module_name: str,
    out_features: int,
    in_features: int,
    rank: int,
    basis: torch.Tensor,
    candidate: Candidate,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Project the same dense Gaussian seed into a task-activation right basis.

    This preserves dense candidate identity: sample the canonical dense Gaussian
    update `G`, then use the right-subspace projection `G V V.T`.  LoRA factors
    are `B = G V` and `A = V.T`.
    """

    if basis.ndim != 2:
        raise ValueError(f"activation basis for {module_name} must be 2D, got shape {tuple(basis.shape)}")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if rank == 0:
        return torch.zeros((0, in_features), dtype=torch.float32), torch.zeros((out_features, 0), dtype=torch.float32)
    k = min(rank, out_features, in_features, int(basis.shape[0]), int(basis.shape[1]))
    if k == 0:
        return torch.zeros((rank, in_features), dtype=torch.float32), torch.zeros((out_features, rank), dtype=torch.float32)

    rows = basis[:k, :in_features].cpu().float().contiguous()
    v, _ = torch.linalg.qr(rows.T, mode="reduced")
    v = v[:, :k].contiguous()

    canonical_name = canonical_module_name(module_name)
    gen = torch.Generator(device="cpu")
    gen.manual_seed((candidate.seed + stable_int(canonical_name + ":dense_gaussian")) % (2**63 - 1))
    scale = activation_right_projected_scale(candidate.family)
    dense = torch.randn((out_features, in_features), generator=gen, dtype=torch.float32)
    dense.mul_(float(candidate.sign) * float(candidate.sigma) * scale)
    b = dense @ v
    a = v.T.contiguous()
    if k < rank:
        a = torch.cat([a, torch.zeros((rank - k, in_features), dtype=a.dtype)], dim=0)
        b = torch.cat([b, torch.zeros((out_features, rank - k), dtype=b.dtype)], dim=1)
    return a.contiguous(), b.contiguous()


def activation_spectral_lora_factors(
    module_name: str,
    out_features: int,
    in_features: int,
    rank: int,
    basis: torch.Tensor,
    candidate: Candidate,
    singular_values: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Use task/anchor activation right-singular vectors as LoRA A directions."""

    if basis.ndim != 2:
        raise ValueError(f"activation basis for {module_name} must be 2D, got shape {tuple(basis.shape)}")
    if rank < 0:
        raise ValueError("rank must be nonnegative")
    if rank == 0:
        return torch.zeros((0, in_features), dtype=torch.float32), torch.zeros((out_features, 0), dtype=torch.float32)
    k = min(rank, out_features, in_features, int(basis.shape[0]), int(basis.shape[1]))
    if k == 0:
        return torch.zeros((rank, in_features), dtype=torch.float32), torch.zeros((out_features, rank), dtype=torch.float32)

    v_rows = basis[:k, :in_features].cpu().float().contiguous()
    # Re-orthogonalize after truncation/padding so singular scales are meaningful.
    v, _ = torch.linalg.qr(v_rows.T, mode="reduced")
    v_rows = v[:, :k].T.contiguous()

    canonical_name = canonical_module_name(module_name)
    gen_u = torch.Generator(device="cpu")
    gen_u.manual_seed((candidate.seed + stable_int(canonical_name + ":activation_spectral_left")) % (2**63 - 1))
    u_raw = torch.randn((out_features, k), generator=gen_u, dtype=torch.float32)
    u, _ = torch.linalg.qr(u_raw, mode="reduced")

    scale = activation_basis_spectral_scale(candidate.family, canonical_name)
    edge = abs(float(candidate.sigma)) * scale * (math.sqrt(float(out_features)) + math.sqrt(float(in_features)))
    singulars = torch.full((k,), edge, dtype=torch.float32)
    if activation_basis_spectral_uses_singular_values(candidate.family) and singular_values is not None:
        values = singular_values[:k].cpu().float().abs()
        if values.shape[0] < k:
            values = F.pad(values, (0, k - values.shape[0]), value=0.0)
        if bool(torch.isfinite(values).all()) and float(values.sum().item()) > 0.0:
            weights = torch.sqrt(values.clamp_min(1e-12) / values.mean().clamp_min(1e-12))
            weights = weights.clamp(0.25, 4.0)
            singulars = edge * weights / weights.mean().clamp_min(1e-12)
    root_s = torch.sqrt(singulars)
    b = float(candidate.sign) * u[:, :k] * root_s.unsqueeze(0)
    a = root_s.unsqueeze(1) * v_rows
    if k < rank:
        a = torch.cat([a, torch.zeros((rank - k, in_features), dtype=a.dtype)], dim=0)
        b = torch.cat([b, torch.zeros((out_features, rank - k), dtype=b.dtype)], dim=1)
    return a.contiguous(), b.contiguous()


def lookup_family_state_spec(family_state: dict | None, lookup_key: str, canonical_name: str):
    if not family_state:
        return None
    if lookup_key in family_state:
        return family_state[lookup_key]
    if canonical_name in family_state:
        return family_state[canonical_name]
    for key, value in family_state.items():
        if canonical_module_name(str(key)) == canonical_name:
            return value
    return None


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
    family_spec = lookup_family_state_spec(family_state, lookup_key, canonical_name)
    if isinstance(family_spec, dict):
        spec = family_spec
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
    if candidate.family.startswith("activation_projected_gaussian_rank_r") or candidate.family.startswith(
        "activation_generalized_projected_gaussian_rank_r"
    ):
        if family_spec is None:
            raise ValueError(f"{candidate.family} requires an activation basis for {lookup_key}")
        basis = family_spec.get("basis") if isinstance(family_spec, dict) else family_spec
        if basis is None:
            raise ValueError(f"{candidate.family} requires a non-empty activation basis for {lookup_key}")
        return activation_projected_gaussian_factors(
            canonical_name,
            int(b_shape[0]),
            int(a_shape[1]),
            rank,
            basis,
            candidate,
        )
    if candidate.family.startswith("activation_spectral_lora") or candidate.family.startswith(
        "activation_generalized_spectral_lora"
    ):
        if family_spec is None:
            raise ValueError(f"{candidate.family} requires an activation basis for {lookup_key}")
        basis = family_spec.get("basis") if isinstance(family_spec, dict) else family_spec
        singular_values = family_spec.get("singular_values") if isinstance(family_spec, dict) else None
        if basis is None:
            raise ValueError(f"{candidate.family} requires a non-empty activation basis for {lookup_key}")
        return activation_spectral_lora_factors(
            canonical_name,
            int(b_shape[0]),
            int(a_shape[1]),
            rank,
            basis,
            candidate,
            singular_values=singular_values,
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
    if candidate.family in {"anzo", "target_svd", "random_ortho", "anzo_random_target"} and family_spec is not None:
        basis = family_spec.get("basis") if isinstance(family_spec, dict) else family_spec
        basis = basis.cpu().float()
        rows = min(a_noise.shape[0], basis.shape[0])
        cols = min(a_noise.shape[1], basis.shape[1])
        a_noise[:rows, :cols] = basis[:rows, :cols]
    elif isinstance(family_spec, dict):
        spec = family_spec
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
    return_metadata: bool = False,
    mode: str = "target_residual_svd",
) -> dict:
    """Build a cheap target-active/anchor-quiet input basis for LoRA A matrices.

    For each LoRA target module, collect input activations on target and anchor
    prompts, then use either the top right singular vectors of target residual
    energy:

        X_target - projection_on_anchor_subspace(X_target)

    or a small generalized target-vs-anchor basis.  The returned tensors have
    shape [rank, in_features], matching LoRA A.
    """

    device = model.device
    handles = []
    acts: dict[str, dict[str, list[torch.Tensor]]] = {}

    def make_hook(name: str):
        def hook(_module, inputs, _output):
            x = inputs[0].detach().float()
            mask = getattr(model, "_anzo_attention_mask", None)
            if mask is not None and mask.shape[:2] == x.shape[:2]:
                x = x[mask.to(device=x.device, dtype=torch.bool)]
            else:
                x = x.reshape(-1, x.shape[-1])
            x = x.cpu()
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
        model._anzo_attention_mask = inputs.get("attention_mask")
        _ = model(**inputs)

    try:
        run(anchor_prompts, "anchor")
        run(target_prompts, "target")
    finally:
        for h in handles:
            h.remove()
        if hasattr(model, "_anzo_phase"):
            delattr(model, "_anzo_phase")
        if hasattr(model, "_anzo_attention_mask"):
            delattr(model, "_anzo_attention_mask")

    state = {}
    for name, by_phase in acts.items():
        target = torch.cat(by_phase.get("target", []), dim=0)
        anchor = torch.cat(by_phase.get("anchor", []), dim=0)
        if mode == "generalized_target_anchor":
            basis, scores = generalized_activation_basis(target, anchor, rank, use_anchor=subtract_anchor)
            if return_metadata:
                state[name] = {
                    "basis": basis.contiguous(),
                    "singular_values": scores.contiguous(),
                    "anchor_rank": 0,
                    "subtract_anchor": bool(subtract_anchor),
                    "target_rows": int(target.shape[0]),
                    "anchor_rows": int(anchor.shape[0]),
                    "mode": mode,
                }
            else:
                state[name] = basis.contiguous()
            continue
        if mode != "target_residual_svd":
            raise ValueError(f"unknown activation basis mode: {mode}")
        target = target - target.mean(dim=0, keepdim=True)
        anchor = anchor - anchor.mean(dim=0, keepdim=True)
        # Anchor subspace from a small SVD. Keep it modest so target signal can survive.
        q = min(rank * 4, anchor.shape[0], anchor.shape[1])
        if subtract_anchor and q > 0:
            _, _, vh_a = torch.linalg.svd(anchor, full_matrices=False)
            anchor_basis = vh_a[:q].T
            target = target - (target @ anchor_basis) @ anchor_basis.T
        _, singular_values, vh_t = torch.linalg.svd(target, full_matrices=False)
        basis = vh_t[:rank]
        if basis.shape[0] < rank:
            basis = F.pad(basis, (0, 0, 0, rank - basis.shape[0]))
        if singular_values.shape[0] < rank:
            singular_values = F.pad(singular_values, (0, rank - singular_values.shape[0]))
        if return_metadata:
            state[name] = {
                "basis": basis.contiguous(),
                "singular_values": singular_values[:rank].contiguous(),
                "anchor_rank": int(q),
                "subtract_anchor": bool(subtract_anchor),
                "target_rows": int(target.shape[0]),
                "anchor_rows": int(anchor.shape[0]),
            }
        else:
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
