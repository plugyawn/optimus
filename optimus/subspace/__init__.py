"""Activation-site projected RandOpt data contracts.

This package owns stable public schemas for subspace search. Runtime backends
must consume these contracts instead of using adapter or LoRA request objects as
candidate identity.
"""

from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from typing import Any, Literal


BasisKind = Literal["activation-svd", "random-orthonormal", "shuffled-activation-svd"]
ScaleMode = Literal["projected-dense", "relative-output-rms"]
BudgetPolicy = Literal["raw-dense", "per-target-equal", "per-layer-equal", "per-block-equal", "custom-json"]
CandidateSign = Literal["+", "-"]
RandomFieldVersion = Literal["gaussian_hash_v1", "torch_generator_field_v1", "counter_gaussian_v1"]


@dataclass(frozen=True)
class ActivationSite:
    site_id: str
    architecture_family: str
    layer_index: int
    block_path: str
    read_tensor_path: str
    hook_point: str
    norm_position: str
    shape_convention: str
    runtime_dtype: str
    accumulation_dtype: str
    tensor_parallel_sharding_policy: str
    target_module_ids: tuple[str, ...]
    calibration_prompt_ids_hash: str
    calibration_decode_config_hash: str
    basis_control_seed: int | None
    transductive: bool
    input_dim: int
    basis_kind: BasisKind
    requested_rank: int
    effective_rank: int
    basis_tensor_key: str
    basis_tensor_sha256: str
    singular_values: tuple[float, ...]
    captured_energy: float
    prefill_captured_energy: float | None
    decode_captured_energy: float | None
    H_s: float
    A_s: float
    orthonormality_error: float
    gram_error: float
    num_calibration_tokens: int


@dataclass(frozen=True)
class TargetModule:
    target_id: str
    activation_site_id: str
    output_dim: int


@dataclass(frozen=True)
class SubspaceState:
    basis_hash: str
    target_set_hash: str
    basis_collection_config_hash: str
    activation_sites: tuple[ActivationSite, ...]
    targets: tuple[TargetModule, ...]


@dataclass(frozen=True)
class SubspaceCandidate:
    candidate_id: str
    direction_seed: int
    sign: CandidateSign
    basis_hash: str
    target_set_hash: str
    scale_mode: ScaleMode
    rho_or_sigma_w: float
    budget_policy: BudgetPolicy
    budget_hash: str
    runtime_dtype: str
    radius_index: int
    target_preset: str
    basis_rank: int
    shard_id: str
    shard_population_start: int
    shard_population_end: int
    worker_id: str
    device_id: str
    prompt_scoring_config_hash: str
    rng_version: str = "gaussian_hash_v1"


def canonical_target_id(target_id: str) -> str:
    canonical = target_id.strip()
    if not canonical:
        raise ValueError("target_id must be nonempty")
    if "\x00" in canonical:
        raise ValueError("target_id must not contain NUL bytes")
    return canonical


def sign_value(sign: CandidateSign | int) -> int:
    if sign in {"+", 1}:
        return 1
    if sign in {"-", -1}:
        return -1
    raise ValueError("sign must be '+', '-', -1, or 1")


def gaussian_hash_v1(
    *,
    direction_seed: int,
    target_id: str,
    output_index: int,
    basis_index: int,
    salt: str = "",
    sign: CandidateSign | int = "+",
    rng_version: str = "gaussian_hash_v1",
) -> float:
    if rng_version != "gaussian_hash_v1":
        raise ValueError(f"unsupported rng_version {rng_version!r}")
    resolved_sign = sign_value(sign)
    fields = [
        rng_version,
        str(int(direction_seed)),
        canonical_target_id(target_id),
        str(int(output_index)),
        str(int(basis_index)),
        salt,
    ]
    payload = "\x00".join(fields).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    word0, word1 = struct.unpack("<QQ", digest[:16])
    denom = float(1 << 64)
    u0 = (float(word0) + 0.5) / denom
    u1 = (float(word1) + 0.5) / denom
    value = math.sqrt(-2.0 * math.log(u0)) * math.cos(2.0 * math.pi * u1)
    return float(resolved_sign) * value


def stable_u32(text: str) -> int:
    """Stable low-32-bit hash for kernel-side counter RNG metadata."""

    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "little")


def _u32(value: int) -> int:
    return int(value) & 0xFFFFFFFF


def _mix_u32(value: int) -> int:
    x = _u32(value)
    x ^= x >> 16
    x = _u32(x * 0x7FEB352D)
    x ^= x >> 15
    x = _u32(x * 0x846CA68B)
    x ^= x >> 16
    return _u32(x)


def counter_gaussian_v1(
    *,
    direction_seed: int,
    target_hash: int,
    output_index: int,
    basis_index: int,
    sign: CandidateSign | int = "+",
) -> float:
    """Kernel-friendly stateless normal field used by fused lazy kernels."""

    resolved_sign = sign_value(sign)
    key = (
        _u32(direction_seed)
        ^ _u32(target_hash)
        ^ _u32(int(output_index) * 0x9E3779B9)
        ^ _u32(int(basis_index) * 0x85EBCA6B)
    )
    h0 = _mix_u32(key)
    h1 = _mix_u32(key ^ 0xD1B54A32)
    u0 = max((float(h0) + 0.5) / 4294967296.0, 1e-12)
    u1 = (float(h1) + 0.5) / 4294967296.0
    value = math.sqrt(-2.0 * math.log(u0)) * math.cos(2.0 * math.pi * u1)
    return float(resolved_sign) * value


def random_field_tensor(
    *,
    direction_seed: int,
    sign: CandidateSign | int,
    target_id: str,
    output_dim: int,
    basis_rank: int,
    rng_version: str = "gaussian_hash_v1",
    dtype: Any | None = None,
    device: Any | None = None,
    source_rank: int | None = None,
    output_offset: int = 0,
    salt: str = "",
) -> torch.Tensor:
    """Materialize a candidate random field for compatibility paths.

    Production lazy kernels should avoid this allocation. The function exists
    so adapter export, torch reference execution, and bridge backends all agree
    on the same candidate law.
    """

    import torch

    dtype = torch.float32 if dtype is None else dtype

    output_dim = int(output_dim)
    basis_rank = int(basis_rank)
    source_rank = int(source_rank or basis_rank)
    if output_dim < 0 or basis_rank < 0 or source_rank < basis_rank:
        raise ValueError("invalid random field shape")
    target_id = canonical_target_id(target_id)
    if rng_version == "torch_generator_field_v1":
        if salt:
            raise ValueError("torch_generator_field_v1 does not support salt")
        seed_payload = f"{int(direction_seed)}\0{target_id}\0torch_generator_field_v1".encode("utf-8")
        seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) % (2**63 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        field = torch.randn((output_dim, source_rank), generator=gen, dtype=torch.float32)
        if sign_value(sign) < 0:
            field = -field
        field = field[:, :basis_rank].contiguous()
    elif rng_version == "gaussian_hash_v1":
        values = [
            gaussian_hash_v1(
                direction_seed=direction_seed,
                target_id=target_id,
                output_index=int(output_offset) + out_idx,
                basis_index=basis_idx,
                sign=sign,
                rng_version="gaussian_hash_v1",
                salt=salt,
            )
            for out_idx in range(output_dim)
            for basis_idx in range(basis_rank)
        ]
        field = torch.tensor(values, dtype=torch.float32).reshape(output_dim, basis_rank)
    elif rng_version == "counter_gaussian_v1":
        if salt:
            raise ValueError("counter_gaussian_v1 does not support salt")
        target_hash = stable_u32(target_id)
        values = [
            counter_gaussian_v1(
                direction_seed=direction_seed,
                target_hash=target_hash,
                output_index=int(output_offset) + out_idx,
                basis_index=basis_idx,
                sign=sign,
            )
            for out_idx in range(output_dim)
            for basis_idx in range(basis_rank)
        ]
        field = torch.tensor(values, dtype=torch.float32).reshape(output_dim, basis_rank)
    else:
        raise ValueError(f"unsupported rng_version {rng_version!r}")
    if device is not None:
        field = field.to(device)
    if dtype != torch.float32:
        field = field.to(dtype=dtype)
    return field.contiguous()


__all__ = [
    "ActivationSite",
    "BasisKind",
    "BudgetPolicy",
    "CandidateSign",
    "canonical_target_id",
    "counter_gaussian_v1",
    "gaussian_hash_v1",
    "random_field_tensor",
    "RandomFieldVersion",
    "ScaleMode",
    "sign_value",
    "stable_u32",
    "SubspaceCandidate",
    "SubspaceState",
    "TargetModule",
]
