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
from typing import Literal


BasisKind = Literal["activation-svd", "random-orthonormal", "shuffled-activation-svd"]
ScaleMode = Literal["projected-dense", "relative-output-rms"]
BudgetPolicy = Literal["raw-dense", "per-target-equal", "per-layer-equal", "per-block-equal", "custom-json"]
CandidateSign = Literal["+", "-"]


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


__all__ = [
    "ActivationSite",
    "BasisKind",
    "BudgetPolicy",
    "CandidateSign",
    "canonical_target_id",
    "gaussian_hash_v1",
    "ScaleMode",
    "sign_value",
    "SubspaceCandidate",
    "SubspaceState",
    "TargetModule",
]
