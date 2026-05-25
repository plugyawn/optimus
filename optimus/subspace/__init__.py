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


@dataclass(frozen=True)
class ActivationSite:
    site_id: str
    input_dim: int
    basis_kind: BasisKind
    requested_rank: int
    effective_rank: int


@dataclass(frozen=True)
class TargetModule:
    target_id: str
    activation_site_id: str
    output_dim: int


@dataclass(frozen=True)
class SubspaceCandidate:
    candidate_id: str
    direction_seed: int
    sign: int
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
    rng_version: str = "gaussian_hash_v1"


def canonical_target_id(target_id: str) -> str:
    canonical = target_id.strip()
    if not canonical:
        raise ValueError("target_id must be nonempty")
    if "\x00" in canonical:
        raise ValueError("target_id must not contain NUL bytes")
    return canonical


def gaussian_hash_v1(
    *,
    direction_seed: int,
    target_id: str,
    output_index: int,
    basis_index: int,
    salt: str = "",
    sign: int = 1,
    rng_version: str = "gaussian_hash_v1",
) -> float:
    if rng_version != "gaussian_hash_v1":
        raise ValueError(f"unsupported rng_version {rng_version!r}")
    if sign not in {-1, 1}:
        raise ValueError("sign must be -1 or 1")
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
    return float(sign) * value


__all__ = [
    "ActivationSite",
    "BasisKind",
    "BudgetPolicy",
    "canonical_target_id",
    "gaussian_hash_v1",
    "ScaleMode",
    "SubspaceCandidate",
    "TargetModule",
]
