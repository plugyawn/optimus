"""Activation-site projected RandOpt data contracts.

This package owns stable public schemas for subspace search. Runtime backends
must consume these contracts instead of using adapter or LoRA request objects as
candidate identity.
"""

from __future__ import annotations

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
    radius: float
    budget_policy: BudgetPolicy
    rng_version: str = "gaussian_hash_v1"


__all__ = [
    "ActivationSite",
    "BasisKind",
    "BudgetPolicy",
    "ScaleMode",
    "SubspaceCandidate",
    "TargetModule",
]
