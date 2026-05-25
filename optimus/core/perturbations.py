from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, Protocol

import numpy as np


PerturbationMethod = Literal["dense", "lora", "subspace"]
VALID_PERTURBATION_METHODS = frozenset({"dense", "lora", "subspace"})


def stable_int(text: str) -> int:
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def canonical_module_name(name: str) -> str:
    """Return the bare transformer module path shared by dense and adapter paths."""

    for marker in ("model.language_model.layers.", "model.layers."):
        idx = name.find(marker)
        if idx >= 0:
            return name[idx:]
    return name


def _normalize_method(method: str) -> PerturbationMethod:
    if method not in VALID_PERTURBATION_METHODS:
        raise ValueError(f"perturbation method must be one of {sorted(VALID_PERTURBATION_METHODS)}, got {method!r}")
    return method  # type: ignore[return-value]


def _normalize_family(family: str) -> str:
    return family


def _is_subspace_family_name(family: str) -> bool:
    return family.startswith("subspace_gaussian_rank_r")


def _normalize_targets(targets: Sequence[str] | str | None) -> tuple[str, ...]:
    if targets is None:
        return ()
    if isinstance(targets, str):
        return tuple(item.strip() for item in targets.split(",") if item.strip())
    return tuple(str(item).strip() for item in targets if str(item).strip())


@dataclass(frozen=True, init=False)
class PerturbationSpec:
    """Stable identity for a zeroth-order perturbation candidate.

    `method` states whether the perturbation is a dense update, a low-rank
    adapter update, or an implicit activation-subspace update. Public manifest
    keys are method-qualified; legacy four-field keys are rejected.
    """

    family: str
    seed: int
    sigma: float
    sign: int
    method: PerturbationMethod
    rank: int | None
    targets: tuple[str, ...]

    def __init__(
        self,
        family: str,
        seed: int,
        sigma: float,
        sign: int = 1,
        *,
        method: str = "lora",
        rank: int | None = None,
        targets: Sequence[str] | str | None = None,
    ) -> None:
        family = _normalize_family(str(family))
        if not family:
            raise ValueError("perturbation family must be non-empty")
        sigma = float(sigma)
        if not math.isfinite(sigma):
            raise ValueError(f"perturbation sigma must be finite, got {sigma!r}")
        sign = int(sign)
        if sign not in {-1, 1}:
            raise ValueError(f"perturbation sign must be -1 or 1, got {sign!r}")
        if rank is not None:
            rank = int(rank)
            if rank <= 0:
                raise ValueError(f"perturbation rank must be positive when set, got {rank!r}")
        object.__setattr__(self, "family", family)
        object.__setattr__(self, "seed", int(seed))
        object.__setattr__(self, "sigma", sigma)
        object.__setattr__(self, "sign", sign)
        if method == "dense" and family != "dense_gaussian":
            raise ValueError("dense perturbations currently require family='dense_gaussian'")
        if family == "dense_gaussian" and method != "dense":
            raise ValueError("dense_gaussian perturbations require method='dense'")
        method = _normalize_method(str(method))
        if method == "subspace" and not _is_subspace_family_name(family):
            raise ValueError("subspace perturbations require a subspace family")
        object.__setattr__(self, "method", method)
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "targets", _normalize_targets(targets))

    @property
    def key(self) -> str:
        parts = [self.method, self.family, f"seed{self.seed}", f"s{self.sigma:g}", f"sign{self.sign}"]
        if self.rank is not None:
            parts.append(f"r{self.rank}")
        if self.targets:
            parts.append("t" + ",".join(self.targets))
        return ":".join(parts)

    def with_method(self, method: str) -> "PerturbationSpec":
        return PerturbationSpec(
            self.family,
            self.seed,
            self.sigma,
            self.sign,
            method=method,
            rank=self.rank,
            targets=self.targets,
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "method": self.method,
            "family": self.family,
            "seed": self.seed,
            "sigma": self.sigma,
            "sign": self.sign,
            "rank": self.rank,
            "targets": list(self.targets),
        }

    @classmethod
    def from_record(cls, record: Mapping[str, Any], *, default_method: str = "lora") -> "PerturbationSpec":
        method = str(record.get("method") or default_method)
        family = record.get("family")
        seed = record.get("seed")
        sigma = record.get("sigma")
        sign = record.get("sign", 1)
        if family is None or seed is None or sigma is None:
            key = record.get("key") or record.get("candidate")
            if key is None:
                raise ValueError(f"missing perturbation identity fields: {record!r}")
            else:
                parsed = parse_perturbation_key(str(key))
            rank = record.get("rank")
            targets = record.get("targets")
            if rank in {"", None} and targets is None:
                return parsed
            return cls(
                parsed.family,
                parsed.seed,
                parsed.sigma,
                parsed.sign,
                method=parsed.method,
                rank=parsed.rank if rank in {"", None} else int(rank),
                targets=parsed.targets if targets is None else targets,
            )
        return cls(
            str(family),
            int(seed),
            float(sigma),
            int(sign),
            method=method,
            rank=None if record.get("rank") in {"", None} else int(record["rank"]),
            targets=record.get("targets"),
        )


@dataclass(frozen=True)
class PerturbationScore:
    perturbation: PerturbationSpec | None
    split: str
    metric: str
    value: float
    details: Mapping[str, Any] | None = None

    @property
    def candidate(self) -> str:
        return "base" if self.perturbation is None else self.perturbation.key


class PerturbationEvaluator(Protocol):
    """Backend contract for scoring a base model or one perturbation."""

    def evaluate(self, perturbation: PerturbationSpec | None, *, split: str) -> PerturbationScore:
        ...


class PerturbationMaterializer(Protocol):
    """Backend contract for turning a perturbation into runnable model state."""

    method: PerturbationMethod

    def materialize(self, perturbation: PerturbationSpec, output_path: Path) -> Mapping[str, Any]:
        ...


def parse_perturbation_key(key: str) -> PerturbationSpec:
    parts = key.split(":")
    rank = None
    targets: tuple[str, ...] = ()
    if len(parts) >= 5:
        method, family, seed_text, sigma_text, sign_text, *extras = parts
        for item in extras:
            if item.startswith("r"):
                rank = int(item.removeprefix("r"))
            elif item.startswith("t"):
                targets = tuple(target for target in item.removeprefix("t").split(",") if target)
            else:
                raise ValueError(f"invalid perturbation key component {item!r}: {key}")
    else:
        if len(parts) == 4:
            raise ValueError("legacy perturbation keys are unsupported; use method-qualified keys")
        raise ValueError(f"invalid perturbation key: {key}")
    return PerturbationSpec(
        family,
        int(seed_text.removeprefix("seed")),
        float(sigma_text.removeprefix("s")),
        int(sign_text.removeprefix("sign")),
        method=method,
        rank=rank,
        targets=targets,
    )


def perturbation_panel(
    method: str,
    family: str,
    population: int,
    sigma: float,
    seed: int,
    antithetic: bool,
    sigma_values: list[float] | None = None,
    *,
    rank: int | None = None,
    targets: Sequence[str] | str | None = None,
) -> list[PerturbationSpec]:
    population = int(population)
    if population < 0:
        raise ValueError(f"population must be non-negative, got {population!r}")
    rng = np.random.default_rng(seed)
    sigmas = sigma_values or [sigma]
    seed_count = population if not antithetic else (population + 1) // 2
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=seed_count)]
    sampled_sigmas = [float(x) for x in rng.choice(sigmas, size=len(seeds), replace=True)]
    out = []
    for candidate_seed, sampled_sigma in zip(seeds, sampled_sigmas):
        out.append(PerturbationSpec(family, candidate_seed, sampled_sigma, 1, method=method, rank=rank, targets=targets))
        if antithetic:
            out.append(PerturbationSpec(family, candidate_seed, sampled_sigma, -1, method=method, rank=rank, targets=targets))
    return out[:population]


def read_perturbation_file(path: str | Path, *, default_method: str = "lora") -> list[PerturbationSpec]:
    perturbations: list[PerturbationSpec] = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item: str | Mapping[str, Any] = json.loads(line)
            except json.JSONDecodeError:
                item = line
            try:
                if isinstance(item, Mapping):
                    perturbations.append(PerturbationSpec.from_record(item, default_method=default_method))
                else:
                    perturbations.append(parse_perturbation_key(str(item)))
            except ValueError as exc:
                raise ValueError(f"{path}:{line_no}: {exc}") from exc
    if not perturbations:
        raise ValueError(f"perturbation file is empty: {path}")
    return perturbations


def write_perturbation_file(path: str | Path, perturbations: Iterable[PerturbationSpec]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        for perturbation in perturbations:
            f.write(json.dumps(perturbation.to_record(), sort_keys=True) + "\n")


def require_method(perturbations: Iterable[PerturbationSpec], method: str, *, backend: str) -> None:
    expected = _normalize_method(method)
    bad = [perturbation.key for perturbation in perturbations if perturbation.method != expected]
    if bad:
        sample = ", ".join(bad[:3])
        raise ValueError(f"{backend} only supports {expected!r} perturbations; incompatible candidates: {sample}")


def require_materialization_contract(
    perturbations: Iterable[PerturbationSpec],
    *,
    backend: str,
    method: str,
    rank: int | None = None,
    targets: Sequence[str] | str | None = None,
    require_explicit: bool = False,
) -> None:
    items = list(perturbations)
    require_method(items, method, backend=backend)
    expected_targets = _normalize_targets(targets)
    errors = []
    for perturbation in items:
        if method == "dense" and perturbation.family != "dense_gaussian":
            errors.append(f"{perturbation.key}: dense method requires dense_gaussian family")
        if require_explicit and rank is not None and perturbation.rank is None:
            errors.append(f"{perturbation.key}: missing rank")
        if require_explicit and expected_targets and not perturbation.targets:
            errors.append(f"{perturbation.key}: missing targets")
        if perturbation.rank is not None and rank is not None and int(perturbation.rank) != int(rank):
            errors.append(f"{perturbation.key}: rank {perturbation.rank} != requested {rank}")
        if perturbation.targets and expected_targets and perturbation.targets != expected_targets:
            errors.append(f"{perturbation.key}: targets {list(perturbation.targets)} != requested {list(expected_targets)}")
    if errors:
        raise ValueError(f"{backend} perturbation materialization contract failed: " + "; ".join(errors[:5]))


__all__ = [
    "PerturbationEvaluator",
    "PerturbationMaterializer",
    "PerturbationMethod",
    "PerturbationScore",
    "PerturbationSpec",
    "canonical_module_name",
    "parse_perturbation_key",
    "perturbation_panel",
    "read_perturbation_file",
    "require_materialization_contract",
    "require_method",
    "stable_int",
    "write_perturbation_file",
]
