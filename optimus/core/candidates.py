from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class SearchCandidate:
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

    for marker in ("model.language_model.layers.", "model.layers."):
        idx = name.find(marker)
        if idx >= 0:
            return name[idx:]
    return name


def candidate_panel(
    family: str,
    population: int,
    sigma: float,
    seed: int,
    antithetic: bool,
    sigma_values: list[float] | None = None,
) -> list[SearchCandidate]:
    rng = np.random.default_rng(seed)
    sigmas = sigma_values or [sigma]
    seeds = [int(x) for x in rng.integers(1, 2**31 - 1, size=population if not antithetic else population // 2)]
    sampled_sigmas = [float(x) for x in rng.choice(sigmas, size=len(seeds), replace=True)]
    out = []
    for candidate_seed, sampled_sigma in zip(seeds, sampled_sigmas):
        out.append(SearchCandidate(family, candidate_seed, sampled_sigma, 1))
        if antithetic:
            out.append(SearchCandidate(family, candidate_seed, sampled_sigma, -1))
    return out[:population]


def parse_candidate_key(key: str) -> SearchCandidate:
    parts = key.split(":")
    if len(parts) != 4:
        raise ValueError(f"invalid candidate key: {key}")
    return SearchCandidate(
        parts[0],
        int(parts[1].removeprefix("seed")),
        float(parts[2].removeprefix("s")),
        int(parts[3].removeprefix("sign")),
    )


def read_candidate_file(path: str) -> list[SearchCandidate]:
    candidates = []
    with Path(path).open() as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                item = line
            key = item.get("candidate") if isinstance(item, dict) else str(item)
            if not key:
                raise ValueError(f"missing candidate key in {path}:{line_no}")
            candidates.append(parse_candidate_key(str(key)))
    if not candidates:
        raise ValueError(f"candidate file is empty: {path}")
    return candidates

__all__ = [
    "SearchCandidate",
    "candidate_panel",
    "canonical_module_name",
    "parse_candidate_key",
    "read_candidate_file",
    "stable_int",
]
