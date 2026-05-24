from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any

from optimus.core.perturbations import PerturbationSpec, perturbation_panel


@dataclass(frozen=True)
class EvaluationRecord:
    perturbation: PerturbationSpec
    metric: str
    value: float
    split: str = "screen"
    details: dict[str, Any] | None = None

    @property
    def candidate(self) -> str:
        return self.perturbation.key

    def to_record(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate,
            "method": self.perturbation.method,
            "family": self.perturbation.family,
            "seed": self.perturbation.seed,
            "sigma": self.perturbation.sigma,
            "sign": self.perturbation.sign,
            "rank": self.perturbation.rank,
            "targets": list(self.perturbation.targets),
            "metric": self.metric,
            "value": self.value,
            "split": self.split,
            **(self.details or {}),
        }


@dataclass(frozen=True)
class SearchResult:
    records: tuple[EvaluationRecord, ...]
    metric: str
    maximize: bool = True

    @property
    def best(self) -> EvaluationRecord | None:
        if not self.records:
            return None
        return sorted_records(self.records, metric=self.metric, maximize=self.maximize)[0]

    def top_k(self, k: int) -> tuple[EvaluationRecord, ...]:
        return tuple(sorted_records(self.records, metric=self.metric, maximize=self.maximize)[:k])


def sorted_records(
    records: Iterable[EvaluationRecord],
    *,
    metric: str,
    maximize: bool = True,
) -> list[EvaluationRecord]:
    filtered = [record for record in records if record.metric == metric]
    return sorted(filtered, key=lambda record: record.value, reverse=maximize)


def select_top_k(
    rows: Sequence[dict[str, Any]],
    *,
    score_column: str,
    k: int,
    maximize: bool = True,
) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: float(row[score_column]), reverse=maximize)[:k]


class ZerothOrderStudy:
    """Small ask/tell wrapper for backend-agnostic perturbation screens."""

    def __init__(
        self,
        *,
        method: str,
        family: str,
        population: int,
        sigma: float,
        seed: int,
        antithetic: bool = False,
        sigma_values: list[float] | None = None,
        rank: int | None = None,
        targets: Sequence[str] | str | None = None,
        metric: str = "exact_mean",
        maximize: bool = True,
    ):
        self.method = method
        self.family = family
        self.metric = metric
        self.maximize = maximize
        self._perturbations = tuple(
            perturbation_panel(
                method,
                family,
                population,
                sigma,
                seed,
                antithetic,
                sigma_values,
                rank=rank,
                targets=targets,
            )
        )
        self._records: list[EvaluationRecord] = []

    def ask(self) -> tuple[PerturbationSpec, ...]:
        return self._perturbations

    def tell(
        self,
        perturbation: PerturbationSpec,
        value: float,
        *,
        split: str = "screen",
        details: dict[str, Any] | None = None,
    ) -> EvaluationRecord:
        record = EvaluationRecord(
            perturbation=perturbation,
            metric=self.metric,
            value=float(value),
            split=split,
            details=details,
        )
        self._records.append(record)
        return record

    def result(self) -> SearchResult:
        return SearchResult(tuple(self._records), metric=self.metric, maximize=self.maximize)


__all__ = [
    "EvaluationRecord",
    "SearchResult",
    "ZerothOrderStudy",
    "select_top_k",
    "sorted_records",
]
