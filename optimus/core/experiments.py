from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal


RunStatus = Literal["queued", "running", "skipped", "completed", "failed", "dry_run"]


@dataclass(frozen=True)
class ExperimentKey:
    """Stable coordinates for one experiment point in a GPU sweep."""

    name: str
    kind: str
    method: str
    backend: str
    population: int | None = None
    model: str | None = None
    seed: int | None = None
    extra: Mapping[str, Any] | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "kind": self.kind,
            "method": self.method,
            "backend": self.backend,
            "population": self.population,
            "model": self.model,
            "seed": self.seed,
            "extra": dict(self.extra or {}),
        }


@dataclass(frozen=True)
class RunRecord:
    """Machine-readable run status for resumable GPU execution."""

    key: ExperimentKey
    output_path: Path
    command: tuple[str, ...]
    status: RunStatus
    marker: Path | None = None
    started_at: float | None = None
    finished_at: float | None = None
    returncode: int | None = None
    error: str | None = None

    @property
    def elapsed_s(self) -> float | None:
        if self.started_at is None or self.finished_at is None:
            return None
        return max(0.0, self.finished_at - self.started_at)

    def to_record(self) -> dict[str, Any]:
        return {
            "key": self.key.to_record(),
            "name": self.key.name,
            "kind": self.key.kind,
            "method": self.key.method,
            "backend": self.key.backend,
            "population": self.key.population,
            "model": self.key.model,
            "seed": self.key.seed,
            "output_path": str(self.output_path),
            "command": list(self.command),
            "status": self.status,
            "marker": None if self.marker is None else str(self.marker),
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "elapsed_s": self.elapsed_s,
            "returncode": self.returncode,
            "error": self.error,
        }


@dataclass(frozen=True)
class ThroughputRecord:
    """Common systems metrics emitted by backends and reports."""

    backend: str
    method: str
    candidate_sec: float | None = None
    prompts_per_sec: float | None = None
    tokens_per_sec: float | None = None
    eval_elapsed_s: float | None = None
    gpu_count: int | None = None
    batch_size: int | None = None
    tensor_parallel_size: int | None = None
    gpu_memory_utilization: float | None = None

    def to_record(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "method": self.method,
            "candidate_sec": self.candidate_sec,
            "prompts_per_sec": self.prompts_per_sec,
            "tokens_per_sec": self.tokens_per_sec,
            "eval_elapsed_s": self.eval_elapsed_s,
            "gpu_count": self.gpu_count,
            "batch_size": self.batch_size,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
        }


def now_s() -> float:
    return time.time()


def status_record(
    *,
    key: ExperimentKey,
    output_path: Path,
    command: Sequence[str],
    status: RunStatus,
    marker: Path | None = None,
    started_at: float | None = None,
    finished_at: float | None = None,
    returncode: int | None = None,
    error: str | None = None,
) -> dict[str, Any]:
    return RunRecord(
        key=key,
        output_path=output_path,
        command=tuple(command),
        status=status,
        marker=marker,
        started_at=started_at,
        finished_at=finished_at,
        returncode=returncode,
        error=error,
    ).to_record()


__all__ = [
    "ExperimentKey",
    "RunRecord",
    "RunStatus",
    "ThroughputRecord",
    "now_s",
    "status_record",
]
