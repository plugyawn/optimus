#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch

from optimus.kernels import (
    triton_subspace_add_counter_,
    triton_subspace_add_counter_qv_,
    triton_subspace_expand,
    triton_subspace_expand_counter,
)


@dataclass(frozen=True)
class Shape:
    rows: int
    rank: int
    output_dim: int
    candidates: int

    @property
    def label(self) -> str:
        return f"rows{self.rows}_r{self.rank}_out{self.output_dim}_c{self.candidates}"


DEFAULT_SHAPES = (
    Shape(rows=64, rank=64, output_dim=1024, candidates=16),
    Shape(rows=256, rank=64, output_dim=1024, candidates=16),
    Shape(rows=256, rank=128, output_dim=4096, candidates=16),
    Shape(rows=512, rank=128, output_dim=4096, candidates=16),
)


def parse_shape(text: str) -> Shape:
    parts = [part.strip() for part in text.replace("x", ",").split(",") if part.strip()]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("shape must be rows,rank,output_dim,candidates")
    try:
        rows, rank, output_dim, candidates = (int(part) for part in parts)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"shape contains a non-integer value: {text!r}") from exc
    if min(rows, rank, output_dim, candidates) <= 0:
        raise argparse.ArgumentTypeError("shape values must be positive")
    return Shape(rows=rows, rank=rank, output_dim=output_dim, candidates=candidates)


def _time_cuda(fn: Callable[[], None], *, iters: int) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000.0 / float(iters)


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    columns = [
        "shape",
        "rows",
        "rank",
        "output_dim",
        "candidates",
        "max_inplace_diff",
        "mean_inplace_diff",
        "rmse_inplace_diff",
        "max_expected_abs",
        "max_actual_abs",
        "qx_ms",
        "materialized_expand_ms",
        "counter_expand_ms",
        "counter_expand_plus_add_ms",
        "counter_inplace_add_ms",
        "total_counter_expand_plus_add_ms",
        "total_counter_inplace_ms",
        "inplace_add_speedup",
        "total_inplace_speedup",
        "qx_fraction_of_inplace_total",
        "qv_q_dim",
        "qv_kv_dim",
        "qv_split_inplace_ms",
        "qv_packed_inplace_ms",
        "qv_packed_speedup",
        "qv_max_diff",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plot(path: Path, rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimus-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["shape"]) for row in rows]
    x = list(range(len(rows)))
    width = 0.18
    series = [
        ("Qx", "qx_ms", "#7c3aed"),
        ("counter expand", "counter_expand_ms", "#2563eb"),
        ("expand+add", "counter_expand_plus_add_ms", "#0891b2"),
        ("in-place add", "counter_inplace_add_ms", "#047857"),
        ("Qx+in-place", "total_counter_inplace_ms", "#d97706"),
    ]
    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(rows)), 5.2))
    offsets = [(-2 + idx) * width for idx in range(len(series))]
    for (name, key, color), offset in zip(series, offsets):
        vals = [float(row[key]) for row in rows]
        ax.bar([idx + offset for idx in x], vals, width=width, label=name, color=color)
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("milliseconds / call")
    ax.set_title("Subspace Delta Kernel Ablation")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_speedup(path: Path, rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimus-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["shape"]) for row in rows]
    add_speedups = [float(row["inplace_add_speedup"]) for row in rows]
    total_speedups = [float(row["total_inplace_speedup"]) for row in rows]
    x = list(range(len(rows)))
    fig, ax = plt.subplots(figsize=(max(8.0, 1.8 * len(rows)), 4.8))
    ax.bar([idx - 0.18 for idx in x], add_speedups, width=0.36, label="add-only", color="#047857")
    ax.bar([idx + 0.18 for idx in x], total_speedups, width=0.36, label="Qx+add", color="#d97706")
    ax.axhline(1.0, color="#4b5563", linewidth=1.0, linestyle="--")
    ax.set_xticks(x, labels)
    ax.tick_params(axis="x", rotation=20)
    ax.set_ylabel("speedup vs out-of-place")
    ax.set_title("In-Place Counter Add Speedup")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _plot_qv_speedup(path: Path, rows: list[dict[str, object]]) -> None:
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/optimus-matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [str(row["shape"]) for row in rows]
    split = [float(row["qv_split_inplace_ms"]) for row in rows]
    packed = [float(row["qv_packed_inplace_ms"]) for row in rows]
    speedup = [float(row["qv_packed_speedup"]) for row in rows]
    x = list(range(len(rows)))
    fig, axes = plt.subplots(2, 1, figsize=(max(8.0, 1.8 * len(rows)), 7.0), sharex=True)
    axes[0].bar([idx - 0.18 for idx in x], split, width=0.36, label="split q/v launches", color="#2563eb")
    axes[0].bar([idx + 0.18 for idx in x], packed, width=0.36, label="packed q/v launch", color="#047857")
    axes[0].set_ylabel("milliseconds / call")
    axes[0].set_title("Packed q/v Counter Add")
    axes[0].grid(axis="y", alpha=0.25)
    axes[0].legend(fontsize=8)
    axes[1].bar(x, speedup, width=0.48, color="#d97706")
    axes[1].axhline(1.0, color="#4b5563", linewidth=1.0, linestyle="--")
    axes[1].set_ylabel("speedup")
    axes[1].set_xticks(x, labels)
    axes[1].tick_params(axis="x", rotation=20)
    axes[1].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _benchmark_shape(shape: Shape, *, dtype: torch.dtype, input_dim: int, iters: int, warmup: int) -> dict[str, object]:
    x = torch.randn((shape.rows, input_dim), device="cuda", dtype=dtype)
    basis = torch.randn((shape.rank, input_dim), device="cuda", dtype=dtype)
    base = torch.randn((shape.rows, shape.output_dim), device="cuda", dtype=dtype)
    seeds = torch.arange(1000, 1000 + shape.candidates, device="cuda", dtype=torch.int64)
    signs = torch.where(torch.arange(shape.candidates, device="cuda") % 2 == 0, 1, -1).to(torch.int32)
    mapping = (torch.arange(shape.rows, device="cuda") % shape.candidates).to(torch.int32)
    target_hash = 0x1234ABCD
    v_target_hash = 0x9876DCBA
    beta = 0.03125
    v_beta = 0.046875
    z = x @ basis.T
    materialized_b = torch.randn((shape.candidates, shape.output_dim, shape.rank), device="cuda", dtype=dtype)
    qv_q_dim = shape.output_dim
    qv_kv_dim = max(1, shape.output_dim // 4)
    qv_output_dim = qv_q_dim + 2 * qv_kv_dim
    qv_base = torch.randn((shape.rows, qv_output_dim), device="cuda", dtype=dtype)

    expected = base + triton_subspace_expand_counter(
        z,
        seeds,
        signs,
        mapping,
        target_hash=target_hash,
        beta=beta,
        output_dim=shape.output_dim,
    )
    actual = base.clone()
    triton_subspace_add_counter_(
        z,
        seeds,
        signs,
        mapping,
        actual,
        target_hash=target_hash,
        beta=beta,
        output_dim=shape.output_dim,
    )
    torch.cuda.synchronize()
    diff = (actual - expected).to(dtype=torch.float32)
    abs_diff = diff.abs()
    max_diff = float(abs_diff.max().item())
    mean_diff = float(abs_diff.mean().item())
    rmse_diff = float(torch.sqrt((diff * diff).mean()).item())
    max_expected_abs = float(expected.to(dtype=torch.float32).abs().max().item())
    max_actual_abs = float(actual.to(dtype=torch.float32).abs().max().item())

    qv_expected = qv_base.clone()
    triton_subspace_add_counter_(
        z,
        seeds,
        signs,
        mapping,
        qv_expected,
        target_hash=target_hash,
        beta=beta,
        output_dim=qv_q_dim,
        output_offset=0,
    )
    triton_subspace_add_counter_(
        z,
        seeds,
        signs,
        mapping,
        qv_expected,
        target_hash=v_target_hash,
        beta=v_beta,
        output_dim=qv_kv_dim,
        output_offset=qv_q_dim + qv_kv_dim,
    )
    qv_actual = qv_base.clone()
    triton_subspace_add_counter_qv_(
        z,
        seeds,
        signs,
        mapping,
        qv_actual,
        q_target_hash=target_hash,
        v_target_hash=v_target_hash,
        q_beta=beta,
        v_beta=v_beta,
        q_dim=qv_q_dim,
        kv_dim=qv_kv_dim,
    )
    torch.cuda.synchronize()
    qv_max_diff = float((qv_expected - qv_actual).to(dtype=torch.float32).abs().max().item())

    for _ in range(warmup):
        z = x @ basis.T
        _ = triton_subspace_expand(z, materialized_b, mapping)
        _ = triton_subspace_expand_counter(z, seeds, signs, mapping, target_hash=target_hash, beta=beta, output_dim=shape.output_dim)
        _ = base + triton_subspace_expand_counter(z, seeds, signs, mapping, target_hash=target_hash, beta=beta, output_dim=shape.output_dim)
        triton_subspace_add_counter_(z, seeds, signs, mapping, actual, target_hash=target_hash, beta=beta, output_dim=shape.output_dim)
        triton_subspace_add_counter_qv_(z, seeds, signs, mapping, qv_actual, q_target_hash=target_hash, v_target_hash=v_target_hash, q_beta=beta, v_beta=v_beta, q_dim=qv_q_dim, kv_dim=qv_kv_dim)

    holder: dict[str, torch.Tensor] = {"z": z}

    def qx() -> None:
        holder["z"] = x @ basis.T

    def materialized_expand() -> None:
        holder["materialized"] = triton_subspace_expand(holder["z"], materialized_b, mapping)

    def counter_expand() -> None:
        holder["counter"] = triton_subspace_expand_counter(
            holder["z"],
            seeds,
            signs,
            mapping,
            target_hash=target_hash,
            beta=beta,
            output_dim=shape.output_dim,
        )

    def counter_expand_plus_add() -> None:
        holder["out"] = base + triton_subspace_expand_counter(
            holder["z"],
            seeds,
            signs,
            mapping,
            target_hash=target_hash,
            beta=beta,
            output_dim=shape.output_dim,
        )

    def counter_inplace_add() -> None:
        triton_subspace_add_counter_(
            holder["z"],
            seeds,
            signs,
            mapping,
            actual,
            target_hash=target_hash,
            beta=beta,
            output_dim=shape.output_dim,
        )

    def total_counter_expand_plus_add() -> None:
        local_z = x @ basis.T
        holder["out"] = base + triton_subspace_expand_counter(
            local_z,
            seeds,
            signs,
            mapping,
            target_hash=target_hash,
            beta=beta,
            output_dim=shape.output_dim,
        )

    def total_counter_inplace() -> None:
        local_z = x @ basis.T
        triton_subspace_add_counter_(
            local_z,
            seeds,
            signs,
            mapping,
            actual,
            target_hash=target_hash,
            beta=beta,
            output_dim=shape.output_dim,
        )

    def qv_split_inplace() -> None:
        local_base = qv_base.clone()
        triton_subspace_add_counter_(
            holder["z"],
            seeds,
            signs,
            mapping,
            local_base,
            target_hash=target_hash,
            beta=beta,
            output_dim=qv_q_dim,
            output_offset=0,
        )
        triton_subspace_add_counter_(
            holder["z"],
            seeds,
            signs,
            mapping,
            local_base,
            target_hash=v_target_hash,
            beta=v_beta,
            output_dim=qv_kv_dim,
            output_offset=qv_q_dim + qv_kv_dim,
        )

    def qv_packed_inplace() -> None:
        local_base = qv_base.clone()
        triton_subspace_add_counter_qv_(
            holder["z"],
            seeds,
            signs,
            mapping,
            local_base,
            q_target_hash=target_hash,
            v_target_hash=v_target_hash,
            q_beta=beta,
            v_beta=v_beta,
            q_dim=qv_q_dim,
            kv_dim=qv_kv_dim,
        )

    qx_ms = _time_cuda(qx, iters=iters)
    materialized_expand_ms = _time_cuda(materialized_expand, iters=iters)
    counter_expand_ms = _time_cuda(counter_expand, iters=iters)
    counter_expand_plus_add_ms = _time_cuda(counter_expand_plus_add, iters=iters)
    counter_inplace_add_ms = _time_cuda(counter_inplace_add, iters=iters)
    total_counter_expand_plus_add_ms = _time_cuda(total_counter_expand_plus_add, iters=iters)
    total_counter_inplace_ms = _time_cuda(total_counter_inplace, iters=iters)
    qv_split_inplace_ms = _time_cuda(qv_split_inplace, iters=iters)
    qv_packed_inplace_ms = _time_cuda(qv_packed_inplace, iters=iters)

    return {
        "shape": shape.label,
        "rows": shape.rows,
        "rank": shape.rank,
        "output_dim": shape.output_dim,
        "candidates": shape.candidates,
        "max_inplace_diff": max_diff,
        "mean_inplace_diff": mean_diff,
        "rmse_inplace_diff": rmse_diff,
        "max_expected_abs": max_expected_abs,
        "max_actual_abs": max_actual_abs,
        "qx_ms": qx_ms,
        "materialized_expand_ms": materialized_expand_ms,
        "counter_expand_ms": counter_expand_ms,
        "counter_expand_plus_add_ms": counter_expand_plus_add_ms,
        "counter_inplace_add_ms": counter_inplace_add_ms,
        "total_counter_expand_plus_add_ms": total_counter_expand_plus_add_ms,
        "total_counter_inplace_ms": total_counter_inplace_ms,
        "inplace_add_speedup": counter_expand_plus_add_ms / counter_inplace_add_ms,
        "total_inplace_speedup": total_counter_expand_plus_add_ms / total_counter_inplace_ms,
        "qx_fraction_of_inplace_total": qx_ms / total_counter_inplace_ms,
        "qv_q_dim": qv_q_dim,
        "qv_kv_dim": qv_kv_dim,
        "qv_split_inplace_ms": qv_split_inplace_ms,
        "qv_packed_inplace_ms": qv_packed_inplace_ms,
        "qv_packed_speedup": qv_split_inplace_ms / qv_packed_inplace_ms,
        "qv_max_diff": qv_max_diff,
    }


def run(shapes: Iterable[Shape], *, dtype: torch.dtype, input_dim: int, iters: int, warmup: int) -> dict[str, object]:
    if not torch.cuda.is_available():
        raise RuntimeError("bench_subspace_delta_kernels requires CUDA")
    torch.manual_seed(123)
    rows = [_benchmark_shape(shape, dtype=dtype, input_dim=input_dim, iters=iters, warmup=warmup) for shape in shapes]
    return {
        "schema_version": "subspace_delta_kernel_bench_v1",
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "dtype": str(dtype).replace("torch.", ""),
        "input_dim": int(input_dim),
        "iters": int(iters),
        "warmup": int(warmup),
        "rows": rows,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark Optimus subspace lazy-delta Triton kernels.")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--shape", action="append", type=parse_shape, help="rows,rank,output_dim,candidates")
    parser.add_argument("--iters", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    shapes = tuple(args.shape or DEFAULT_SHAPES)
    if args.input_dim <= 0:
        raise ValueError("--input-dim must be positive")
    payload = run(shapes, dtype=dtype, input_dim=args.input_dim, iters=args.iters, warmup=args.warmup)
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "kernel_ablation_summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    rows = list(payload["rows"])
    _write_csv(args.out / "kernel_ablation.csv", rows)
    _plot(args.out / "kernel_ablation_latency.png", rows)
    _plot_speedup(args.out / "kernel_ablation_speedup.png", rows)
    _plot_qv_speedup(args.out / "kernel_ablation_qv_speedup.png", rows)
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
