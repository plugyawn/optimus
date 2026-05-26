from __future__ import annotations

import torch

try:  # pragma: no cover - availability is environment dependent.
    import triton
    import triton.language as tl
except Exception:  # pragma: no cover - exercised on non-CUDA local hosts.
    triton = None
    tl = None


if triton is not None and tl is not None:  # pragma: no branch - import guard.

    @triton.jit
    def _mix_u32(x):
        x = x.to(tl.uint32)
        x = x ^ (x >> 16)
        x = x * 0x7FEB352D
        x = x ^ (x >> 15)
        x = x * 0x846CA68B
        x = x ^ (x >> 16)
        return x

    @triton.jit
    def _normal_from_counter(seed, target_hash, output_index, basis_index):
        output_u = output_index.to(tl.uint32)
        target_u = (output_u * 0 + target_hash).to(tl.uint32)
        basis_u = (output_u * 0 + basis_index).to(tl.uint32)
        key = (
            seed.to(tl.uint32)
            ^ target_u
            ^ (output_u * 0x9E3779B9)
            ^ (basis_u * 0x85EBCA6B)
        )
        h0 = _mix_u32(key)
        h1 = _mix_u32(key ^ 0xD1B54A32)
        u0 = tl.maximum((h0.to(tl.float32) + 0.5) * 2.3283064365386963e-10, 1.0e-12)
        u1 = (h1.to(tl.float32) + 0.5) * 2.3283064365386963e-10
        return tl.sqrt(-2.0 * tl.log(u0)) * tl.cos(6.283185307179586 * u1)

    @triton.jit
    def _subspace_expand_kernel(
        z_ptr,
        b_ptr,
        row_candidate_ptr,
        out_ptr,
        rows: tl.constexpr,
        output_dim: tl.constexpr,
        rank: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_n < rows
        col_mask = offs_m < output_dim
        candidate = tl.load(row_candidate_ptr + offs_n, mask=row_mask, other=0).to(tl.int64)

        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for r in range(rank):
            z = tl.load(z_ptr + offs_n * rank + r, mask=row_mask, other=0.0).to(tl.float32)
            b = tl.load(
                b_ptr + candidate[:, None] * output_dim * rank + offs_m[None, :] * rank + r,
                mask=row_mask[:, None] & col_mask[None, :],
                other=0.0,
            ).to(tl.float32)
            acc += z[:, None] * b

        tl.store(out_ptr + offs_n[:, None] * output_dim + offs_m[None, :], acc, mask=row_mask[:, None] & col_mask[None, :])

    @triton.jit
    def _subspace_counter_kernel(
        z_ptr,
        seed_ptr,
        sign_ptr,
        row_candidate_ptr,
        out_ptr,
        rows: tl.constexpr,
        output_dim: tl.constexpr,
        rank: tl.constexpr,
        target_hash: tl.constexpr,
        beta: tl.constexpr,
        output_offset: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_m = tl.program_id(1)
        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_n < rows
        col_mask = offs_m < output_dim
        candidate = tl.load(row_candidate_ptr + offs_n, mask=row_mask, other=0).to(tl.int64)
        seed = tl.load(seed_ptr + candidate, mask=row_mask, other=0).to(tl.uint32)
        sign = tl.load(sign_ptr + candidate, mask=row_mask, other=1).to(tl.float32)

        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for r in range(rank):
            z = tl.load(z_ptr + offs_n * rank + r, mask=row_mask, other=0.0).to(tl.float32)
            normal = _normal_from_counter(seed[:, None], target_hash, output_offset + offs_m[None, :], r)
            acc += z[:, None] * normal * sign[:, None] * beta

        tl.store(out_ptr + offs_n[:, None] * output_dim + offs_m[None, :], acc, mask=row_mask[:, None] & col_mask[None, :])
else:
    _subspace_expand_kernel = None
    _subspace_counter_kernel = None


def triton_subspace_expand(
    z: torch.Tensor,
    b_stack: torch.Tensor,
    row_candidate_indices: torch.Tensor,
    *,
    block_n: int = 16,
    block_m: int = 32,
) -> torch.Tensor:
    """Candidate-routed expand kernel for ``delta = B_c @ z``.

    ``z`` is shaped ``[rows, rank]`` and ``b_stack`` is shaped
    ``[candidates, output_dim, rank]``. ``row_candidate_indices`` maps each row
    to a candidate index in ``b_stack``. The field stack is intentionally still
    materialized in this v0 kernel so it can replay existing
    ``torch_generator_field_v1`` candidates exactly; the later production
    kernel should replace the stack with deterministic random-field generation.
    """

    if triton is None or _subspace_expand_kernel is None:
        raise RuntimeError("OPTIMUS_LAZY_DELTA_BACKEND=triton requires triton")
    if not z.is_cuda or not b_stack.is_cuda:
        raise RuntimeError("triton_subspace_expand requires CUDA tensors")
    if z.ndim != 2 or b_stack.ndim != 3:
        raise ValueError(f"expected z [rows, rank] and b_stack [candidates, output, rank], got {tuple(z.shape)} and {tuple(b_stack.shape)}")
    rows, rank = int(z.shape[0]), int(z.shape[1])
    _, output_dim, b_rank = (int(x) for x in b_stack.shape)
    if b_rank != rank:
        raise ValueError(f"rank mismatch: z={tuple(z.shape)} b_stack={tuple(b_stack.shape)}")
    mapping = row_candidate_indices.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if int(mapping.numel()) != rows:
        raise ValueError(f"row_candidate_indices length {int(mapping.numel())} does not match rows {rows}")
    z_contig = z.contiguous()
    b_contig = b_stack.contiguous()
    out = torch.empty((rows, output_dim), device=z.device, dtype=z.dtype)
    grid = (triton.cdiv(rows, int(block_n)), triton.cdiv(output_dim, int(block_m)))
    _subspace_expand_kernel[grid](
        z_contig,
        b_contig,
        mapping,
        out,
        rows,
        output_dim,
        rank,
        BLOCK_N=int(block_n),
        BLOCK_M=int(block_m),
        num_warps=4,
    )
    return out


def triton_subspace_expand_counter(
    z: torch.Tensor,
    direction_seeds: torch.Tensor,
    signs: torch.Tensor,
    row_candidate_indices: torch.Tensor,
    *,
    target_hash: int,
    beta: float,
    output_dim: int,
    output_offset: int = 0,
    block_n: int = 16,
    block_m: int = 32,
) -> torch.Tensor:
    """Stateless candidate-routed expand for ``counter_gaussian_v1`` fields."""

    if triton is None or _subspace_counter_kernel is None:
        raise RuntimeError("OPTIMUS_LAZY_DELTA_BACKEND=triton-counter requires triton")
    if not z.is_cuda:
        raise RuntimeError("triton_subspace_expand_counter requires CUDA tensors")
    if z.ndim != 2:
        raise ValueError(f"expected z [rows, rank], got {tuple(z.shape)}")
    rows, rank = int(z.shape[0]), int(z.shape[1])
    output_dim = int(output_dim)
    if output_dim < 0:
        raise ValueError("output_dim must be nonnegative")
    mapping = row_candidate_indices.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if int(mapping.numel()) != rows:
        raise ValueError(f"row_candidate_indices length {int(mapping.numel())} does not match rows {rows}")
    seeds = direction_seeds.to(device=z.device, dtype=torch.int64, non_blocking=True).contiguous()
    sign_values = signs.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if seeds.ndim != 1 or sign_values.ndim != 1 or int(seeds.numel()) != int(sign_values.numel()):
        raise ValueError("direction_seeds and signs must be one-dimensional tensors with matching length")
    z_contig = z.contiguous()
    out = torch.empty((rows, output_dim), device=z.device, dtype=z.dtype)
    grid = (triton.cdiv(rows, int(block_n)), triton.cdiv(output_dim, int(block_m)))
    _subspace_counter_kernel[grid](
        z_contig,
        seeds,
        sign_values,
        mapping,
        out,
        rows,
        output_dim,
        rank,
        target_hash=int(target_hash) & 0xFFFFFFFF,
        beta=float(beta),
        output_offset=int(output_offset),
        BLOCK_N=int(block_n),
        BLOCK_M=int(block_m),
        num_warps=4,
    )
    return out
