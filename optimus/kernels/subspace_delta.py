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
        rows,
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
        rows,
        output_dim: tl.constexpr,
        rank: tl.constexpr,
        target_hash: tl.constexpr,
        beta: tl.constexpr,
        field_output_offset: tl.constexpr,
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
            normal = _normal_from_counter(seed[:, None], target_hash, field_output_offset + offs_m[None, :], r)
            acc += z[:, None] * normal * sign[:, None] * beta

        tl.store(out_ptr + offs_n[:, None] * output_dim + offs_m[None, :], acc, mask=row_mask[:, None] & col_mask[None, :])

    @triton.jit
    def _subspace_counter_add_kernel(
        z_ptr,
        seed_ptr,
        sign_ptr,
        row_candidate_ptr,
        out_ptr,
        rows,
        output_stride,
        output_dim: tl.constexpr,
        rank: tl.constexpr,
        target_hash: tl.constexpr,
        beta: tl.constexpr,
        field_output_offset: tl.constexpr,
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
            normal = _normal_from_counter(seed[:, None], target_hash, field_output_offset + offs_m[None, :], r)
            acc += z[:, None] * normal * sign[:, None] * beta

        out_offsets = offs_n[:, None] * output_stride + output_offset + offs_m[None, :]
        base = tl.load(out_ptr + out_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, base + acc, mask=row_mask[:, None] & col_mask[None, :])

    @triton.jit
    def _subspace_counter_qv_add_kernel(
        z_ptr,
        seed_ptr,
        sign_ptr,
        row_candidate_ptr,
        out_ptr,
        rows,
        output_stride,
        q_dim: tl.constexpr,
        kv_dim: tl.constexpr,
        rank: tl.constexpr,
        q_target_hash: tl.constexpr,
        v_target_hash: tl.constexpr,
        q_beta: tl.constexpr,
        v_beta: tl.constexpr,
        q_field_output_offset: tl.constexpr,
        v_field_output_offset: tl.constexpr,
        q_blocks: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_part_m = tl.program_id(1)
        is_q = pid_part_m < q_blocks
        local_pid_m = tl.where(is_q, pid_part_m, pid_part_m - q_blocks)
        width = tl.where(is_q, q_dim, kv_dim)
        output_offset = tl.where(is_q, 0, q_dim + kv_dim)
        field_output_offset = tl.where(is_q, q_field_output_offset, v_field_output_offset)
        target_hash = tl.where(is_q, q_target_hash, v_target_hash)
        beta = tl.where(is_q, q_beta, v_beta)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = local_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_n < rows
        col_mask = offs_m < width
        candidate = tl.load(row_candidate_ptr + offs_n, mask=row_mask, other=0).to(tl.int64)
        seed = tl.load(seed_ptr + candidate, mask=row_mask, other=0).to(tl.uint32)
        sign = tl.load(sign_ptr + candidate, mask=row_mask, other=1).to(tl.float32)

        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        for r in range(rank):
            z = tl.load(z_ptr + offs_n * rank + r, mask=row_mask, other=0.0).to(tl.float32)
            normal = _normal_from_counter(seed[:, None], target_hash, field_output_offset + offs_m[None, :], r)
            acc += z[:, None] * normal * sign[:, None] * beta

        out_offsets = offs_n[:, None] * output_stride + output_offset + offs_m[None, :]
        base = tl.load(out_ptr + out_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, base + acc, mask=row_mask[:, None] & col_mask[None, :])

    @triton.jit
    def _subspace_counter_qv_add_from_x_kernel(
        x_ptr,
        basis_ptr,
        seed_ptr,
        sign_ptr,
        row_candidate_ptr,
        out_ptr,
        rows,
        output_stride,
        input_dim: tl.constexpr,
        q_dim: tl.constexpr,
        kv_dim: tl.constexpr,
        rank: tl.constexpr,
        q_target_hash: tl.constexpr,
        v_target_hash: tl.constexpr,
        q_beta: tl.constexpr,
        v_beta: tl.constexpr,
        q_field_output_offset: tl.constexpr,
        v_field_output_offset: tl.constexpr,
        q_blocks: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_n = tl.program_id(0)
        pid_part_m = tl.program_id(1)
        is_q = pid_part_m < q_blocks
        local_pid_m = tl.where(is_q, pid_part_m, pid_part_m - q_blocks)
        width = tl.where(is_q, q_dim, kv_dim)
        output_offset = tl.where(is_q, 0, q_dim + kv_dim)
        field_output_offset = tl.where(is_q, q_field_output_offset, v_field_output_offset)
        target_hash = tl.where(is_q, q_target_hash, v_target_hash)
        beta = tl.where(is_q, q_beta, v_beta)

        offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
        offs_m = local_pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        row_mask = offs_n < rows
        col_mask = offs_m < width
        candidate = tl.load(row_candidate_ptr + offs_n, mask=row_mask, other=0).to(tl.int64)
        seed = tl.load(seed_ptr + candidate, mask=row_mask, other=0).to(tl.uint32)
        sign = tl.load(sign_ptr + candidate, mask=row_mask, other=1).to(tl.float32)

        acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)
        offs_d = tl.arange(0, BLOCK_D)
        for r in tl.range(0, rank):
            z_acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
            for d0 in tl.range(0, input_dim, BLOCK_D):
                d = d0 + offs_d
                d_mask = d < input_dim
                x_vals = tl.load(
                    x_ptr + offs_n[:, None] * input_dim + d[None, :],
                    mask=row_mask[:, None] & d_mask[None, :],
                    other=0.0,
                ).to(tl.float32)
                q_vals = tl.load(
                    basis_ptr + r * input_dim + d,
                    mask=d_mask,
                    other=0.0,
                ).to(tl.float32)
                z_acc += tl.sum(x_vals * q_vals[None, :], axis=1)
            normal = _normal_from_counter(seed[:, None], target_hash, field_output_offset + offs_m[None, :], r)
            acc += z_acc[:, None] * normal * sign[:, None] * beta

        out_offsets = offs_n[:, None] * output_stride + output_offset + offs_m[None, :]
        base = tl.load(out_ptr + out_offsets, mask=row_mask[:, None] & col_mask[None, :], other=0.0).to(tl.float32)
        tl.store(out_ptr + out_offsets, base + acc, mask=row_mask[:, None] & col_mask[None, :])
else:
    _subspace_expand_kernel = None
    _subspace_counter_kernel = None
    _subspace_counter_add_kernel = None
    _subspace_counter_qv_add_kernel = None
    _subspace_counter_qv_add_from_x_kernel = None


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
    field_output_offset: int = 0,
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
    field_output_offset = int(field_output_offset)
    output_offset = int(output_offset)
    if field_output_offset == 0 and output_offset != 0:
        field_output_offset = output_offset
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
        field_output_offset=field_output_offset,
        output_offset=output_offset,
        BLOCK_N=int(block_n),
        BLOCK_M=int(block_m),
        num_warps=4,
    )
    return out


def triton_subspace_add_counter_(
    z: torch.Tensor,
    direction_seeds: torch.Tensor,
    signs: torch.Tensor,
    row_candidate_indices: torch.Tensor,
    output: torch.Tensor,
    *,
    target_hash: int,
    beta: float,
    output_dim: int,
    field_output_offset: int = 0,
    output_offset: int = 0,
    block_n: int = 16,
    block_m: int = 32,
) -> torch.Tensor:
    """In-place ``output += beta * G_c @ z`` for ``counter_gaussian_v1`` fields.

    ``output`` is a two-dimensional row-major view of the linear output. The
    kernel may write a contiguous output slice by setting ``output_offset`` and
    ``output_dim``. ``field_output_offset`` is separate because target-split
    fused-qkv fields use local q/v output indices while writing into global qkv
    output slices.
    """

    if triton is None or _subspace_counter_add_kernel is None:
        raise RuntimeError("OPTIMUS_LAZY_DELTA_BACKEND=triton-counter-inplace requires triton")
    if not z.is_cuda or not output.is_cuda:
        raise RuntimeError("triton_subspace_add_counter_ requires CUDA tensors")
    if z.ndim != 2 or output.ndim != 2:
        raise ValueError(f"expected z [rows, rank] and output [rows, output], got {tuple(z.shape)} and {tuple(output.shape)}")
    rows, rank = int(z.shape[0]), int(z.shape[1])
    if int(output.shape[0]) != rows:
        raise ValueError(f"output row count {int(output.shape[0])} does not match z rows {rows}")
    output_dim = int(output_dim)
    field_output_offset = int(field_output_offset)
    output_offset = int(output_offset)
    if output_dim < 0:
        raise ValueError("output_dim must be nonnegative")
    if output_dim == 0 or rows == 0:
        return output
    if field_output_offset < 0:
        raise ValueError("field_output_offset must be nonnegative")
    if output_offset < 0 or output_offset + output_dim > int(output.shape[1]):
        raise ValueError(
            f"output slice [{output_offset}, {output_offset + output_dim}) is out of bounds for output width {int(output.shape[1])}"
        )
    mapping = row_candidate_indices.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if int(mapping.numel()) != rows:
        raise ValueError(f"row_candidate_indices length {int(mapping.numel())} does not match rows {rows}")
    seeds = direction_seeds.to(device=z.device, dtype=torch.int64, non_blocking=True).contiguous()
    sign_values = signs.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if seeds.ndim != 1 or sign_values.ndim != 1 or int(seeds.numel()) != int(sign_values.numel()):
        raise ValueError("direction_seeds and signs must be one-dimensional tensors with matching length")
    if not output.is_contiguous():
        raise ValueError("triton_subspace_add_counter_ requires a contiguous row-major output view")
    z_contig = z.contiguous()
    output_stride = int(output.stride(0))
    grid = (triton.cdiv(rows, int(block_n)), triton.cdiv(output_dim, int(block_m)))
    _subspace_counter_add_kernel[grid](
        z_contig,
        seeds,
        sign_values,
        mapping,
        output,
        rows,
        output_stride,
        output_dim,
        rank,
        target_hash=int(target_hash) & 0xFFFFFFFF,
        beta=float(beta),
        field_output_offset=field_output_offset,
        output_offset=output_offset,
        BLOCK_N=int(block_n),
        BLOCK_M=int(block_m),
        num_warps=4,
    )
    return output


def triton_subspace_add_counter_qv_(
    z: torch.Tensor,
    direction_seeds: torch.Tensor,
    signs: torch.Tensor,
    row_candidate_indices: torch.Tensor,
    output: torch.Tensor,
    *,
    q_target_hash: int,
    v_target_hash: int,
    q_beta: float,
    v_beta: float,
    q_dim: int,
    kv_dim: int,
    q_field_output_offset: int = 0,
    v_field_output_offset: int = 0,
    block_n: int = 16,
    block_m: int = 32,
) -> torch.Tensor:
    """In-place packed q/v ``counter_gaussian_v1`` add for fused qkv outputs.

    The output is a row-major fused qkv tensor with q at ``[0:q_dim]``, k at
    ``[q_dim:q_dim + kv_dim]``, and v at
    ``[q_dim + kv_dim:q_dim + 2 * kv_dim]``. The kernel updates q and v in one
    launch and intentionally leaves the k slice untouched.
    """

    if triton is None or _subspace_counter_qv_add_kernel is None:
        raise RuntimeError("triton_subspace_add_counter_qv_ requires triton")
    if not z.is_cuda or not output.is_cuda:
        raise RuntimeError("triton_subspace_add_counter_qv_ requires CUDA tensors")
    if z.ndim != 2 or output.ndim != 2:
        raise ValueError(f"expected z [rows, rank] and output [rows, output], got {tuple(z.shape)} and {tuple(output.shape)}")
    rows, rank = int(z.shape[0]), int(z.shape[1])
    q_dim = int(q_dim)
    kv_dim = int(kv_dim)
    if rows != int(output.shape[0]):
        raise ValueError(f"output row count {int(output.shape[0])} does not match z rows {rows}")
    if q_dim <= 0 or kv_dim <= 0:
        raise ValueError("q_dim and kv_dim must be positive")
    required_width = q_dim + 2 * kv_dim
    if int(output.shape[1]) < required_width:
        raise ValueError(f"output width {int(output.shape[1])} is smaller than fused qkv width {required_width}")
    if rows == 0:
        return output
    if not output.is_contiguous():
        raise ValueError("triton_subspace_add_counter_qv_ requires a contiguous row-major output view")
    q_field_output_offset = int(q_field_output_offset)
    v_field_output_offset = int(v_field_output_offset)
    if q_field_output_offset < 0 or v_field_output_offset < 0:
        raise ValueError("field output offsets must be nonnegative")
    mapping = row_candidate_indices.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if int(mapping.numel()) != rows:
        raise ValueError(f"row_candidate_indices length {int(mapping.numel())} does not match rows {rows}")
    seeds = direction_seeds.to(device=z.device, dtype=torch.int64, non_blocking=True).contiguous()
    sign_values = signs.to(device=z.device, dtype=torch.int32, non_blocking=True).contiguous()
    if seeds.ndim != 1 or sign_values.ndim != 1 or int(seeds.numel()) != int(sign_values.numel()):
        raise ValueError("direction_seeds and signs must be one-dimensional tensors with matching length")
    z_contig = z.contiguous()
    output_stride = int(output.stride(0))
    q_blocks = triton.cdiv(q_dim, int(block_m))
    v_blocks = triton.cdiv(kv_dim, int(block_m))
    grid = (triton.cdiv(rows, int(block_n)), q_blocks + v_blocks)
    _subspace_counter_qv_add_kernel[grid](
        z_contig,
        seeds,
        sign_values,
        mapping,
        output,
        rows,
        output_stride,
        q_dim,
        kv_dim,
        rank,
        q_target_hash=int(q_target_hash) & 0xFFFFFFFF,
        v_target_hash=int(v_target_hash) & 0xFFFFFFFF,
        q_beta=float(q_beta),
        v_beta=float(v_beta),
        q_field_output_offset=q_field_output_offset,
        v_field_output_offset=v_field_output_offset,
        q_blocks=q_blocks,
        BLOCK_N=int(block_n),
        BLOCK_M=int(block_m),
        num_warps=4,
    )
    return output


def triton_subspace_add_counter_qv_from_x_(
    x: torch.Tensor,
    basis: torch.Tensor,
    direction_seeds: torch.Tensor,
    signs: torch.Tensor,
    row_candidate_indices: torch.Tensor,
    output: torch.Tensor,
    *,
    q_target_hash: int,
    v_target_hash: int,
    q_beta: float,
    v_beta: float,
    q_dim: int,
    kv_dim: int,
    q_field_output_offset: int = 0,
    v_field_output_offset: int = 0,
    block_n: int = 8,
    block_m: int = 32,
    block_d: int = 32,
) -> torch.Tensor:
    """Prototype fused ``Qx + packed q/v counter add`` for fused qkv outputs.

    This is intentionally guarded as a prototype path. It avoids writing the
    intermediate ``z = Qx`` tensor, but the current implementation recomputes
    the local ``Qx`` values per output tile. Benchmarks decide whether this
    shape is useful before it becomes a runtime default.
    """

    if triton is None or _subspace_counter_qv_add_from_x_kernel is None:
        raise RuntimeError("triton_subspace_add_counter_qv_from_x_ requires triton")
    if not x.is_cuda or not basis.is_cuda or not output.is_cuda:
        raise RuntimeError("triton_subspace_add_counter_qv_from_x_ requires CUDA tensors")
    if x.ndim != 2 or basis.ndim != 2 or output.ndim != 2:
        raise ValueError(
            f"expected x [rows, input], basis [rank, input], and output [rows, output], "
            f"got {tuple(x.shape)}, {tuple(basis.shape)}, and {tuple(output.shape)}"
        )
    rows, input_dim = int(x.shape[0]), int(x.shape[1])
    rank, basis_input_dim = int(basis.shape[0]), int(basis.shape[1])
    q_dim = int(q_dim)
    kv_dim = int(kv_dim)
    if basis_input_dim != input_dim:
        raise ValueError(f"basis input dim {basis_input_dim} does not match x input dim {input_dim}")
    if rows != int(output.shape[0]):
        raise ValueError(f"output row count {int(output.shape[0])} does not match x rows {rows}")
    if q_dim <= 0 or kv_dim <= 0:
        raise ValueError("q_dim and kv_dim must be positive")
    required_width = q_dim + 2 * kv_dim
    if int(output.shape[1]) < required_width:
        raise ValueError(f"output width {int(output.shape[1])} is smaller than fused qkv width {required_width}")
    if rows == 0:
        return output
    if not output.is_contiguous():
        raise ValueError("triton_subspace_add_counter_qv_from_x_ requires a contiguous row-major output view")
    q_field_output_offset = int(q_field_output_offset)
    v_field_output_offset = int(v_field_output_offset)
    if q_field_output_offset < 0 or v_field_output_offset < 0:
        raise ValueError("field output offsets must be nonnegative")
    mapping = row_candidate_indices.to(device=x.device, dtype=torch.int32, non_blocking=True).contiguous()
    if int(mapping.numel()) != rows:
        raise ValueError(f"row_candidate_indices length {int(mapping.numel())} does not match rows {rows}")
    seeds = direction_seeds.to(device=x.device, dtype=torch.int64, non_blocking=True).contiguous()
    sign_values = signs.to(device=x.device, dtype=torch.int32, non_blocking=True).contiguous()
    if seeds.ndim != 1 or sign_values.ndim != 1 or int(seeds.numel()) != int(sign_values.numel()):
        raise ValueError("direction_seeds and signs must be one-dimensional tensors with matching length")
    x_contig = x.contiguous()
    basis_contig = basis.contiguous()
    output_stride = int(output.stride(0))
    q_blocks = triton.cdiv(q_dim, int(block_m))
    v_blocks = triton.cdiv(kv_dim, int(block_m))
    grid = (triton.cdiv(rows, int(block_n)), q_blocks + v_blocks)
    _subspace_counter_qv_add_from_x_kernel[grid](
        x_contig,
        basis_contig,
        seeds,
        sign_values,
        mapping,
        output,
        rows,
        output_stride,
        input_dim,
        q_dim,
        kv_dim,
        rank,
        q_target_hash=int(q_target_hash) & 0xFFFFFFFF,
        v_target_hash=int(v_target_hash) & 0xFFFFFFFF,
        q_beta=float(q_beta),
        v_beta=float(v_beta),
        q_field_output_offset=q_field_output_offset,
        v_field_output_offset=v_field_output_offset,
        q_blocks=q_blocks,
        BLOCK_N=int(block_n),
        BLOCK_M=int(block_m),
        BLOCK_D=int(block_d),
        num_warps=4,
    )
    return output
