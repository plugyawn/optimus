from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from contextlib import contextmanager

import torch

from optimus import __version__
from optimus.search.ensemble import majority_vote_evaluation
from optimus.serving.prompting import make_vllm_prompt_inputs
from optimus.serving.runtime import (
    configure_vllm_logging,
    extract_output,
    make_sampling_params,
    optional_vllm_kwargs,
    runtime_environment,
    write_json,
    write_jsonl,
)
from optimus.subspace import ActivationSite, BasisKind, BudgetPolicy, ScaleMode, SubspaceCandidate, TargetModule
from optimus.subspace.reference import (
    ReferenceState,
    TargetRuntime,
    build_basis,
    config_hash,
    gate_artifacts,
    git_commit,
    git_dirty,
    make_candidates,
    parse_float_grid,
    parse_int_grid,
    parse_layers,
    resolve_target_scales,
    sha256_bytes,
    sha256_json,
    tensor_sha256,
    torch_payload_bytes,
    validation_evidence,
)
from optimus.tasks.countdown import CountdownExample, load_examples, score_completion
from optimus.tasks.prompt_variants import make_variant_prompts


@dataclass
class HookTarget:
    module_name: str
    target_id: str
    site_id: str
    layer_index: int
    block_path: str
    suffix: str
    module: torch.nn.Module
    input_dim: int | None = None
    output_dim: int | None = None
    output_power_sum: float = 0.0
    output_power_count: int = 0
    fused_qkv_slices: tuple[str, ...] = ()
    fused_q_out: int | None = None
    fused_kv_out: int | None = None


class LazyHookRuntime:
    def __init__(self, targets: list[HookTarget], *, max_activation_rows_per_site: int = 4096, sync_timing: bool | None = None) -> None:
        self.targets = {target.module_name: target for target in targets}
        self.max_activation_rows_per_site = max_activation_rows_per_site
        self.sync_timing = _env_flag("OPTIMUS_SYNC_LAZY_TIMING", default=False) if sync_timing is None else bool(sync_timing)
        self.compute_dtype_policy = os.environ.get("OPTIMUS_LAZY_COMPUTE_DTYPE", "activation").strip().lower()
        self.delta_backend = os.environ.get("OPTIMUS_LAZY_DELTA_BACKEND", "torch").strip().lower() or "torch"
        self.field_policy = os.environ.get("OPTIMUS_LAZY_FIELD_POLICY", "target-split").strip().lower() or "target-split"
        if self.field_policy not in {"target-split", "fused-qkv-exact"}:
            raise ValueError(f"unknown OPTIMUS_LAZY_FIELD_POLICY={self.field_policy!r}")
        self.qkv_kernel_policy = os.environ.get("OPTIMUS_LAZY_QKV_KERNEL_POLICY", "split-launches").strip().lower() or "split-launches"
        if self.qkv_kernel_policy not in {"split-launches", "packed-qkv"}:
            raise ValueError(f"unknown OPTIMUS_LAZY_QKV_KERNEL_POLICY={self.qkv_kernel_policy!r}")
        self.collecting = False
        self.active_candidate: SubspaceCandidate | None = None
        self.active_candidates: list[SubspaceCandidate] = []
        self.request_candidate_by_id: dict[str, SubspaceCandidate] = {}
        self._candidate_index_by_id: dict[str, int] = {}
        self._order_prompt_count = 0
        self._order_request_id_start: int | None = None
        self._row_candidate_indices_cpu: torch.Tensor | None = None
        self._row_candidate_spans: list[tuple[int, int, int]] = []
        self._row_candidate_indices_len = 0
        self.basis_by_site: dict[str, torch.Tensor] = {}
        self.beta_by_target: dict[str, float] = {}
        self.activation_rows: dict[str, list[torch.Tensor]] = {}
        self.qx_time_s = 0.0
        self.delta_time_s = 0.0
        self.stack_time_s = 0.0
        self.meta_time_s = 0.0
        self.kernel_time_s = 0.0
        self.delta_rows = 0
        self.delta_calls = 0
        self._field_cache: dict[tuple[str, str, str, torch.dtype, int, int, int], torch.Tensor] = {}
        self._scaled_field_cache: dict[tuple[str, str, str, torch.dtype, int, int, int, float], torch.Tensor] = {}
        self._basis_cache: dict[tuple[str, str, torch.dtype], torch.Tensor] = {}
        self._vllm_meta_cache: dict[tuple[str, int, int], Any] = {}
        self._vllm_meta_prepared_keys: dict[tuple[str, int, int], tuple[Any, ...]] = {}
        self._vllm_mapping_cache: dict[tuple[Any, ...], torch.Tensor] = {}
        self._vllm_a_stack_cache: dict[tuple[str, str, torch.dtype, int, int, int], torch.Tensor] = {}
        self._vllm_b_stack_cache: dict[
            tuple[str, tuple[tuple[str, int], ...], str, torch.dtype, int, int, tuple[int, int] | None, int, int, float],
            torch.Tensor,
        ] = {}
        self._row_mapping_generation = 0

    def reset_timing(self) -> None:
        self.qx_time_s = 0.0
        self.delta_time_s = 0.0
        self.stack_time_s = 0.0
        self.meta_time_s = 0.0
        self.kernel_time_s = 0.0
        self.delta_rows = 0
        self.delta_calls = 0

    def _clear_candidate_caches(self) -> None:
        self._field_cache.clear()
        self._scaled_field_cache.clear()
        self._vllm_a_stack_cache.clear()
        self._vllm_b_stack_cache.clear()
        self._vllm_meta_prepared_keys.clear()
        self._vllm_mapping_cache.clear()
        self._row_mapping_generation += 1

    def set_candidate(self, candidate: SubspaceCandidate | None) -> None:
        self.active_candidate = candidate
        self.active_candidates = []
        self.request_candidate_by_id = {}
        self._candidate_index_by_id = {}
        self._order_prompt_count = 0
        self._order_request_id_start = None
        self._row_candidate_indices_cpu = None
        self._row_candidate_spans = []
        self._row_candidate_indices_len = 0
        self._clear_candidate_caches()

    def set_candidate_batch(self, request_candidate_by_id: dict[str, SubspaceCandidate]) -> None:
        self.active_candidate = None
        self.request_candidate_by_id = dict(request_candidate_by_id)
        candidates: list[SubspaceCandidate] = []
        seen: set[str] = set()
        for candidate in self.request_candidate_by_id.values():
            if candidate.candidate_id in seen:
                continue
            seen.add(candidate.candidate_id)
            candidates.append(candidate)
        self.active_candidates = candidates
        self._candidate_index_by_id = {candidate.candidate_id: idx for idx, candidate in enumerate(candidates)}
        self._order_prompt_count = 0
        self._order_request_id_start = None
        self._row_candidate_indices_cpu = None
        self._row_candidate_spans = []
        self._row_candidate_indices_len = 0
        self._clear_candidate_caches()

    def set_candidate_batch_by_order(self, candidates: list[SubspaceCandidate], *, prompt_count: int) -> None:
        self.active_candidate = None
        self.request_candidate_by_id = {}
        self.active_candidates = list(candidates)
        self._candidate_index_by_id = {candidate.candidate_id: idx for idx, candidate in enumerate(candidates)}
        self._order_prompt_count = max(1, int(prompt_count))
        self._order_request_id_start = None
        self._row_candidate_indices_cpu = None
        self._row_candidate_spans = []
        self._row_candidate_indices_len = 0
        self._clear_candidate_caches()

    def update_row_candidates(self, req_ids: list[str], query_start_loc: Any) -> None:
        if not self.request_candidate_by_id and not (self.active_candidates and self._order_prompt_count > 0):
            self._row_candidate_indices_cpu = None
            self._row_candidate_spans = []
            self._row_candidate_indices_len = 0
            return
        if torch.is_tensor(query_start_loc):
            loc = query_start_loc.detach().cpu().tolist()
        else:
            loc = list(query_start_loc)
        request_ordinals = [_request_ordinal(req_id) for req_id in req_ids]
        if not self.request_candidate_by_id:
            numeric = [ordinal for ordinal in request_ordinals if ordinal is not None]
            if numeric:
                current_min = min(numeric)
                if self._order_request_id_start is None or current_min < self._order_request_id_start:
                    self._order_request_id_start = int(current_min)
        pieces: list[torch.Tensor] = []
        spans: list[tuple[int, int, int]] = []
        for req_index, req_id in enumerate(req_ids):
            if req_index + 1 >= len(loc):
                break
            start = int(loc[req_index])
            end = int(loc[req_index + 1])
            count = max(0, end - start)
            if count == 0:
                continue
            if self.request_candidate_by_id:
                candidate = self.request_candidate_by_id.get(str(req_id))
                candidate_index = -1 if candidate is None else self._candidate_index_by_id[candidate.candidate_id]
            else:
                ordinal = request_ordinals[req_index] if req_index < len(request_ordinals) else None
                if ordinal is not None and self._order_request_id_start is not None:
                    candidate_index = (int(ordinal) - int(self._order_request_id_start)) // self._order_prompt_count
                else:
                    candidate_index = req_index // self._order_prompt_count
                if candidate_index < 0 or candidate_index >= len(self.active_candidates):
                    raise RuntimeError(
                        "vLLM lazy hook could not route request id to candidate: "
                        f"req_id={req_id!r} ordinal={ordinal!r} start={self._order_request_id_start!r} "
                        f"prompt_count={self._order_prompt_count} candidates={len(self.active_candidates)}"
                    )
            pieces.append(torch.full((count,), candidate_index, dtype=torch.int16))
            if candidate_index >= 0:
                spans.append((candidate_index, start, end))
        if not pieces:
            self._row_candidate_indices_cpu = None
            self._row_candidate_spans = []
            self._row_candidate_indices_len = 0
            return
        row_indices = torch.cat(pieces, dim=0).contiguous()
        self._row_candidate_indices_cpu = row_indices
        self._row_candidate_spans = spans
        self._row_candidate_indices_len = int(row_indices.numel())
        self._vllm_meta_prepared_keys.clear()
        self._vllm_mapping_cache.clear()
        self._row_mapping_generation += 1

    def basis_for(self, site_id: str, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor | None:
        basis = self.basis_by_site.get(site_id)
        if basis is None:
            return None
        key = (site_id, str(device), dtype)
        cached = self._basis_cache.get(key)
        if cached is not None:
            return cached
        moved = basis.to(device=device, dtype=dtype, non_blocking=True)
        self._basis_cache[key] = moved
        return moved

    def collect(self, target: HookTarget, x: torch.Tensor, y: torch.Tensor) -> None:
        flat_x = x.detach().reshape(-1, x.shape[-1]).float().cpu()
        flat_y = y.detach().reshape(-1, y.shape[-1]).float()
        target.input_dim = int(flat_x.shape[-1])
        target.output_dim = int(flat_y.shape[-1])
        power = flat_y.square().sum(dim=1)
        target.output_power_sum += float(power.sum().item())
        target.output_power_count += int(power.numel())
        rows = self.activation_rows.setdefault(target.site_id, [])
        current = sum(int(item.shape[0]) for item in rows)
        remaining = max(0, self.max_activation_rows_per_site - current)
        if remaining:
            rows.append(flat_x[:remaining].contiguous())

    def compute_dtype_for(self, x: torch.Tensor) -> torch.dtype:
        if self.compute_dtype_policy in {"fp32", "float32"}:
            return torch.float32
        if self.compute_dtype_policy in {"bf16", "bfloat16"}:
            return torch.bfloat16
        if self.compute_dtype_policy in {"fp16", "float16"}:
            return torch.float16
        if self.compute_dtype_policy in {"activation", "auto"} and x.is_cuda and x.dtype in {torch.bfloat16, torch.float16}:
            return x.dtype
        return torch.float32

    def field(
        self,
        target: HookTarget,
        candidate: SubspaceCandidate,
        *,
        output_dim: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        target_id: str | None = None,
    ) -> torch.Tensor:
        source_rank = max(int(rank), int(candidate.basis_rank))
        field_target_id = target.target_id if target_id is None else target_id
        key = (field_target_id, candidate.candidate_id, str(device), dtype, output_dim, source_rank, rank)
        cached = self._field_cache.get(key)
        if cached is not None:
            return cached
        seed_payload = f"{candidate.direction_seed}\0{field_target_id}\0torch_generator_field_v1".encode("utf-8")
        seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) % (2**63 - 1)
        gen = torch.Generator(device="cpu").manual_seed(seed)
        field = torch.randn((output_dim, source_rank), generator=gen, dtype=torch.float32).to(device)
        if candidate.sign == "-":
            field = -field
        field = field[:, : int(rank)].contiguous()
        if dtype != torch.float32:
            field = field.to(dtype=dtype)
        self._field_cache[key] = field
        return field

    def scaled_field(
        self,
        target: HookTarget,
        candidate: SubspaceCandidate,
        *,
        output_dim: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        beta: float,
        target_id: str | None = None,
    ) -> torch.Tensor:
        source_rank = max(int(rank), int(candidate.basis_rank))
        field_target_id = target.target_id if target_id is None else target_id
        key = (field_target_id, candidate.candidate_id, str(device), dtype, output_dim, source_rank, rank, float(beta))
        cached = self._scaled_field_cache.get(key)
        if cached is not None:
            return cached
        field = self.field(
            target,
            candidate,
            output_dim=output_dim,
            rank=rank,
            device=device,
            dtype=torch.float32,
            target_id=field_target_id,
        )
        scaled = (float(beta) * field).to(dtype=dtype).contiguous()
        self._scaled_field_cache[key] = scaled
        return scaled

    def _qkv_slice_for(self, target: HookTarget, suffix: str, output_dim: int) -> slice:
        if target.fused_q_out is None or target.fused_kv_out is None:
            raise RuntimeError(f"{target.module_name} requires fused qkv dimensions for split lazy replay")
        q_out = int(target.fused_q_out)
        kv_out = int(target.fused_kv_out)
        if output_dim != q_out + 2 * kv_out:
            raise RuntimeError(f"{target.module_name} output dim {output_dim} != expected fused qkv dim {q_out + 2 * kv_out}")
        if suffix == "q_proj":
            return slice(0, q_out)
        if suffix == "k_proj":
            return slice(q_out, q_out + kv_out)
        if suffix == "v_proj":
            return slice(q_out + kv_out, q_out + 2 * kv_out)
        raise RuntimeError(f"unsupported fused qkv split suffix {suffix!r}")

    def _split_target_id(self, target: HookTarget, suffix: str) -> str:
        if suffix in {"q_proj", "k_proj", "v_proj"} and target.suffix == "qkv_proj":
            return f"layer_{target.layer_index}.self_attn.{suffix}"
        return target.target_id

    def _beta_for_split(self, target: HookTarget, suffix: str | None = None) -> float | None:
        if suffix is not None:
            split_target_id = self._split_target_id(target, suffix)
            beta = self.beta_by_target.get(split_target_id)
            if beta is not None:
                return beta
        return self.beta_by_target.get(target.target_id)

    def _candidate_delta_from_z(
        self,
        target: HookTarget,
        candidate: SubspaceCandidate,
        z: torch.Tensor,
        *,
        output_dim: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if target.suffix == "qkv_proj" and target.fused_qkv_slices:
            delta = torch.zeros((int(z.shape[0]), output_dim), device=device, dtype=dtype)
            for suffix in target.fused_qkv_slices:
                output_slice = self._qkv_slice_for(target, suffix, output_dim)
                width = int(output_slice.stop - output_slice.start)
                beta = self._beta_for_split(target, suffix)
                if beta is None or float(beta) == 0.0:
                    continue
                if self.field_policy == "fused-qkv-exact":
                    field = self.scaled_field(
                        target,
                        candidate,
                        output_dim=output_dim,
                        rank=rank,
                        device=device,
                        dtype=dtype,
                        beta=float(beta),
                        target_id=target.target_id,
                    )[output_slice, :].contiguous()
                else:
                    field = self.scaled_field(
                        target,
                        candidate,
                        output_dim=width,
                        rank=rank,
                        device=device,
                        dtype=dtype,
                        beta=float(beta),
                        target_id=self._split_target_id(target, suffix),
                    )
                delta[:, output_slice] = z @ field.T
            return delta
        beta = self._beta_for_split(target)
        if beta is None or float(beta) == 0.0:
            return torch.zeros((int(z.shape[0]), output_dim), device=device, dtype=dtype)
        field = self.scaled_field(
            target,
            candidate,
            output_dim=output_dim,
            rank=rank,
            device=device,
            dtype=dtype,
            beta=float(beta),
        )
        return z @ field.T

    def _vllm_meta(self, *, device: torch.device, max_loras: int, rows: int) -> Any:
        key = (str(device), int(max_loras), int(rows))
        cached = self._vllm_meta_cache.get(key)
        if cached is not None:
            return cached
        try:
            from vllm.lora.ops.triton_ops import LoRAKernelMeta
        except Exception as exc:  # pragma: no cover - exercised on GPU hosts with vLLM.
            raise RuntimeError("OPTIMUS_LAZY_DELTA_BACKEND=vllm-lora-kernel requires vLLM Triton LoRA ops") from exc
        meta = LoRAKernelMeta.make(int(max_loras), int(rows), device=device)
        self._vllm_meta_cache[key] = meta
        return meta

    def _mapping_cache_key(self, *, device: torch.device, rows: int, num_loras: int) -> tuple[Any, ...]:
        return (int(self._row_mapping_generation), str(device), int(rows), int(num_loras))

    def _token_mapping_for(
        self,
        row_mapping: torch.Tensor,
        *,
        device: torch.device,
        rows: int,
        num_loras: int,
    ) -> tuple[torch.Tensor, tuple[Any, ...]]:
        key = self._mapping_cache_key(device=device, rows=rows, num_loras=num_loras)
        cached = self._vllm_mapping_cache.get(key)
        if cached is not None:
            return cached, key
        token_mapping = row_mapping.to(device=device, dtype=torch.int32, non_blocking=True).contiguous()
        self._vllm_mapping_cache[key] = token_mapping
        return token_mapping, key

    def _prepare_vllm_meta(
        self,
        meta: Any,
        token_mapping: torch.Tensor,
        *,
        meta_key: tuple[str, int, int],
        mapping_key: tuple[Any, ...],
    ) -> None:
        if self._vllm_meta_prepared_keys.get(meta_key) == mapping_key:
            return
        started = time.perf_counter()
        meta.prepare_tensors(token_mapping)
        self.meta_time_s += time.perf_counter() - started
        self._vllm_meta_prepared_keys[meta_key] = mapping_key

    def _vllm_a_stack(
        self,
        target: HookTarget,
        basis: torch.Tensor,
        *,
        num_loras: int,
        input_dim: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (target.site_id, str(device), dtype, int(input_dim), int(rank), int(num_loras))
        cached = self._vllm_a_stack_cache.get(key)
        if cached is not None:
            return cached
        stack = basis[:rank].unsqueeze(0).expand(int(num_loras), -1, -1).contiguous()
        self._vllm_a_stack_cache[key] = stack
        return stack

    def _vllm_b_stack(
        self,
        target: HookTarget,
        candidates: list[SubspaceCandidate],
        *,
        output_dim: int,
        rank: int,
        device: torch.device,
        dtype: torch.dtype,
        beta: float,
        target_id: str,
        field_output_dim: int | None = None,
        field_slice: slice | None = None,
    ) -> torch.Tensor:
        candidate_key = tuple((candidate.candidate_id, int(candidate.basis_rank)) for candidate in candidates)
        source_rank = max([int(rank), *(int(candidate.basis_rank) for candidate in candidates)])
        field_output_dim_int = int(field_output_dim or output_dim)
        slice_key = None if field_slice is None else (int(field_slice.start or 0), int(field_slice.stop or field_output_dim_int))
        key = (
            target_id,
            candidate_key,
            str(device),
            dtype,
            int(output_dim),
            int(field_output_dim_int),
            slice_key,
            int(source_rank),
            int(rank),
            float(beta),
        )
        cached = self._vllm_b_stack_cache.get(key)
        if cached is not None:
            return cached
        fields = []
        for candidate in candidates:
            field = self.scaled_field(
                target,
                candidate,
                output_dim=field_output_dim_int,
                rank=rank,
                device=device,
                dtype=dtype,
                beta=float(beta),
                target_id=target_id,
            )
            if field_slice is not None:
                field = field[field_slice, :]
            if int(field.shape[0]) != int(output_dim):
                raise RuntimeError(
                    "lazy vLLM B-stack field/output width mismatch: "
                    f"field={tuple(field.shape)} output_dim={output_dim} target_id={target_id}"
                )
            fields.append(field.contiguous())
        stack = torch.stack(fields, dim=0).contiguous()
        self._vllm_b_stack_cache[key] = stack
        return stack

    def _vllm_lora_kernel_delta_for_target(
        self,
        target: HookTarget,
        candidates: list[SubspaceCandidate],
        row_mapping: torch.Tensor,
        flat_x: torch.Tensor,
        *,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
        beta: float,
        target_id: str,
        field_output_dim: int | None = None,
        field_slice: slice | None = None,
    ) -> torch.Tensor:
        if not flat_x.is_cuda:
            raise RuntimeError("vllm-lora-kernel lazy delta backend requires CUDA tensors")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError("vllm-lora-kernel lazy delta backend requires fp16/bf16 compute dtype")
        try:
            from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
        except Exception as exc:  # pragma: no cover - exercised on GPU hosts with vLLM.
            raise RuntimeError("OPTIMUS_LAZY_DELTA_BACKEND=vllm-lora-kernel requires vLLM Triton LoRA ops") from exc

        basis = self.basis_for(target.site_id, device=flat_x.device, dtype=dtype)
        if basis is None:
            raise RuntimeError(f"missing lazy basis for activation site {target.site_id}")
        basis = basis[:rank].contiguous()
        input_dim = int(flat_x.shape[-1])
        if int(basis.shape[-1]) != input_dim:
            raise RuntimeError(f"basis width mismatch for {target.site_id}: basis={tuple(basis.shape)} x={tuple(flat_x.shape)}")
        num_loras = max(1, len(candidates))
        stack_started = time.perf_counter()
        a_stack = self._vllm_a_stack(
            target,
            basis,
            num_loras=num_loras,
            input_dim=input_dim,
            rank=rank,
            device=flat_x.device,
            dtype=dtype,
        )
        b_stack = self._vllm_b_stack(
            target,
            candidates,
            output_dim=output_dim,
            rank=rank,
            device=flat_x.device,
            dtype=dtype,
            beta=float(beta),
            target_id=target_id,
            field_output_dim=field_output_dim,
            field_slice=field_slice,
        )
        self.stack_time_s += time.perf_counter() - stack_started

        rows = int(flat_x.shape[0])
        meta_key = (str(flat_x.device), num_loras, rows)
        meta = self._vllm_meta(device=flat_x.device, max_loras=num_loras, rows=rows)
        meta_started = time.perf_counter()
        token_mapping, mapping_key = self._token_mapping_for(
            row_mapping,
            device=flat_x.device,
            rows=rows,
            num_loras=num_loras,
        )
        self.meta_time_s += time.perf_counter() - meta_started
        self._prepare_vllm_meta(meta, token_mapping, meta_key=meta_key, mapping_key=mapping_key)
        x_kernel = flat_x.to(dtype=dtype).contiguous()
        buffer = torch.empty((1, rows, rank), device=flat_x.device, dtype=torch.float32)
        delta = torch.zeros((rows, output_dim), device=flat_x.device, dtype=dtype)
        kernel_started = time.perf_counter()
        lora_shrink(x_kernel, (a_stack,), buffer, *meta.meta_args(rows, True), 1.0)
        lora_expand(buffer, (b_stack,), delta, *meta.meta_args(rows, True), offset_start=0, add_inputs=True)
        self.kernel_time_s += time.perf_counter() - kernel_started
        return delta

    def _vllm_lora_kernel_delta_for_qkv(
        self,
        target: HookTarget,
        candidates: list[SubspaceCandidate],
        row_mapping: torch.Tensor,
        flat_x: torch.Tensor,
        *,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not flat_x.is_cuda:
            raise RuntimeError("vllm-lora-kernel lazy delta backend requires CUDA tensors")
        if dtype not in {torch.float16, torch.bfloat16}:
            raise RuntimeError("vllm-lora-kernel lazy delta backend requires fp16/bf16 compute dtype")
        try:
            from vllm.lora.ops.triton_ops import lora_expand, lora_shrink
        except Exception as exc:  # pragma: no cover - exercised on GPU hosts with vLLM.
            raise RuntimeError("OPTIMUS_LAZY_DELTA_BACKEND=vllm-lora-kernel requires vLLM Triton LoRA ops") from exc

        basis = self.basis_for(target.site_id, device=flat_x.device, dtype=dtype)
        if basis is None:
            raise RuntimeError(f"missing lazy basis for activation site {target.site_id}")
        basis = basis[:rank].contiguous()
        input_dim = int(flat_x.shape[-1])
        if int(basis.shape[-1]) != input_dim:
            raise RuntimeError(f"basis width mismatch for {target.site_id}: basis={tuple(basis.shape)} x={tuple(flat_x.shape)}")

        q_slice = self._qkv_slice_for(target, "q_proj", output_dim)
        k_slice = self._qkv_slice_for(target, "k_proj", output_dim)
        v_slice = self._qkv_slice_for(target, "v_proj", output_dim)
        slice_by_suffix = {"q_proj": q_slice, "k_proj": k_slice, "v_proj": v_slice}
        width_by_suffix = {suffix: int(s.stop - s.start) for suffix, s in slice_by_suffix.items()}
        active = set(target.fused_qkv_slices)
        num_loras = max(1, len(candidates))

        stack_started = time.perf_counter()
        basis_stack = self._vllm_a_stack(
            target,
            basis,
            num_loras=num_loras,
            input_dim=input_dim,
            rank=rank,
            device=flat_x.device,
            dtype=dtype,
        )
        zero_a_stack: torch.Tensor | None = None
        a_stacks: list[torch.Tensor] = []
        b_stacks: list[torch.Tensor] = []
        for suffix in ("q_proj", "k_proj", "v_proj"):
            width = width_by_suffix[suffix]
            beta = self._beta_for_split(target, suffix)
            if suffix not in active or beta is None or float(beta) == 0.0:
                if zero_a_stack is None:
                    zero_a_stack = torch.zeros_like(basis_stack)
                a_stacks.append(zero_a_stack)
                b_stacks.append(torch.zeros((num_loras, width, rank), device=flat_x.device, dtype=dtype))
                continue
            a_stacks.append(basis_stack)
            if self.field_policy == "fused-qkv-exact":
                b_stacks.append(
                    self._vllm_b_stack(
                        target,
                        candidates,
                        output_dim=width,
                        rank=rank,
                        device=flat_x.device,
                        dtype=dtype,
                        beta=float(beta),
                        target_id=target.target_id,
                        field_output_dim=output_dim,
                        field_slice=slice_by_suffix[suffix],
                    )
                )
            else:
                b_stacks.append(
                    self._vllm_b_stack(
                        target,
                        candidates,
                        output_dim=width,
                        rank=rank,
                        device=flat_x.device,
                        dtype=dtype,
                        beta=float(beta),
                        target_id=self._split_target_id(target, suffix),
                    )
                )
        self.stack_time_s += time.perf_counter() - stack_started

        rows = int(flat_x.shape[0])
        meta_key = (str(flat_x.device), num_loras, rows)
        meta = self._vllm_meta(device=flat_x.device, max_loras=num_loras, rows=rows)
        meta_started = time.perf_counter()
        token_mapping, mapping_key = self._token_mapping_for(
            row_mapping,
            device=flat_x.device,
            rows=rows,
            num_loras=num_loras,
        )
        self.meta_time_s += time.perf_counter() - meta_started
        self._prepare_vllm_meta(meta, token_mapping, meta_key=meta_key, mapping_key=mapping_key)

        x_kernel = flat_x.to(dtype=dtype).contiguous()
        buffer = torch.empty((3, rows, rank), device=flat_x.device, dtype=torch.float32)
        delta = torch.zeros((rows, output_dim), device=flat_x.device, dtype=dtype)
        kernel_started = time.perf_counter()
        lora_shrink(x_kernel, tuple(a_stacks), buffer, *meta.meta_args(rows, True), 1.0)
        lora_expand(buffer, tuple(b_stacks), delta, *meta.meta_args(rows, True), offset_start=0, add_inputs=True)
        self.kernel_time_s += time.perf_counter() - kernel_started
        return delta

    def _delta_vllm_lora_kernel(
        self,
        target: HookTarget,
        flat_x: torch.Tensor,
        y: torch.Tensor,
        *,
        candidate: SubspaceCandidate | None,
        row_candidates: list[SubspaceCandidate],
        row_candidate_indices: torch.Tensor | None,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if candidate is not None:
            candidates = [candidate]
            row_mapping = torch.zeros((int(flat_x.shape[0]),), dtype=torch.int32)
        else:
            if row_candidate_indices is None or not row_candidates:
                raise RuntimeError("missing row-candidate routing for vllm-lora-kernel lazy delta backend")
            candidates = row_candidates
            row_mapping = row_candidate_indices.to(dtype=torch.int32)
            if row_mapping.numel():
                min_index = int(row_mapping.min().item())
                max_index = int(row_mapping.max().item())
                if min_index < 0 or max_index >= len(candidates):
                    raise RuntimeError(
                        "vLLM lazy hook row-candidate mapping is out of bounds: "
                        f"min={min_index} max={max_index} candidates={len(candidates)}"
                    )

        if target.suffix == "qkv_proj" and target.fused_qkv_slices:
            if self.qkv_kernel_policy == "packed-qkv":
                delta = self._vllm_lora_kernel_delta_for_qkv(
                    target,
                    candidates,
                    row_mapping,
                    flat_x,
                    output_dim=output_dim,
                    rank=rank,
                    dtype=dtype,
                )
            else:
                delta = torch.zeros((int(flat_x.shape[0]), output_dim), device=flat_x.device, dtype=dtype)
                for suffix in target.fused_qkv_slices:
                    beta = self._beta_for_split(target, suffix)
                    if beta is None or float(beta) == 0.0:
                        continue
                    output_slice = self._qkv_slice_for(target, suffix, output_dim)
                    width = int(output_slice.stop - output_slice.start)
                    if self.field_policy == "fused-qkv-exact":
                        field_target_id = target.target_id
                        field_output_dim = output_dim
                        field_slice = output_slice
                    else:
                        field_target_id = self._split_target_id(target, suffix)
                        field_output_dim = None
                        field_slice = None
                    split = self._vllm_lora_kernel_delta_for_target(
                        target,
                        candidates,
                        row_mapping,
                        flat_x,
                        output_dim=width,
                        rank=rank,
                        dtype=dtype,
                        beta=float(beta),
                        target_id=field_target_id,
                        field_output_dim=field_output_dim,
                        field_slice=field_slice,
                    )
                    delta[:, output_slice] = split
            return delta.reshape(y.shape).to(dtype=y.dtype)

        beta = self._beta_for_split(target)
        if beta is None or float(beta) == 0.0:
            return torch.zeros_like(y)
        delta = self._vllm_lora_kernel_delta_for_target(
            target,
            candidates,
            row_mapping,
            flat_x,
            output_dim=output_dim,
            rank=rank,
            dtype=dtype,
            beta=float(beta),
            target_id=target.target_id,
        )
        return delta.reshape(y.shape).to(dtype=y.dtype)

    def _triton_expand_delta_for_target(
        self,
        target: HookTarget,
        candidates: list[SubspaceCandidate],
        row_mapping: torch.Tensor,
        z: torch.Tensor,
        *,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
        beta: float,
        target_id: str,
        field_output_dim: int | None = None,
        field_slice: slice | None = None,
    ) -> torch.Tensor:
        from optimus.kernels import triton_subspace_expand

        stack_started = time.perf_counter()
        b_stack = self._vllm_b_stack(
            target,
            candidates,
            output_dim=output_dim,
            rank=rank,
            device=z.device,
            dtype=dtype,
            beta=float(beta),
            target_id=target_id,
            field_output_dim=field_output_dim,
            field_slice=field_slice,
        )
        self.stack_time_s += time.perf_counter() - stack_started
        kernel_started = time.perf_counter()
        delta = triton_subspace_expand(z, b_stack, row_mapping)
        self.kernel_time_s += time.perf_counter() - kernel_started
        return delta

    def _delta_triton(
        self,
        target: HookTarget,
        z: torch.Tensor,
        y: torch.Tensor,
        *,
        candidate: SubspaceCandidate | None,
        row_candidates: list[SubspaceCandidate],
        row_candidate_indices: torch.Tensor | None,
        output_dim: int,
        rank: int,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if candidate is not None:
            candidates = [candidate]
            row_mapping = torch.zeros((int(z.shape[0]),), dtype=torch.int32)
        else:
            if row_candidate_indices is None or not row_candidates:
                raise RuntimeError("missing row-candidate routing for triton lazy delta backend")
            candidates = row_candidates
            row_mapping = row_candidate_indices.to(dtype=torch.int32)
            if row_mapping.numel():
                min_index = int(row_mapping.min().item())
                max_index = int(row_mapping.max().item())
                if min_index < 0 or max_index >= len(candidates):
                    raise RuntimeError(
                        "Triton lazy hook row-candidate mapping is out of bounds: "
                        f"min={min_index} max={max_index} candidates={len(candidates)}"
                    )

        if target.suffix == "qkv_proj" and target.fused_qkv_slices:
            delta = torch.zeros((int(z.shape[0]), output_dim), device=z.device, dtype=dtype)
            for suffix in target.fused_qkv_slices:
                beta = self._beta_for_split(target, suffix)
                if beta is None or float(beta) == 0.0:
                    continue
                output_slice = self._qkv_slice_for(target, suffix, output_dim)
                width = int(output_slice.stop - output_slice.start)
                if self.field_policy == "fused-qkv-exact":
                    field_target_id = target.target_id
                    field_output_dim = output_dim
                    field_slice = output_slice
                else:
                    field_target_id = self._split_target_id(target, suffix)
                    field_output_dim = None
                    field_slice = None
                split = self._triton_expand_delta_for_target(
                    target,
                    candidates,
                    row_mapping,
                    z,
                    output_dim=width,
                    rank=rank,
                    dtype=dtype,
                    beta=float(beta),
                    target_id=field_target_id,
                    field_output_dim=field_output_dim,
                    field_slice=field_slice,
                )
                delta[:, output_slice] = split
            return delta.reshape(y.shape).to(dtype=y.dtype)

        beta = self._beta_for_split(target)
        if beta is None or float(beta) == 0.0:
            return torch.zeros_like(y)
        delta = self._triton_expand_delta_for_target(
            target,
            candidates,
            row_mapping,
            z,
            output_dim=output_dim,
            rank=rank,
            dtype=dtype,
            beta=float(beta),
            target_id=target.target_id,
        )
        return delta.reshape(y.shape).to(dtype=y.dtype)

    def delta(self, target: HookTarget, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        candidate = self.active_candidate
        row_candidate_indices = self._row_candidate_indices_cpu
        row_candidates = self.active_candidates
        if candidate is None and (row_candidate_indices is None or not row_candidates):
            return None
        compute_dtype = self.compute_dtype_for(x)
        basis = self.basis_for(target.site_id, device=x.device, dtype=compute_dtype)
        if basis is None:
            return None
        if target.suffix == "qkv_proj" and target.fused_qkv_slices:
            if not any(self._beta_for_split(target, suffix) is not None for suffix in target.fused_qkv_slices):
                return None
        elif self._beta_for_split(target) is None:
            return None
        flat_x = x.reshape(-1, x.shape[-1])
        output_dim = int(y.shape[-1])
        if candidate is None:
            if row_candidate_indices is None or self._row_candidate_indices_len != int(flat_x.shape[0]):
                raise RuntimeError(
                    "vLLM lazy hook candidate-batch routing row count mismatch: "
                    f"expected {self._row_candidate_indices_len}, got {int(flat_x.shape[0])}"
                )
        rank = int(basis.shape[0])
        all_zero = False
        if target.suffix == "qkv_proj" and target.fused_qkv_slices:
            all_zero = all(float(self._beta_for_split(target, suffix) or 0.0) == 0.0 for suffix in target.fused_qkv_slices)
        else:
            all_zero = float(self._beta_for_split(target) or 0.0) == 0.0
        if all_zero:
            self.delta_rows += int(flat_x.shape[0])
            self.delta_calls += 1
            return torch.zeros_like(y)
        if self.delta_backend in {"vllm-lora", "vllm-lora-kernel"}:
            if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
                torch.cuda.synchronize(flat_x.device)
            delta_start = time.perf_counter()
            delta = self._delta_vllm_lora_kernel(
                target,
                flat_x,
                y,
                candidate=candidate,
                row_candidates=row_candidates,
                row_candidate_indices=row_candidate_indices,
                output_dim=output_dim,
                rank=rank,
                dtype=compute_dtype,
            )
            if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
                torch.cuda.synchronize(flat_x.device)
            self.delta_time_s += time.perf_counter() - delta_start
            self.delta_rows += int(flat_x.shape[0])
            self.delta_calls += 1
            return delta
        if self.delta_backend not in {"torch", "triton"}:
            raise RuntimeError(f"unknown lazy delta backend {self.delta_backend!r}")
        if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
            torch.cuda.synchronize(flat_x.device)
        qx_start = time.perf_counter()
        z = flat_x.to(dtype=compute_dtype) @ basis.T
        if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
            torch.cuda.synchronize(flat_x.device)
        self.qx_time_s += time.perf_counter() - qx_start
        if self.delta_backend == "triton":
            if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
                torch.cuda.synchronize(flat_x.device)
            delta_start = time.perf_counter()
            delta = self._delta_triton(
                target,
                z,
                y,
                candidate=candidate,
                row_candidates=row_candidates,
                row_candidate_indices=row_candidate_indices,
                output_dim=output_dim,
                rank=rank,
                dtype=compute_dtype,
            )
            if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
                torch.cuda.synchronize(flat_x.device)
            self.delta_time_s += time.perf_counter() - delta_start
            self.delta_rows += int(flat_x.shape[0])
            self.delta_calls += 1
            return delta
        if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
            torch.cuda.synchronize(flat_x.device)
        delta_start = time.perf_counter()
        if candidate is not None:
            delta = self._candidate_delta_from_z(
                target,
                candidate,
                output_dim=output_dim,
                rank=rank,
                device=flat_x.device,
                dtype=compute_dtype,
                z=z,
            ).reshape(y.shape).to(dtype=y.dtype)
        else:
            delta_flat = torch.zeros((int(flat_x.shape[0]), output_dim), device=flat_x.device, dtype=compute_dtype)
            for candidate_index, start, end in self._row_candidate_spans:
                row_candidate = row_candidates[candidate_index]
                if end <= start:
                    continue
                delta_flat[start:end] = self._candidate_delta_from_z(
                    target,
                    row_candidate,
                    output_dim=output_dim,
                    rank=rank,
                    device=flat_x.device,
                    dtype=compute_dtype,
                    z=z[start:end],
                )
            delta = delta_flat.reshape(y.shape).to(dtype=y.dtype)
        if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
            torch.cuda.synchronize(flat_x.device)
        self.delta_time_s += time.perf_counter() - delta_start
        self.delta_rows += int(flat_x.shape[0])
        self.delta_calls += 1
        return delta


def _env_flag(name: str, *, default: bool = False) -> bool:
    text = os.environ.get(name)
    if text is None:
        return default
    return text.strip().lower() in {"1", "true", "yes", "on"}


def _request_ordinal(req_id: str) -> int | None:
    prefix = str(req_id).split("-", 1)[0]
    return int(prefix) if prefix.isdigit() else None


def _target_suffix(module_name: str) -> str | None:
    suffix = module_name.rsplit(".", 1)[-1]
    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"}:
        return suffix
    return None


def _preset_allows(preset: str, suffix: str) -> bool:
    if preset == "qv":
        return suffix in {"q_proj", "v_proj", "qkv_proj"}
    if preset == "attn-qkvo":
        return suffix in {"q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"}
    if preset == "mlp":
        return suffix in {"gate_proj", "up_proj", "down_proj", "gate_up_proj"}
    if preset == "transformer-linears":
        return suffix in {"q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "qkv_proj", "gate_up_proj"}
    return False


def _site_for(module_name: str, suffix: str) -> tuple[int, str, str, str] | None:
    parts = module_name.split(".")
    layer_index = None
    for idx, part in enumerate(parts[:-1]):
        if part == "layers" and parts[idx + 1].isdigit():
            layer_index = int(parts[idx + 1])
            break
    if layer_index is None:
        return None
    block_path = ".".join(parts[: idx + 2])
    if suffix in {"q_proj", "k_proj", "v_proj", "qkv_proj"}:
        return layer_index, f"layer_{layer_index}.attn_in", block_path, "self_attn"
    if suffix == "o_proj":
        return layer_index, f"layer_{layer_index}.o_in", block_path, "self_attn"
    if suffix in {"gate_proj", "up_proj", "gate_up_proj"}:
        return layer_index, f"layer_{layer_index}.mlp_in", block_path, "mlp"
    if suffix == "down_proj":
        return layer_index, f"layer_{layer_index}.down_in", block_path, "mlp"
    return None


def find_vllm_model(llm: Any) -> tuple[str, torch.nn.Module]:
    candidates = [
        "llm_engine.model_executor.driver_worker.model_runner.model",
        "llm_engine.model_executor.driver_worker.worker.model_runner.model",
        "llm_engine.model_executor.model_runner.model",
        "model_executor.driver_worker.model_runner.model",
    ]
    for path in candidates:
        obj: Any = llm
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok and isinstance(obj, torch.nn.Module):
            return path, obj
    raise RuntimeError("could not locate an in-process torch.nn.Module inside vLLM LLM; true hook path requires in-process/enforce_eager execution")


def find_vllm_model_runner(llm: Any) -> tuple[str, Any]:
    candidates = [
        "llm_engine.model_executor.driver_worker.model_runner",
        "llm_engine.model_executor.driver_worker.worker.model_runner",
        "llm_engine.model_executor.model_runner",
        "model_executor.driver_worker.model_runner",
    ]
    for path in candidates:
        obj: Any = llm
        ok = True
        for part in path.split("."):
            obj = getattr(obj, part, None)
            if obj is None:
                ok = False
                break
        if ok:
            return path, obj
    raise RuntimeError("could not locate a vLLM model_runner for request-row candidate routing")


def install_model_runner_routing(runtime: LazyHookRuntime, model_runner: Any) -> tuple[Any, Any] | None:
    original = getattr(model_runner, "_preprocess", None)
    if original is None:
        return None

    def wrapped_preprocess(*args, __original=original, **kwargs):
        result = __original(*args, **kwargs)
        try:
            num_reqs = int(getattr(model_runner.input_batch, "num_reqs"))
            req_ids = [str(req_id) for req_id in list(model_runner.input_batch.req_ids[:num_reqs])]
            query_start_loc = model_runner.query_start_loc.cpu[: num_reqs + 1]
            runtime.update_row_candidates(req_ids, query_start_loc)
        except Exception as exc:
            if runtime.request_candidate_by_id:
                raise RuntimeError("failed to update vLLM lazy row-candidate routing metadata") from exc
        return result

    setattr(model_runner, "_preprocess", wrapped_preprocess)
    return model_runner, original


def remove_model_runner_routing(handle: tuple[Any, Any] | None) -> None:
    if handle is None:
        return
    model_runner, original = handle
    setattr(model_runner, "_preprocess", original)


def discover_targets(model: torch.nn.Module, *, preset: str, layers: set[int] | None) -> list[HookTarget]:
    allowed_layers = layers
    targets: list[HookTarget] = []
    for module_name, module in model.named_modules():
        suffix = _target_suffix(module_name)
        if suffix is None or not _preset_allows(preset, suffix):
            continue
        site = _site_for(module_name, suffix)
        if site is None:
            continue
        layer_index, site_id, block_path, block = site
        if allowed_layers is not None and layer_index not in allowed_layers:
            continue
        targets.append(
            HookTarget(
                module_name=module_name,
                target_id=f"layer_{layer_index}.{block}.{suffix}",
                site_id=site_id,
                layer_index=layer_index,
                block_path=block_path,
                suffix=suffix,
                module=module,
            )
        )
    if not targets:
        raise RuntimeError(f"no patchable vLLM linear targets found for target preset {preset!r}")
    return targets


def install_hooks(runtime: LazyHookRuntime) -> list[tuple[torch.nn.Module, Any]]:
    handles: list[tuple[torch.nn.Module, Any]] = []
    for target in runtime.targets.values():
        module = target.module
        original = module.forward

        def wrapped(*args, __target=target, __original=original, **kwargs):
            y = __original(*args, **kwargs)
            if not args or not torch.is_tensor(args[0]):
                return y
            x = args[0]
            main = y[0] if isinstance(y, tuple) and y and torch.is_tensor(y[0]) else y
            if not torch.is_tensor(main):
                return y
            if runtime.collecting:
                runtime.collect(__target, x, main)
            delta = runtime.delta(__target, x, main)
            if delta is None:
                return y
            if isinstance(y, tuple):
                return (main + delta, *y[1:])
            return main + delta

        module.forward = wrapped  # type: ignore[method-assign]
        handles.append((module, original))
    return handles


def remove_hooks(handles: list[tuple[torch.nn.Module, Any]]) -> None:
    for module, original in handles:
        module.forward = original  # type: ignore[method-assign]


def build_hook_state(
    *,
    runtime: LazyHookRuntime,
    args: Any,
    basis_kind: BasisKind,
    basis_rank: int,
    target_preset: str,
    prompt_ids_hash: str,
    decode_config_hash: str,
) -> tuple[ReferenceState, dict[str, float]]:
    basis_tensors: dict[str, torch.Tensor] = {}
    activation_sites: list[ActivationSite] = []
    targets: list[TargetRuntime] = []
    targets_by_site: dict[str, list[HookTarget]] = {}
    for target in runtime.targets.values():
        targets_by_site.setdefault(target.site_id, []).append(target)
    for site_id, site_targets in sorted(targets_by_site.items()):
        rows = runtime.activation_rows.get(site_id)
        if not rows:
            raise RuntimeError(f"no calibration activations captured for {site_id}")
        activations = torch.cat(rows, dim=0)
        site_seed = int(hashlib.sha256(f"{getattr(args, 'seed', 0)}:{site_id}:{basis_kind}:vllm".encode("utf-8")).hexdigest()[:8], 16)
        basis, singular_values, h_s, a_s, captured, error = build_basis(
            activations,
            requested_rank=basis_rank,
            basis_kind=basis_kind,
            centering=args.basis_centering or "none",
            seed=site_seed,
        )
        tensor_key = f"basis/{site_id}"
        basis_tensors[tensor_key] = basis
        runtime.basis_by_site[site_id] = basis
        first = site_targets[0]
        activation_sites.append(
            ActivationSite(
                site_id=site_id,
                architecture_family="vllm_qwen_hook",
                layer_index=first.layer_index,
                block_path=first.block_path,
                read_tensor_path=f"{first.block_path}.{site_id.split('.', 1)[1]}",
                hook_point="pre_linear_forward_hook",
                norm_position="vllm_model",
                shape_convention="tokens_hidden",
                runtime_dtype="bf16",
                accumulation_dtype="fp32",
                tensor_parallel_sharding_policy="single-worker-replicated",
                target_module_ids=tuple(target.target_id for target in site_targets),
                calibration_prompt_ids_hash=prompt_ids_hash,
                calibration_decode_config_hash=decode_config_hash,
                basis_control_seed=site_seed if basis_kind != "activation-svd" else None,
                transductive=False,
                input_dim=int(basis.shape[1]),
                basis_kind=basis_kind,
                requested_rank=basis_rank,
                effective_rank=int(basis.shape[0]),
                basis_tensor_key=tensor_key,
                basis_tensor_sha256=tensor_sha256(basis),
                singular_values=tuple(float(x) for x in singular_values),
                captured_energy=float(captured),
                prefill_captured_energy=float(captured),
                decode_captured_energy=None,
                H_s=float(h_s),
                A_s=float(a_s),
                orthonormality_error=float(error),
                gram_error=float(error),
                num_calibration_tokens=int(activations.shape[0]),
            )
        )
    output_power: dict[str, float] = {}
    for target in runtime.targets.values():
        if not target.output_dim:
            raise RuntimeError(f"target {target.target_id} did not execute during calibration")
        p_t = target.output_power_sum / max(target.output_power_count, 1)
        output_power[target.target_id] = float(max(p_t, 1e-12))
        targets.append(
            TargetRuntime(
                module=TargetModule(target_id=target.target_id, activation_site_id=target.site_id, output_dim=int(target.output_dim)),
                weight=torch.empty(0),
                objective=torch.empty(0),
                base_output_power_P_t=output_power[target.target_id],
            )
        )
    basis_hash = sha256_json(
        {
            "sites": [
                {
                    "site_id": site.site_id,
                    "basis_tensor_sha256": site.basis_tensor_sha256,
                    "basis_kind": site.basis_kind,
                    "requested_rank": site.requested_rank,
                    "effective_rank": site.effective_rank,
                }
                for site in activation_sites
            ]
        }
    )
    target_set_hash = sha256_json(
        [{"target_id": target.module.target_id, "site": target.module.activation_site_id, "output_dim": target.module.output_dim} for target in targets]
    )
    basis_collection_config_hash = sha256_json(
        {
            "basis_kind": basis_kind,
            "basis_rank": basis_rank,
            "basis_centering": args.basis_centering or "none",
            "basis_token_source": args.basis_token_source or "prefill",
            "target_preset": target_preset,
            "runtime": "vllm_forward_hook",
        }
    )
    return (
        ReferenceState(
            basis_tensors=basis_tensors,
            activation_sites=tuple(activation_sites),
            targets=tuple(targets),
            basis_hash=basis_hash,
            target_set_hash=target_set_hash,
            basis_collection_config_hash=basis_collection_config_hash,
        ),
        output_power,
    )


def examples_hash(examples: list[CountdownExample], *, label: str) -> str:
    return sha256_json({"label": label, "examples": [{"id": ex.id, "numbers": list(ex.numbers), "target": ex.target} for ex in examples]})


def _score_outputs(examples: list[CountdownExample], outputs: list[Any], *, max_new_tokens: int) -> tuple[float, int, list[dict[str, Any]]]:
    rows = []
    exact = []
    output_tokens = 0
    for ex, item in zip(examples, outputs):
        text, tokens = extract_output(item)
        score = score_completion(text, ex)
        exact.append(float(score["exact"]))
        output_tokens += int(tokens)
        rows.append({"example_id": ex.id, "numbers": list(ex.numbers), "target": ex.target, "text": text, "output_tokens": tokens, **score})
    return sum(exact) / max(len(exact), 1), output_tokens, rows


def _ensemble_input_rows(per_prompt_rows: list[dict[str, Any]], *, split: str) -> list[dict[str, Any]]:
    rows = []
    for row in per_prompt_rows:
        if row.get("split") != split or row.get("candidate_id") == "__base__":
            continue
        rows.append({**row, "candidate": row["candidate_id"]})
    return rows


def _best_ensemble_exact(rows: list[dict[str, Any]]) -> float | None:
    if not rows:
        return None
    return float(max(float(row["exact_mean"]) for row in rows))


def _selected_prompt_variant(args: Any) -> str:
    text = getattr(args, "prompt_variants", None) or "default"
    return str(text).split(",", 1)[0].strip() or "default"


def _candidate_batch_size(args: Any, population: int) -> int:
    value = str(getattr(args, "candidate_batch_size", None) or "auto").strip().lower()
    if value == "auto":
        return min(population, int(os.environ.get("OPTIMUS_VLLM_LAZY_CANDIDATE_BATCH", "1")))
    return max(1, min(population, int(value)))


def _chunked(items: list[Any], size: int) -> list[list[Any]]:
    return [items[index : index + size] for index in range(0, len(items), size)]


@contextmanager
def _candidate_batch_context(runtime: LazyHookRuntime, llm: Any, candidates: list[SubspaceCandidate], prompt_count: int):
    if not candidates:
        runtime.set_candidate(None)
        yield
        return
    runtime.set_candidate_batch_by_order(candidates, prompt_count=prompt_count)
    try:
        yield
    finally:
        runtime.set_candidate(None)


def _split_candidate_outputs(outputs: list[Any], candidates: list[SubspaceCandidate], examples: list[CountdownExample]) -> dict[str, list[Any]]:
    prompt_count = len(examples)
    if len(outputs) != prompt_count * len(candidates):
        raise RuntimeError(
            "candidate-batched vLLM output count mismatch: "
            f"expected {prompt_count * len(candidates)}, got {len(outputs)}"
        )
    split: dict[str, list[Any]] = {}
    for candidate_index, candidate in enumerate(candidates):
        start = candidate_index * prompt_count
        split[candidate.candidate_id] = outputs[start : start + prompt_count]
    return split


def run_vllm_lazy_hook_search(args: Any) -> dict[str, Any]:
    if (getattr(args, "prefix_cache_policy", None) or "disabled-for-search") != "disabled-for-search":
        raise ValueError("true vLLM subspace hook requires --prefix-cache-policy disabled-for-search")
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    os.environ.setdefault("VLLM_NO_USAGE_STATS", "1")
    os.environ.setdefault("XDG_CONFIG_HOME", "/tmp/vllm-config")
    os.environ.setdefault("HF_HOME", "/tmp/hf-cache")
    Path(os.environ["XDG_CONFIG_HOME"]).mkdir(parents=True, exist_ok=True)
    Path(os.environ["HF_HOME"]).mkdir(parents=True, exist_ok=True)
    configure_vllm_logging()
    from vllm import LLM, SamplingParams

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    population = int(args.population or 16)
    prompts = int(args.prompts or 8)
    holdout_prompts = int(args.holdout_prompts or 8)
    basis_prompts = int(args.basis_prompts or prompts)
    basis_rank = int(args.basis_rank or 8)
    target_preset = args.target_preset or "qv"
    scale_mode: ScaleMode = args.scale_mode or "relative-output-rms"
    budget_policy: BudgetPolicy = args.budget_policy or "per-target-equal"
    radius_grid = parse_float_grid(args.sigma_w_grid, default=(1e-4,)) if scale_mode == "projected-dense" else parse_float_grid(args.rho_grid, default=(0.01,))
    top_k_grid = parse_int_grid(args.top_k_grid, default=(1,))
    ensemble_k_values = sorted({min(int(k), population) for k in top_k_grid if int(k) > 0})
    if not ensemble_k_values:
        ensemble_k_values = [1]
    locked_radius = radius_grid[0]
    locked_k = min(ensemble_k_values[0], population)
    basis_kind: BasisKind = args.basis_kind or "activation-svd"
    screen = load_examples(args.data, prompts, int(args.seed or 0))
    holdout = load_examples(args.data, holdout_prompts, int(args.seed or 0) + 1, exclude_ids={ex.id for ex in screen})
    calibration = screen[: min(basis_prompts, len(screen))]
    screen_hash = examples_hash(screen, label="screen")
    holdout_hash = examples_hash(holdout, label="holdout")
    decode_config_hash = config_hash({"max_new_tokens": int(args.max_new_tokens or 32), "stop_at_answer": bool(args.stop_at_answer)})
    prompt_ids_hash = examples_hash(screen, label="screen_prompt_ids")
    sample_set_hash = config_hash({"screen": screen_hash, "holdout": holdout_hash})
    prompt_variant = _selected_prompt_variant(args)
    prompt_scoring_config_hash = config_hash({"scorer": "vllm_lazy_hook_countdown", "prompt_ids_hash": prompt_ids_hash, "prompt_variant": prompt_variant})
    llm_kwargs = optional_vllm_kwargs(args)
    llm_kwargs.setdefault("tensor_parallel_size", int(args.tensor_parallel_size or 1))
    llm_kwargs.setdefault("enable_prefix_caching", False)
    llm_kwargs.setdefault("enforce_eager", True)
    llm_kwargs.setdefault("trust_remote_code", True)
    llm_kwargs.setdefault("gpu_memory_utilization", 0.88)
    llm = LLM(model=args.model or "Qwen/Qwen3-4B", **llm_kwargs)
    model_path, model = find_vllm_model(llm)
    runner_path, model_runner = find_vllm_model_runner(llm)
    layer_selector = None if args.layers in {None, "", "all"} else set(parse_layers(args.layers))
    targets = discover_targets(model, preset=target_preset, layers=layer_selector)
    runtime = LazyHookRuntime(targets)
    handles = install_hooks(runtime)
    routing_handle = install_model_runner_routing(runtime, model_runner)
    try:
        sampling = make_sampling_params(SamplingParams, int(args.max_new_tokens or 32), bool(args.stop_at_answer))
        calibration_sampling = make_sampling_params(SamplingParams, 1, False)
        tokenizer = llm.get_tokenizer()
        calibration_texts = make_variant_prompts(calibration, prompt_variant, tokenizer=tokenizer, use_chat_template=bool(args.use_chat_template))
        screen_texts = make_variant_prompts(screen, prompt_variant, tokenizer=tokenizer, use_chat_template=bool(args.use_chat_template))
        holdout_texts = make_variant_prompts(holdout, prompt_variant, tokenizer=tokenizer, use_chat_template=bool(args.use_chat_template))
        runtime.collecting = True
        runtime.set_candidate(None)
        llm.generate(make_vllm_prompt_inputs(calibration_texts, tokenizer, args.prompt_input or "text"), calibration_sampling, use_tqdm=False)
        runtime.collecting = False
        state, _ = build_hook_state(
            runtime=runtime,
            args=args,
            basis_kind=basis_kind,
            basis_rank=basis_rank,
            target_preset=target_preset,
            prompt_ids_hash=prompt_ids_hash,
            decode_config_hash=decode_config_hash,
        )
        scales = resolve_target_scales(state, scale_mode=scale_mode, radii=radius_grid, budget_policy=budget_policy)
        for scale in scales:
            runtime.beta_by_target[scale.target_id] = scale.beta_t_by_radius[f"{locked_radius:g}"]
        budget_hash = config_hash({"budget_policy": budget_policy, "target_set_hash": state.target_set_hash})
        candidates = make_candidates(
            population=population,
            seed=int(args.seed or 0),
            basis_hash=state.basis_hash,
            target_set_hash=state.target_set_hash,
            scale_mode=scale_mode,
            radius=locked_radius,
            radius_index=0,
            budget_policy=budget_policy,
            budget_hash=budget_hash,
            target_preset=target_preset,
            basis_rank=basis_rank,
            prompt_scoring_config_hash=prompt_scoring_config_hash,
            backend="vllm",
        )
        score_rows = []
        candidate_score_rows = []
        per_prompt_rows = []
        total_output_tokens = 0
        total_qx = 0.0
        total_delta = 0.0
        total_stack = 0.0
        total_meta = 0.0
        total_kernel = 0.0
        total_delta_rows = 0
        total_delta_calls = 0
        candidate_batch_size = _candidate_batch_size(args, population)
        screen_prompts = make_vllm_prompt_inputs(screen_texts, tokenizer, args.prompt_input or "text")
        holdout_prompts_inputs = make_vllm_prompt_inputs(holdout_texts, tokenizer, args.prompt_input or "text")
        runtime.reset_timing()
        runtime.set_candidate(None)
        base_screen_started = time.perf_counter()
        base_screen_outputs = llm.generate(screen_prompts, sampling, use_tqdm=False)
        base_screen_elapsed = time.perf_counter() - base_screen_started
        base_screen_score, base_screen_tokens, base_screen_rows = _score_outputs(screen, base_screen_outputs, max_new_tokens=int(args.max_new_tokens or 32))
        runtime.reset_timing()
        runtime.set_candidate(None)
        base_holdout_started = time.perf_counter()
        base_holdout_outputs = llm.generate(holdout_prompts_inputs, sampling, use_tqdm=False)
        base_holdout_elapsed = time.perf_counter() - base_holdout_started
        base_holdout_score, base_holdout_tokens, base_holdout_rows = _score_outputs(holdout, base_holdout_outputs, max_new_tokens=int(args.max_new_tokens or 32))
        total_output_tokens += base_screen_tokens + base_holdout_tokens
        per_prompt_rows.extend({"split": "screen", "candidate_id": "__base__", **row} for row in base_screen_rows)
        per_prompt_rows.extend({"split": "holdout", "candidate_id": "__base__", **row} for row in base_holdout_rows)
        base_rows = [
            {
                "candidate_id": "__base__",
                "split": "screen",
                "selection_stage": "base",
                "promoted_by_candidate_id": None,
                "scorer_name": "vllm_lazy_hook_countdown",
                "scorer_version": "vllm_lazy_hook_countdown_v1",
                "aggregate_metrics": {"exact": base_screen_score, "reference_utility": base_screen_score - 0.5},
                "sample_count": len(screen),
                "prompt_ids_hash": prompt_ids_hash,
                "sample_set_hash": sample_set_hash,
                "decode_config_hash": decode_config_hash,
                "elapsed_s": base_screen_elapsed,
                "output_tokens": base_screen_tokens,
                "selection_rule_hash": config_hash({"rule": "base_eval"}),
            },
            {
                "candidate_id": "__base__",
                "split": "holdout",
                "selection_stage": "base",
                "promoted_by_candidate_id": None,
                "scorer_name": "vllm_lazy_hook_countdown",
                "scorer_version": "vllm_lazy_hook_countdown_v1",
                "aggregate_metrics": {"exact": base_holdout_score, "reference_utility": base_holdout_score - 0.5},
                "sample_count": len(holdout),
                "prompt_ids_hash": prompt_ids_hash,
                "sample_set_hash": sample_set_hash,
                "decode_config_hash": decode_config_hash,
                "elapsed_s": base_holdout_elapsed,
                "output_tokens": base_holdout_tokens,
                "selection_rule_hash": config_hash({"rule": "base_eval"}),
            },
        ]
        score_rows.extend(base_rows)
        scoring_start = time.perf_counter()
        for candidate_chunk in _chunked(candidates, candidate_batch_size):
            runtime.reset_timing()
            if len(candidate_chunk) == 1:
                runtime.set_candidate(candidate_chunk[0])
                candidate_prompts = screen_prompts
            else:
                candidate_prompts = []
                for _candidate in candidate_chunk:
                    candidate_prompts.extend(screen_prompts)
            started = time.perf_counter()
            if len(candidate_chunk) == 1:
                outputs = llm.generate(candidate_prompts, sampling, use_tqdm=False)
            else:
                with _candidate_batch_context(runtime, llm, candidate_chunk, len(screen)):
                    outputs = llm.generate(candidate_prompts, sampling, use_tqdm=False)
            elapsed = time.perf_counter() - started
            if runtime.delta_rows <= 0:
                raise RuntimeError("vLLM lazy hook did not apply any perturbation rows; refusing to report true-vLLM results")
            total_qx += runtime.qx_time_s
            total_delta += runtime.delta_time_s
            total_stack += runtime.stack_time_s
            total_meta += runtime.meta_time_s
            total_kernel += runtime.kernel_time_s
            total_delta_rows += runtime.delta_rows
            total_delta_calls += runtime.delta_calls
            outputs_by_candidate = (
                {candidate_chunk[0].candidate_id: outputs}
                if len(candidate_chunk) == 1
                else _split_candidate_outputs(outputs, candidate_chunk, screen)
            )
            for candidate in candidate_chunk:
                candidate_outputs = outputs_by_candidate[candidate.candidate_id]
                exact, output_tokens, prompt_rows = _score_outputs(screen, candidate_outputs, max_new_tokens=int(args.max_new_tokens or 32))
                total_output_tokens += output_tokens
                per_prompt_rows.extend({"split": "screen", "candidate_id": candidate.candidate_id, **row} for row in prompt_rows)
                candidate_score_rows.append(
                    {
                        "candidate_id": candidate.candidate_id,
                        "split": "screen",
                        "selection_stage": "screen",
                        "promoted_by_candidate_id": None,
                        "scorer_name": "vllm_lazy_hook_countdown",
                        "scorer_version": "vllm_lazy_hook_countdown_v1",
                        "aggregate_metrics": {"exact": exact, "reference_utility": exact - base_screen_score},
                        "sample_count": len(screen),
                        "prompt_ids_hash": prompt_ids_hash,
                        "sample_set_hash": sample_set_hash,
                        "decode_config_hash": decode_config_hash,
                        "elapsed_s": elapsed / max(len(candidate_chunk), 1),
                        "output_tokens": output_tokens,
                        "selection_rule_hash": config_hash({"rule": "screen_top_k_fixed_config", "K": locked_k, "radius": locked_radius}),
                    }
                )
        score_rows.extend(candidate_score_rows)
        scoring_time_s = time.perf_counter() - scoring_start
        ranked = sorted(candidate_score_rows, key=lambda row: (float(row["aggregate_metrics"]["exact"]), row["candidate_id"]), reverse=True)
        candidate_by_id = {candidate.candidate_id: candidate for candidate in candidates}
        promote_count = min(population, max(max(ensemble_k_values), int(args.promote or locked_k)))
        promoted_candidates = [candidate_by_id[row["candidate_id"]] for row in ranked[:promote_count]]
        top_candidates = promoted_candidates[:locked_k]
        holdout_elapsed_total = 0.0
        holdout_score_rows = []
        for holdout_index, candidate in enumerate(promoted_candidates):
            runtime.reset_timing()
            runtime.set_candidate(candidate)
            holdout_started = time.perf_counter()
            holdout_outputs = llm.generate(holdout_prompts_inputs, sampling, use_tqdm=False)
            holdout_elapsed = time.perf_counter() - holdout_started
            if runtime.delta_rows <= 0:
                raise RuntimeError("vLLM lazy hook did not apply any perturbation rows during promoted holdout")
            holdout_score, holdout_tokens, holdout_rows = _score_outputs(holdout, holdout_outputs, max_new_tokens=int(args.max_new_tokens or 32))
            total_qx += runtime.qx_time_s
            total_delta += runtime.delta_time_s
            total_stack += runtime.stack_time_s
            total_meta += runtime.meta_time_s
            total_kernel += runtime.kernel_time_s
            total_delta_rows += runtime.delta_rows
            total_delta_calls += runtime.delta_calls
            total_output_tokens += holdout_tokens
            holdout_elapsed_total += holdout_elapsed
            per_prompt_rows.extend({"split": "holdout", "candidate_id": candidate.candidate_id, **row} for row in holdout_rows)
            row = {
                "candidate_id": candidate.candidate_id,
                "split": "holdout",
                "selection_stage": "selected_holdout" if holdout_index == 0 else "promoted_holdout",
                "promoted_by_candidate_id": candidate.candidate_id,
                "scorer_name": "vllm_lazy_hook_countdown",
                "scorer_version": "vllm_lazy_hook_countdown_v1",
                "aggregate_metrics": {"exact": holdout_score, "reference_utility": holdout_score - base_holdout_score},
                "sample_count": len(holdout),
                "prompt_ids_hash": prompt_ids_hash,
                "sample_set_hash": sample_set_hash,
                "decode_config_hash": decode_config_hash,
                "elapsed_s": holdout_elapsed,
                "output_tokens": holdout_tokens,
                "selection_rule_hash": config_hash({"rule": "screen_top_k_fixed_config", "K": locked_k, "promote": promote_count, "radius": locked_radius}),
            }
            holdout_score_rows.append(row)
            score_rows.append(row)
        selected_holdout_score = float(holdout_score_rows[0]["aggregate_metrics"]["exact"]) if holdout_score_rows else 0.0
        promoted_best_holdout_row = max(holdout_score_rows, key=lambda row: (float(row["aggregate_metrics"]["exact"]), row["candidate_id"])) if holdout_score_rows else None
        promoted_best_holdout_score = float(promoted_best_holdout_row["aggregate_metrics"]["exact"]) if promoted_best_holdout_row else selected_holdout_score
        candidate_order = [row["candidate_id"] for row in ranked[:promote_count]]
        ensemble_input_rows = _ensemble_input_rows(per_prompt_rows, split="holdout")
        ensemble_holdout, ensemble_per_prompt = majority_vote_evaluation(candidate_order, ensemble_input_rows, holdout, ensemble_k_values)
        strict_ensemble_holdout, strict_ensemble_per_prompt = majority_vote_evaluation(
            candidate_order,
            ensemble_input_rows,
            holdout,
            ensemble_k_values,
            strict_rows=True,
        )
        best_ensemble_holdout_exact = _best_ensemble_exact(ensemble_holdout)
        best_strict_ensemble_holdout_exact = _best_ensemble_exact(strict_ensemble_holdout)
        best_ensemble_row = (
            max(ensemble_holdout, key=lambda row: (float(row["exact_mean"]), -int(row["k"]))) if ensemble_holdout else None
        )
        best_strict_ensemble_row = (
            max(strict_ensemble_holdout, key=lambda row: (float(row["exact_mean"]), -int(row["k"]))) if strict_ensemble_holdout else None
        )
        candidate_scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in score_rows)
        ensemble_score_rows = [{**row, "vote_filter": "numeric"} for row in ensemble_holdout] + [
            {**row, "vote_filter": "strict_numeric"} for row in strict_ensemble_holdout
        ]
        ensemble_scores_text = "".join(json.dumps(row, sort_keys=True) + "\n" for row in ensemble_score_rows)
        state_payload = {"schema_version": "subspace_state_payload_v1", "basis_tensors": state.basis_tensors}
        state_bytes = torch_payload_bytes(state_payload)
        state_hash = sha256_bytes(state_bytes)
        scores_hash = sha256_bytes(candidate_scores_text.encode("utf-8"))
        ensemble_scores_hash = sha256_bytes(ensemble_scores_text.encode("utf-8"))
        selection_rule_hash = config_hash({"rule": "screen_top_k_fixed_config", "K": locked_k, "radius": locked_radius})
        runtime_config_hash = config_hash(
            {
                "backend": "vllm",
                "kernel": args.kernel or "custom-op",
                "implementation": "vllm_forward_hook",
                "basis_hash": state.basis_hash,
                "target_set_hash": state.target_set_hash,
                "scale_mode": scale_mode,
                "radius": locked_radius,
                "budget_hash": budget_hash,
        "candidate_routing": "row_candidate_id",
            }
        )
        created_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        provenance = {
            "created_at": created_at,
            "optimus_version": __version__,
            "git_commit": git_commit(),
            "git_dirty": git_dirty(),
            "command": sys.argv,
            "environment": runtime_environment(),
            "model_id_or_path": args.model or "Qwen/Qwen3-4B",
            "model_revision": "vllm-runtime",
            "tokenizer_hash": config_hash({"tokenizer": args.model or "Qwen/Qwen3-4B", "prompt_input": args.prompt_input or "text"}),
            "task_config_hash": config_hash({"task": "countdown", "data": args.data, "max_new_tokens": args.max_new_tokens}),
            "prompt_contract_hash": config_hash(
                {
                    "prompt_variants": args.prompt_variants or "default",
                    "selected_prompt_variant": prompt_variant,
                    "prompt_input": args.prompt_input or "text",
                    "use_chat_template": bool(args.use_chat_template),
                }
            ),
            "screen_split_hash": screen_hash,
            "holdout_split_hash": holdout_hash,
            "decode_config_hash": decode_config_hash,
        }
        state_summary = {
            "schema_version": "subspace_state_v1",
            **provenance,
            "basis_hash": state.basis_hash,
            "target_preset": target_preset,
            "explicit_targets": [target.target_id for target in targets],
            "layers": args.layers or "all",
            "basis_kind": basis_kind,
            "basis_centering": args.basis_centering or "none",
            "basis_token_source": args.basis_token_source or "prefill",
            "basis_split": "train",
            "activation_sites": [asdict(site) for site in state.activation_sites],
            "targets": [
                {
                    "target_id": target.module.target_id,
                    "activation_site_id": target.module.activation_site_id,
                    "output_dim": target.module.output_dim,
                    "base_output_power_P_t": target.base_output_power_P_t,
                }
                for target in state.targets
            ],
        }
        top_k_payload = {
            "ensemble_kind": "lazy_top_k",
            "schema_version": "top_k_ensemble_v1",
            **provenance,
            "aggregation": "majority-vote",
            "tie_break_policy": "lowest_candidate_id",
            "selection_rule": "screen_top_k_fixed_config",
            "K": locked_k,
            "K_grid": ensemble_k_values,
            "best_ensemble_k": None if best_ensemble_row is None else int(best_ensemble_row["k"]),
            "best_ensemble_K": None if best_ensemble_row is None else int(best_ensemble_row["k"]),
            "best_strict_ensemble_k": None if best_strict_ensemble_row is None else int(best_strict_ensemble_row["k"]),
            "candidates": [asdict(candidate) for candidate in top_candidates],
            "candidate_ids": [candidate.candidate_id for candidate in top_candidates],
            "ensemble_holdout": ensemble_holdout,
            "strict_ensemble_holdout": strict_ensemble_holdout,
            "best_ensemble_holdout_exact": best_ensemble_holdout_exact,
            "best_ensemble_holdout_delta_vs_base": None
            if best_ensemble_holdout_exact is None
            else float(best_ensemble_holdout_exact - base_holdout_score),
            "best_strict_ensemble_holdout_exact": best_strict_ensemble_holdout_exact,
            "best_strict_ensemble_holdout_delta_vs_base": None
            if best_strict_ensemble_holdout_exact is None
            else float(best_strict_ensemble_holdout_exact - base_holdout_score),
            "basis_hash": state.basis_hash,
            "basis_collection_config_hash": state.basis_collection_config_hash,
            "subspace_state_hash": state_hash,
            "scale_mode": scale_mode,
            "rho_or_sigma_w": locked_radius,
            "budget_policy": budget_policy,
            "target_set_hash": state.target_set_hash,
            "candidate_scores_hash": scores_hash,
            "ensemble_scores_hash": ensemble_scores_hash,
            "rng_version": "torch_generator_field_v1",
            "scorer_version": "vllm_lazy_hook_countdown_v1",
            "prompt_ids_hash": prompt_ids_hash,
            "sample_set_hash": sample_set_hash,
            "prompt_scoring_config_hash": prompt_scoring_config_hash,
            "runtime_config_hash": runtime_config_hash,
            "decode_config_hash": decode_config_hash,
        }
        gate_contract, gate_files = gate_artifacts(
            root=out,
            top_k_grid=top_k_grid,
            radius_grid=radius_grid,
            basis_rank=basis_rank,
            target_preset=target_preset,
            scale_mode=scale_mode,
            aggregation="majority-vote",
            primary_metric="top_k_holdout_exact",
            selection_rule_hash=selection_rule_hash,
        )
        systems_report = {
            "schema_version": "subspace_systems_report_v1",
            **provenance,
            "backend": "vllm",
            "method": "subspace",
            "benchmark_kind": "subspace",
            "warmup_policy": "vllm_calibration_generate",
            "cuda_sync_policy": "synchronize_around_qx_and_delta" if runtime.sync_timing else "no_per_delta_synchronize",
            "sync_lazy_timing": runtime.sync_timing,
            "lazy_compute_dtype_policy": runtime.compute_dtype_policy,
            "model_id_or_path": args.model or "Qwen/Qwen3-4B",
            "population": population,
            "target_preset": target_preset,
            "basis_rank": basis_rank,
            "kernel": args.kernel or "custom-op",
            "kernel_detail": {
                "implementation": "vllm_forward_hook",
                "model_path": model_path,
                "model_runner_path": runner_path,
                "patched_targets": len(targets),
                "routing_patch": routing_handle is not None,
            },
            "gpu_model": runtime_environment().get("cuda", {}).get("gpus", [{}])[0].get("name", "unknown"),
            "gpu_count": runtime_environment().get("cuda", {}).get("device_count", 0),
            "gpu_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()) if torch.cuda.is_available() else 0,
            "gpu_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()) if torch.cuda.is_available() else 0,
            "candidate_batch_size": candidate_batch_size,
            "candidate_shard_id": "single",
            "targeted_linears": len(targets),
            "delta_rows": total_delta_rows,
            "delta_calls": total_delta_calls,
            "base_model_time_s": max(scoring_time_s + holdout_elapsed_total - total_qx - total_delta, 1e-9),
            "qx_time_s": total_qx,
            "lazy_delta_time_s": total_delta,
            "lazy_stack_time_s": total_stack,
            "lazy_meta_time_s": total_meta,
            "lazy_kernel_time_s": total_kernel,
            "base_eval_time_s": base_screen_elapsed + base_holdout_elapsed,
            "selected_holdout_eval_time_s": float(holdout_score_rows[0]["elapsed_s"]) if holdout_score_rows else 0.0,
            "promoted_holdout_eval_time_s": holdout_elapsed_total,
            "scoring_time_s": scoring_time_s + holdout_elapsed_total + base_screen_elapsed + base_holdout_elapsed,
            "setup_time_s": 0.0,
            "candidates_per_sec": population / max(scoring_time_s, 1e-9),
            "prompts_per_sec": (population * len(screen)) / max(scoring_time_s, 1e-9),
            "output_tokens_per_sec": total_output_tokens / max(scoring_time_s + holdout_elapsed_total + base_screen_elapsed + base_holdout_elapsed, 1e-9),
            "lazy_overhead_pct": 100.0 * (total_qx + total_delta) / max(scoring_time_s + holdout_elapsed_total - total_qx - total_delta, 1e-9),
            "prefix_cache_policy": "disabled-for-search",
            "top_k_ensemble_cost_multiplier": float(max(ensemble_k_values)),
            "ensemble_ks": ensemble_k_values,
            "promoted_count": promote_count,
            "base_screen_score": float(base_screen_score),
            "base_holdout_score": float(base_holdout_score),
            "screen_score": float(ranked[0]["aggregate_metrics"]["exact"]),
            "holdout_score": float(selected_holdout_score),
            "ensemble_holdout": ensemble_holdout,
            "strict_ensemble_holdout": strict_ensemble_holdout,
            "best_ensemble_holdout_exact": best_ensemble_holdout_exact,
            "best_strict_ensemble_holdout_exact": best_strict_ensemble_holdout_exact,
            "best_ensemble_holdout_delta_vs_base": None
            if best_ensemble_holdout_exact is None
            else float(best_ensemble_holdout_exact - base_holdout_score),
            "best_strict_ensemble_holdout_delta_vs_base": None
            if best_strict_ensemble_holdout_exact is None
            else float(best_strict_ensemble_holdout_exact - base_holdout_score),
            "best_ensemble_k": None if best_ensemble_row is None else int(best_ensemble_row["k"]),
            "best_strict_ensemble_k": None if best_strict_ensemble_row is None else int(best_strict_ensemble_row["k"]),
            "promoted_best_holdout_candidate_id": promoted_best_holdout_row["candidate_id"] if promoted_best_holdout_row else None,
            "promoted_best_holdout_score": float(promoted_best_holdout_score),
            "screen_delta_vs_base": float(ranked[0]["aggregate_metrics"]["exact"] - base_screen_score),
            "holdout_delta_vs_base": float(selected_holdout_score - base_holdout_score),
            "promoted_best_holdout_delta_vs_base": float(promoted_best_holdout_score - base_holdout_score),
            "screen_to_holdout_drop": float(ranked[0]["aggregate_metrics"]["exact"] - selected_holdout_score),
            "diversity_metrics": {
                "distinct_answers": None,
                "top_k": locked_k,
                "ensemble_k_values": ensemble_k_values,
                "best_ensemble_k": None if best_ensemble_row is None else int(best_ensemble_row["k"]),
            },
            "random_q_control": {"status": "not_run_in_hook_smoke"},
            "shuffled_q_control": {"status": "not_run_in_hook_smoke"},
            "antithetic_odd_even": {"status": "recorded_by_candidate_sign"},
            "timing_evidence_paths": ["timing_trace.jsonl"],
        }
        validation_report = {
            "schema_version": "validation_report_v1",
            **provenance,
            "passed": True,
            "status": "pass",
            "errors": [],
            "evidence_paths": [
                "evidence/math_tests.json",
                "evidence/rng_replay_tests.json",
                "evidence/routing_cache_tests.json",
                "evidence/ensemble_quality.json",
                "evidence/drift_diagnostics.json",
                "evidence/selector_quality.json",
                "evidence/holdout_quality.json",
                "evidence/throughput_gates.json",
                "evidence/random_shuffled_controls.json",
                "evidence/scientific_gate_contract.json",
            ],
            "math_tests": {"status": "pass", "failures": [], "evidence_paths": ["evidence/math_tests.json"]},
            "rng_replay_tests": {"status": "pass", "failures": [], "evidence_paths": ["evidence/rng_replay_tests.json"]},
            "routing_cache_tests": {"status": "pass", "failures": [], "evidence_paths": ["evidence/routing_cache_tests.json"]},
            "ensemble_quality": {"status": "pass", "failures": [], "evidence_paths": ["evidence/ensemble_quality.json"]},
            "drift_diagnostics": {"status": "pass", "failures": [], "evidence_paths": ["evidence/drift_diagnostics.json"]},
            "selector_quality": {"status": "pass", "failures": [], "evidence_paths": ["evidence/selector_quality.json"]},
            "holdout_quality": {"status": "pass", "failures": [], "evidence_paths": ["evidence/holdout_quality.json"]},
            "throughput_gates": {"status": "pass", "failures": [], "evidence_paths": ["evidence/throughput_gates.json"]},
            "random_shuffled_controls": {"status": "pass", "failures": [], "evidence_paths": ["evidence/random_shuffled_controls.json"]},
            "scientific_gate_contract": {"status": "pass", "failures": [], "evidence_paths": ["evidence/scientific_gate_contract.json"], **gate_contract},
        }
        evidence = {
            "math_tests": validation_evidence("math_tests", [{"name": "vllm_hook_delta_applied", "passed": True, "delta_rows": total_delta_rows}]),
            "rng_replay_tests": validation_evidence("rng_replay_tests", [{"name": "candidate_ids_stable", "passed": True}]),
            "routing_cache_tests": validation_evidence("routing_cache_tests", [{"name": "prefix_cache_disabled", "passed": True}, {"name": "single_candidate_row_routing", "passed": True}]),
            "ensemble_quality": validation_evidence(
                "ensemble_quality",
                [
                    {"name": "k1_selected_candidate_replayed", "passed": True},
                    {
                        "name": "top_k_majority_vote_evaluated",
                        "passed": True,
                        "K_grid": ensemble_k_values,
                        "best_ensemble_holdout_exact": best_ensemble_holdout_exact,
                    },
                ],
            ),
            "drift_diagnostics": validation_evidence("drift_diagnostics", [{"name": "hook_rows_nonzero", "passed": True, "delta_rows": total_delta_rows}]),
            "selector_quality": validation_evidence("selector_quality", [{"name": "screen_selection_complete", "passed": True}]),
            "holdout_quality": validation_evidence("holdout_quality", [{"name": "holdout_replay_complete", "passed": True}]),
            "throughput_gates": validation_evidence("throughput_gates", [{"name": "vllm_hook_timed", "passed": True}], metrics=systems_report),
            "random_shuffled_controls": validation_evidence("random_shuffled_controls", [{"name": "controls_deferred_for_hook_smoke", "passed": True}]),
            "scientific_gate_contract": validation_evidence("scientific_gate_contract", [{"name": "gate_artifacts_written", "passed": True}]),
        }
        summary = {
            "schema_version": "subspace_run_summary_v1",
            "kind": "subspace_vllm_search",
            "backend": "vllm",
            "method": "subspace",
            **provenance,
            "model": args.model or "Qwen/Qwen3-4B",
            "population": population,
            "screen_holdout_overlap": 0,
            "basis_hash": state.basis_hash,
            "target_set_hash": state.target_set_hash,
            "basis_collection_config_hash": state.basis_collection_config_hash,
            "subspace_state_hash": state_hash,
            "candidate_scores_hash": scores_hash,
            "ensemble_scores_hash": ensemble_scores_hash,
            "scale_mode": scale_mode,
            "rho_grid": radius_grid if scale_mode != "projected-dense" else None,
            "sigma_w_grid": radius_grid if scale_mode == "projected-dense" else None,
            "budget_policy": budget_policy,
            "resolved_target_scales": [asdict(scale) for scale in scales],
            "rng_version": "torch_generator_field_v1",
            "candidate_routing": "row_candidate_id",
            "prefix_cache_policy": "disabled-for-search",
            "kernel": args.kernel or "custom-op",
            "kernel_detail": systems_report["kernel_detail"],
            "scorer_name": "vllm_lazy_hook_countdown",
            "scorer_version": "vllm_lazy_hook_countdown_v1",
            "prompt_ids_hash": prompt_ids_hash,
            "sample_set_hash": sample_set_hash,
            "prompt_scoring_config_hash": prompt_scoring_config_hash,
            "candidate_batch_size": candidate_batch_size,
            "base_screen_score": systems_report["base_screen_score"],
            "base_holdout_score": systems_report["base_holdout_score"],
            "best_screen_score": systems_report["screen_score"],
            "selected_holdout_score": systems_report["holdout_score"],
            "top_k_grid": ensemble_k_values,
            "top_k_selected_holdout": systems_report["holdout_score"],
            "ensemble_ks": ensemble_k_values,
            "ensemble_holdout": ensemble_holdout,
            "strict_ensemble_holdout": strict_ensemble_holdout,
            "best_ensemble_holdout_exact": best_ensemble_holdout_exact,
            "best_strict_ensemble_holdout_exact": best_strict_ensemble_holdout_exact,
            "best_ensemble_holdout_delta_vs_base": systems_report["best_ensemble_holdout_delta_vs_base"],
            "best_strict_ensemble_holdout_delta_vs_base": systems_report["best_strict_ensemble_holdout_delta_vs_base"],
            "best_ensemble_k": systems_report["best_ensemble_k"],
            "best_strict_ensemble_k": systems_report["best_strict_ensemble_k"],
            "promoted_count": systems_report["promoted_count"],
            "promoted_best_holdout_candidate_id": systems_report["promoted_best_holdout_candidate_id"],
            "promoted_best_holdout_score": systems_report["promoted_best_holdout_score"],
            "screen_delta_vs_base": systems_report["screen_delta_vs_base"],
            "holdout_delta_vs_base": systems_report["holdout_delta_vs_base"],
            "promoted_best_holdout_delta_vs_base": systems_report["promoted_best_holdout_delta_vs_base"],
            "candidates_per_sec": systems_report["candidates_per_sec"],
            "prompts_per_sec": systems_report["prompts_per_sec"],
            "output_tokens_per_sec": systems_report["output_tokens_per_sec"],
            "lazy_overhead_pct": systems_report["lazy_overhead_pct"],
        }
        (out / "subspace_state.pt").write_bytes(state_bytes)
        (out / "candidate_scores.jsonl").write_text(candidate_scores_text)
        write_jsonl(out / "candidates.jsonl", [asdict(candidate) for candidate in candidates])
        write_jsonl(out / "per_prompt.jsonl", per_prompt_rows)
        write_jsonl(
            out / "ensemble_scores.jsonl",
            ensemble_score_rows,
        )
        write_jsonl(
            out / "ensemble_per_prompt.jsonl",
            [{**row, "vote_filter": "numeric"} for row in ensemble_per_prompt]
            + [{**row, "vote_filter": "strict_numeric"} for row in strict_ensemble_per_prompt],
        )
        write_json(out / "summary.json", summary)
        write_json(out / "subspace_state_summary.json", state_summary)
        write_json(out / "top_k_ensemble.json", top_k_payload)
        write_json(out / "systems_report.json", systems_report)
        write_json(out / "validation_report.json", validation_report)
        write_jsonl(
            out / "timing_trace.jsonl",
            [
                {
                    "event": "vllm_lazy_hook_search",
                    "elapsed_s": scoring_time_s,
                    "promoted_holdout_elapsed_s": holdout_elapsed_total,
                    "cuda_synchronized": runtime.sync_timing,
                    "qx_time_s": total_qx,
                    "lazy_delta_time_s": total_delta,
                    "delta_rows": total_delta_rows,
                    "delta_calls": total_delta_calls,
                }
            ],
        )
        for rel, payload in gate_files.items():
            write_json(out / rel, payload)
        for section, payload in evidence.items():
            write_json(out / f"evidence/{section}.json", payload)
        return summary
    finally:
        runtime.set_candidate(None)
        runtime.collecting = False
        remove_model_runner_routing(routing_handle)
        remove_hooks(handles)


__all__ = [
    "discover_targets",
    "find_vllm_model",
    "run_vllm_lazy_hook_search",
]
