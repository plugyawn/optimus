from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import torch

from optimus.backends.vllm_lazy_hook import HookTarget
from optimus.core.perturbations import PerturbationSpec
from optimus.modeling.noise import lora_noise_tensors


def _dtype_from_name(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.strip().lower()]


@dataclass(frozen=True)
class LoraHookFactors:
    a: torch.Tensor
    b: torch.Tensor


@dataclass(frozen=True)
class FusedQKVSpec:
    q_out: int
    kv_out: int

    @property
    def total_out(self) -> int:
        return self.q_out + 2 * self.kv_out

    def slice_for(self, suffix: str) -> slice:
        if suffix == "q_proj":
            return slice(0, self.q_out)
        if suffix == "k_proj":
            return slice(self.q_out, self.q_out + self.kv_out)
        if suffix == "v_proj":
            return slice(self.q_out + self.kv_out, self.total_out)
        raise ValueError(f"unsupported fused qkv suffix {suffix!r}")


class LazyLoraHookRuntime:
    """Apply deterministic LoRA candidates directly in hooked vLLM linears.

    This is an adapter-free diagnostic runtime for comparing candidate law and
    serving overhead. It deliberately lives outside the public subspace backend.
    """

    def __init__(
        self,
        targets: list[HookTarget],
        *,
        rank: int,
        adapter_dtype: str = "bfloat16",
        fused_qkv: FusedQKVSpec | None = None,
        preserve_factor_cache: bool = False,
        sync_timing: bool = False,
    ) -> None:
        self.targets = {target.module_name: target for target in targets}
        self.rank = int(rank)
        self.adapter_dtype = _dtype_from_name(adapter_dtype)
        self.fused_qkv = fused_qkv
        self.preserve_factor_cache = bool(preserve_factor_cache)
        self.sync_timing = bool(sync_timing)
        self.collecting = False
        self.active_candidate: PerturbationSpec | None = None
        self.active_candidates: list[PerturbationSpec] = []
        self.request_candidate_by_id: dict[str, PerturbationSpec] = {}
        self._candidate_index_by_id: dict[str, int] = {}
        self._order_prompt_count = 0
        self._row_candidate_indices_cpu: torch.Tensor | None = None
        self._row_candidate_spans: list[tuple[int, int, int]] = []
        self._row_candidate_indices_len = 0
        self.qx_time_s = 0.0
        self.delta_time_s = 0.0
        self.delta_rows = 0
        self.delta_calls = 0
        self._factor_cache: dict[tuple[str, str, str, torch.dtype, int, int, int], LoraHookFactors] = {}

    def reset_timing(self) -> None:
        self.qx_time_s = 0.0
        self.delta_time_s = 0.0
        self.delta_rows = 0
        self.delta_calls = 0

    def set_candidate(self, candidate: PerturbationSpec | None) -> None:
        self.active_candidate = candidate
        self.active_candidates = []
        self.request_candidate_by_id = {}
        self._candidate_index_by_id = {}
        self._order_prompt_count = 0
        self._row_candidate_indices_cpu = None
        self._row_candidate_spans = []
        self._row_candidate_indices_len = 0
        if not self.preserve_factor_cache:
            self._factor_cache.clear()

    def set_candidate_batch(self, request_candidate_by_id: dict[str, PerturbationSpec]) -> None:
        self.active_candidate = None
        self.request_candidate_by_id = dict(request_candidate_by_id)
        candidates: list[PerturbationSpec] = []
        seen: set[str] = set()
        for candidate in self.request_candidate_by_id.values():
            if candidate.key in seen:
                continue
            seen.add(candidate.key)
            candidates.append(candidate)
        self.active_candidates = candidates
        self._candidate_index_by_id = {candidate.key: idx for idx, candidate in enumerate(candidates)}
        self._order_prompt_count = 0
        self._row_candidate_indices_cpu = None
        self._row_candidate_spans = []
        self._row_candidate_indices_len = 0
        if not self.preserve_factor_cache:
            self._factor_cache.clear()

    def set_candidate_batch_by_order(self, candidates: list[PerturbationSpec], *, prompt_count: int) -> None:
        self.active_candidate = None
        self.request_candidate_by_id = {}
        self.active_candidates = list(candidates)
        self._candidate_index_by_id = {candidate.key: idx for idx, candidate in enumerate(candidates)}
        self._order_prompt_count = max(1, int(prompt_count))
        self._row_candidate_indices_cpu = None
        self._row_candidate_spans = []
        self._row_candidate_indices_len = 0
        if not self.preserve_factor_cache:
            self._factor_cache.clear()

    def update_row_candidates(self, req_ids: list[str], query_start_loc: Any) -> None:
        if not self.request_candidate_by_id and not (self.active_candidates and self._order_prompt_count > 0):
            self._row_candidate_indices_cpu = None
            self._row_candidate_spans = []
            self._row_candidate_indices_len = 0
            return
        loc = query_start_loc.detach().cpu().tolist() if torch.is_tensor(query_start_loc) else list(query_start_loc)
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
                candidate_index = -1 if candidate is None else self._candidate_index_by_id[candidate.key]
            else:
                candidate_index = min(req_index // self._order_prompt_count, len(self.active_candidates) - 1)
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

    def factors_for_module(
        self,
        module_name: str,
        candidate: PerturbationSpec,
        *,
        input_dim: int,
        output_dim: int,
        device: torch.device,
    ) -> LoraHookFactors:
        key = (module_name, candidate.key, str(device), self.adapter_dtype, input_dim, output_dim, self.rank)
        cached = self._factor_cache.get(key)
        if cached is not None:
            return cached
        a, b = lora_noise_tensors(
            module_name,
            (self.rank, input_dim),
            (output_dim, self.rank),
            candidate,
            self.rank,
            state_key=module_name,
        )
        factors = LoraHookFactors(
            a=a.to(device=device, dtype=self.adapter_dtype, non_blocking=True).contiguous(),
            b=b.to(device=device, dtype=self.adapter_dtype, non_blocking=True).contiguous(),
        )
        self._factor_cache[key] = factors
        return factors

    def factors(
        self,
        target: HookTarget,
        candidate: PerturbationSpec,
        *,
        input_dim: int,
        output_dim: int,
        device: torch.device,
    ) -> LoraHookFactors:
        return self.factors_for_module(
            target.module_name,
            candidate,
            input_dim=input_dim,
            output_dim=output_dim,
            device=device,
        )

    def preload_candidate(self, candidate: PerturbationSpec) -> None:
        for target in self.targets.values():
            weight = getattr(target.module, "weight", None)
            if weight is not None and hasattr(weight, "shape") and len(weight.shape) == 2:
                output_dim = int(weight.shape[0])
                input_dim = int(weight.shape[1])
                device = weight.device
            else:
                if target.input_dim is None or target.output_dim is None:
                    continue
                input_dim = int(target.input_dim)
                output_dim = int(target.output_dim)
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if target.suffix == "qkv_proj":
                if self.fused_qkv is None:
                    continue
                requested = set(candidate.targets) or {"q_proj", "k_proj", "v_proj"}
                for suffix in ("q_proj", "k_proj", "v_proj"):
                    if suffix not in requested:
                        continue
                    output_slice = self.fused_qkv.slice_for(suffix)
                    self.factors_for_module(
                        self._module_name_for_fused_suffix(target, suffix),
                        candidate,
                        input_dim=input_dim,
                        output_dim=int(output_slice.stop - output_slice.start),
                        device=device,
                    )
            else:
                self.factors(target, candidate, input_dim=input_dim, output_dim=output_dim, device=device)

    def _module_name_for_fused_suffix(self, target: HookTarget, suffix: str) -> str:
        return f"{target.module_name.rsplit('.', 1)[0]}.{suffix}"

    def _candidate_delta(self, target: HookTarget, candidate: PerturbationSpec, flat_x: torch.Tensor, output_dim: int) -> torch.Tensor:
        if target.suffix == "qkv_proj":
            return self._candidate_delta_fused_qkv(target, candidate, flat_x, output_dim)
        factors = self.factors(
            target,
            candidate,
            input_dim=int(flat_x.shape[-1]),
            output_dim=output_dim,
            device=flat_x.device,
        )
        z = flat_x.to(dtype=self.adapter_dtype) @ factors.a.T
        return z @ factors.b.T

    def _candidate_delta_fused_qkv(
        self,
        target: HookTarget,
        candidate: PerturbationSpec,
        flat_x: torch.Tensor,
        output_dim: int,
    ) -> torch.Tensor:
        if self.fused_qkv is None:
            raise RuntimeError("qkv_proj LoRA hook requires fused_qkv dimensions")
        if output_dim != self.fused_qkv.total_out:
            raise RuntimeError(f"qkv_proj output dim {output_dim} != expected {self.fused_qkv.total_out}")
        requested = set(candidate.targets)
        if not requested:
            requested = {"q_proj", "k_proj", "v_proj"}
        delta = torch.zeros((int(flat_x.shape[0]), output_dim), device=flat_x.device, dtype=self.adapter_dtype)
        for suffix in ("q_proj", "k_proj", "v_proj"):
            if suffix not in requested:
                continue
            output_slice = self.fused_qkv.slice_for(suffix)
            module_name = self._module_name_for_fused_suffix(target, suffix)
            factors = self.factors_for_module(
                module_name,
                candidate,
                input_dim=int(flat_x.shape[-1]),
                output_dim=int(output_slice.stop - output_slice.start),
                device=flat_x.device,
            )
            z = flat_x.to(dtype=self.adapter_dtype) @ factors.a.T
            delta[:, output_slice] = z @ factors.b.T
        return delta

    def delta(self, target: HookTarget, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor | None:
        candidate = self.active_candidate
        row_candidate_indices = self._row_candidate_indices_cpu
        row_candidates = self.active_candidates
        if candidate is None and (row_candidate_indices is None or not row_candidates):
            return None
        flat_x = x.reshape(-1, x.shape[-1])
        output_dim = int(y.shape[-1])
        if candidate is None and (row_candidate_indices is None or self._row_candidate_indices_len != int(flat_x.shape[0])):
            raise RuntimeError(
                "vLLM lazy LoRA hook candidate-batch routing row count mismatch: "
                f"expected {self._row_candidate_indices_len}, got {int(flat_x.shape[0])}"
            )
        if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
            torch.cuda.synchronize(flat_x.device)
        start = time.perf_counter()
        if candidate is not None:
            delta = self._candidate_delta(target, candidate, flat_x, output_dim).reshape(y.shape)
        else:
            delta_flat = torch.zeros((int(flat_x.shape[0]), output_dim), device=flat_x.device, dtype=self.adapter_dtype)
            for candidate_index, row_start, row_end in self._row_candidate_spans:
                if row_end <= row_start:
                    continue
                delta_flat[row_start:row_end] = self._candidate_delta(
                    target,
                    row_candidates[candidate_index],
                    flat_x[row_start:row_end],
                    output_dim,
                )
            delta = delta_flat.reshape(y.shape)
        if self.sync_timing and torch.cuda.is_available() and flat_x.is_cuda:
            torch.cuda.synchronize(flat_x.device)
        elapsed = time.perf_counter() - start
        self.qx_time_s += elapsed
        self.delta_time_s += elapsed
        self.delta_rows += int(flat_x.shape[0])
        self.delta_calls += 1
        return delta.to(dtype=y.dtype)


__all__ = ["FusedQKVSpec", "LazyLoraHookRuntime", "LoraHookFactors"]
