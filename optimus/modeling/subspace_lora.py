from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import torch

from optimus.modeling.lora import AdapterSpec, adapter_config, write_adapter_json
from optimus.modeling.qwen import qwen_lora_shapes
from optimus.subspace import SubspaceCandidate, sign_value


_LAYER_RE = re.compile(r"\.layers\.(\d+)\.")


def _dtype(name: str) -> torch.dtype:
    return {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }[name.strip().lower()]


def subspace_candidate_from_record(row: dict[str, Any]) -> SubspaceCandidate:
    return SubspaceCandidate(
        candidate_id=str(row["candidate_id"]),
        direction_seed=int(row["direction_seed"]),
        sign=str(row["sign"]),  # type: ignore[arg-type]
        basis_hash=str(row["basis_hash"]),
        target_set_hash=str(row["target_set_hash"]),
        scale_mode=str(row["scale_mode"]),  # type: ignore[arg-type]
        rho_or_sigma_w=float(row["rho_or_sigma_w"]),
        budget_policy=str(row["budget_policy"]),  # type: ignore[arg-type]
        budget_hash=str(row["budget_hash"]),
        runtime_dtype=str(row.get("runtime_dtype", "bf16")),
        radius_index=int(row.get("radius_index", 0)),
        target_preset=str(row.get("target_preset", "")),
        basis_rank=int(row["basis_rank"]),
        shard_id=str(row.get("shard_id", "single")),
        shard_population_start=int(row.get("shard_population_start", 0)),
        shard_population_end=int(row.get("shard_population_end", 0)),
        worker_id=str(row.get("worker_id", "adapter-bridge")),
        device_id=str(row.get("device_id", "cpu")),
        prompt_scoring_config_hash=str(row.get("prompt_scoring_config_hash", "")),
        rng_version=str(row.get("rng_version", "gaussian_hash_v1")),
    )


def load_subspace_candidates(path: Path, wanted: set[str] | None = None) -> list[SubspaceCandidate]:
    candidates: list[SubspaceCandidate] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        candidate = subspace_candidate_from_record(json.loads(line))
        if wanted is not None and candidate.candidate_id not in wanted:
            continue
        candidates.append(candidate)
    if not candidates:
        raise ValueError(f"no subspace candidates selected from {path}")
    return candidates


def _layer_index(module_name: str) -> int:
    match = _LAYER_RE.search(module_name)
    if not match:
        raise ValueError(f"cannot infer layer index from module name {module_name!r}")
    return int(match.group(1))


def _suffix(module_name: str) -> str:
    return module_name.rsplit(".", 1)[-1]


def site_id_for_lora_module(module_name: str) -> str:
    layer = _layer_index(module_name)
    suffix = _suffix(module_name)
    if suffix in {"q_proj", "k_proj", "v_proj"}:
        return f"layer_{layer}.attn_in"
    if suffix == "o_proj":
        return f"layer_{layer}.o_in"
    if suffix in {"gate_proj", "up_proj"}:
        return f"layer_{layer}.mlp_in"
    if suffix == "down_proj":
        return f"layer_{layer}.down_in"
    raise ValueError(f"unsupported LoRA module suffix {suffix!r}")


def split_target_id_for_lora_module(module_name: str) -> str:
    layer = _layer_index(module_name)
    suffix = _suffix(module_name)
    if suffix in {"q_proj", "k_proj", "v_proj", "o_proj"}:
        return f"layer_{layer}.self_attn.{suffix}"
    if suffix in {"gate_proj", "up_proj", "down_proj"}:
        return f"layer_{layer}.mlp.{suffix}"
    raise ValueError(f"unsupported LoRA module suffix {suffix!r}")


def fused_qkv_target_id_for_lora_module(module_name: str) -> str | None:
    suffix = _suffix(module_name)
    if suffix not in {"q_proj", "k_proj", "v_proj"}:
        return None
    return f"layer_{_layer_index(module_name)}.self_attn.qkv_proj"


def _qkv_slice(suffix: str, *, q_out: int, kv_out: int) -> slice:
    if suffix == "q_proj":
        return slice(0, q_out)
    if suffix == "k_proj":
        return slice(q_out, q_out + kv_out)
    if suffix == "v_proj":
        return slice(q_out + kv_out, q_out + 2 * kv_out)
    raise ValueError(f"not a fused qkv suffix: {suffix!r}")


def _torch_generator_field(
    *,
    direction_seed: int,
    sign: str,
    target_id: str,
    output_dim: int,
    basis_rank: int,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    seed_payload = f"{int(direction_seed)}\0{target_id}\0torch_generator_field_v1".encode("utf-8")
    seed = int(hashlib.sha256(seed_payload).hexdigest()[:16], 16) % (2**63 - 1)
    gen = torch.Generator(device="cpu").manual_seed(seed)
    field = torch.randn((int(output_dim), int(basis_rank)), generator=gen, dtype=torch.float32)
    if sign_value(sign) < 0:
        field = -field
    return field.to(dtype=dtype)


def _scale_by_target(summary: dict[str, Any], radius: float) -> dict[str, float]:
    key = f"{radius:g}"
    out: dict[str, float] = {}
    for row in summary.get("resolved_target_scales") or []:
        target_id = str(row["target_id"])
        by_radius = row.get("beta_t_by_radius") or {}
        if key not in by_radius:
            raise ValueError(f"summary has no beta for radius {key!r} on {target_id}")
        out[target_id] = float(by_radius[key])
    if not out:
        raise ValueError("source summary is missing resolved_target_scales")
    return out


def _basis_by_site(state_payload: dict[str, Any], state_summary: dict[str, Any]) -> dict[str, torch.Tensor]:
    tensors = state_payload.get("basis_tensors") or {}
    out: dict[str, torch.Tensor] = {}
    for site in state_summary.get("activation_sites") or []:
        site_id = str(site["site_id"])
        tensor_key = str(site["basis_tensor_key"])
        if tensor_key not in tensors:
            raise ValueError(f"subspace_state.pt is missing basis tensor {tensor_key!r}")
        out[site_id] = tensors[tensor_key].detach().cpu().float().contiguous()
    if not out:
        raise ValueError("subspace state has no activation-site bases")
    return out


def _qkv_dims(config: Any) -> tuple[int, int]:
    hidden = int(config.hidden_size)
    heads = int(config.num_attention_heads)
    kv_heads = int(getattr(config, "num_key_value_heads", heads))
    head_dim = int(getattr(config, "head_dim", hidden // heads))
    return heads * head_dim, kv_heads * head_dim


def subspace_lora_tensors(
    *,
    config: Any,
    state_payload: dict[str, Any],
    state_summary: dict[str, Any],
    source_summary: dict[str, Any],
    candidate: SubspaceCandidate,
    targets: Iterable[str],
    policy: str,
    tensor_dtype: str,
    adapter_rank: int | None = None,
    scale_multiplier: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Return PEFT LoRA tensors for one activation-subspace candidate.

    ``fused-qkv-exact`` reproduces the current vLLM hook contract when vLLM
    exposes a fused ``qkv_proj`` target by generating one field for
    ``layer_i.self_attn.qkv_proj`` and slicing it into q/k/v adapter tensors.
    ``target-split`` uses per-module target ids, which is the intended q/v
    target policy but is not replay-equivalent to older fused-qkv hook runs.
    """

    if policy not in {"fused-qkv-exact", "target-split"}:
        raise ValueError("--adapter-policy must be fused-qkv-exact or target-split")
    dtype = _dtype(tensor_dtype)
    basis = _basis_by_site(state_payload, state_summary)
    beta_by_target = _scale_by_target(source_summary, candidate.rho_or_sigma_w)
    q_out, kv_out = _qkv_dims(config)
    target_list = list(targets)
    tensors: dict[str, torch.Tensor] = {}

    for module_name, in_features, out_features in qwen_lora_shapes(config, target_list):
        suffix = _suffix(module_name)
        site_id = site_id_for_lora_module(module_name)
        q = basis[site_id]
        source_rank = int(q.shape[0])
        if adapter_rank is not None:
            if adapter_rank <= 0 or adapter_rank > source_rank:
                raise ValueError(f"adapter_rank must be in [1, {source_rank}], got {adapter_rank}")
            q = q[: int(adapter_rank)].contiguous()
        if int(q.shape[1]) != int(in_features):
            raise ValueError(f"basis width mismatch for {module_name}: basis={tuple(q.shape)} in_features={in_features}")

        if policy == "fused-qkv-exact" and suffix in {"q_proj", "k_proj", "v_proj"}:
            fused_target = fused_qkv_target_id_for_lora_module(module_name)
            assert fused_target is not None
            total = q_out + 2 * kv_out
            field = _torch_generator_field(
                direction_seed=candidate.direction_seed,
                sign=candidate.sign,
                target_id=fused_target,
                output_dim=total,
                basis_rank=source_rank,
            )[_qkv_slice(suffix, q_out=q_out, kv_out=kv_out), : int(q.shape[0])]
            beta = beta_by_target[fused_target]
        else:
            target_id = split_target_id_for_lora_module(module_name)
            field = _torch_generator_field(
                direction_seed=candidate.direction_seed,
                sign=candidate.sign,
                target_id=target_id,
                output_dim=int(out_features),
                basis_rank=source_rank,
            )[:, : int(q.shape[0])]
            beta = beta_by_target.get(target_id)
            if beta is None:
                fused_target = fused_qkv_target_id_for_lora_module(module_name)
                if fused_target is None or fused_target not in beta_by_target:
                    raise ValueError(f"no beta available for {target_id}")
                beta = beta_by_target[fused_target]

        prefix = f"base_model.model.{module_name}"
        tensors[f"{prefix}.lora_A.weight"] = q.to(dtype=dtype).contiguous()
        tensors[f"{prefix}.lora_B.weight"] = (float(scale_multiplier) * float(beta) * field).to(dtype=dtype).contiguous()
    return tensors


def save_subspace_adapter(
    adapter_dir: Path,
    *,
    model: str,
    config: Any,
    state_payload: dict[str, Any],
    state_summary: dict[str, Any],
    source_summary: dict[str, Any],
    candidate: SubspaceCandidate,
    targets: list[str],
    policy: str,
    tensor_dtype: str,
    adapter_rank: int | None = None,
    scale_multiplier: float = 1.0,
) -> None:
    from safetensors.torch import save_file

    tensors = subspace_lora_tensors(
        config=config,
        state_payload=state_payload,
        state_summary=state_summary,
        source_summary=source_summary,
        candidate=candidate,
        targets=targets,
        policy=policy,
        tensor_dtype=tensor_dtype,
        adapter_rank=adapter_rank,
        scale_multiplier=scale_multiplier,
    )
    rank = int(adapter_rank or candidate.basis_rank)
    adapter_dir.mkdir(parents=True, exist_ok=True)
    write_adapter_json(adapter_dir / "adapter_config.json", adapter_config(model, rank, targets))
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})


def write_subspace_adapter_specs(
    *,
    out: Path,
    model: str,
    config: Any,
    state_payload: dict[str, Any],
    state_summary: dict[str, Any],
    source_summary: dict[str, Any],
    candidates: list[SubspaceCandidate],
    targets: list[str],
    policy: str,
    tensor_dtype: str,
    adapter_rank: int | None = None,
    scale_multiplier: float = 1.0,
) -> list[AdapterSpec]:
    adapter_root = out / "adapters"
    adapter_root.mkdir(parents=True, exist_ok=True)
    specs: list[AdapterSpec] = []
    for idx, candidate in enumerate(candidates):
        name = f"subspace_seed{candidate.direction_seed}_{candidate.sign.replace('+', 'pos').replace('-', 'neg')}_rho{candidate.rho_or_sigma_w:g}"
        path = adapter_root / f"{idx:05d}_{name}"
        save_subspace_adapter(
            path,
            model=model,
            config=config,
            state_payload=state_payload,
            state_summary=state_summary,
            source_summary=source_summary,
            candidate=candidate,
            targets=targets,
            policy=policy,
            tensor_dtype=tensor_dtype,
            adapter_rank=adapter_rank,
            scale_multiplier=scale_multiplier,
        )
        specs.append(
            AdapterSpec(
                index=idx,
                name=name,
                lora_int_id=idx + 1,
                path=str(path.resolve()),
                candidate=candidate.candidate_id,
                seed=int(candidate.direction_seed),
                sigma=float(candidate.rho_or_sigma_w),
                sign=int(sign_value(candidate.sign)),
                method="subspace-as-lora",
            )
        )
    return specs


def specs_to_jsonl(specs: list[AdapterSpec]) -> str:
    return "".join(json.dumps(asdict(spec), sort_keys=True) + "\n" for spec in specs)


__all__ = [
    "load_subspace_candidates",
    "save_subspace_adapter",
    "specs_to_jsonl",
    "subspace_lora_tensors",
    "write_subspace_adapter_specs",
]
