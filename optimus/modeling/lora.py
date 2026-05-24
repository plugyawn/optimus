from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from optimus.core.perturbations import PerturbationSpec as Candidate
from optimus.modeling.qwen import SUPPORTED_QWEN_LORA_TARGETS, qwen_lora_shapes


@dataclass(frozen=True)
class AdapterSpec:
    index: int
    name: str
    lora_int_id: int
    path: str
    candidate: str
    seed: int
    sigma: float
    sign: int
    method: str = "lora"


def parse_targets(text: str) -> list[str]:
    targets = [x.strip() for x in text.split(",") if x.strip()]
    if not targets:
        raise ValueError("--targets must contain at least one module suffix")
    unknown = sorted(set(targets) - SUPPORTED_QWEN_LORA_TARGETS)
    if unknown:
        raise ValueError(
            "Direct adapter generation currently supports Qwen2-style targets "
            f"{sorted(SUPPORTED_QWEN_LORA_TARGETS)}, got unsupported targets {unknown}."
        )
    return targets


def adapter_config(model: str, rank: int, targets: list[str]) -> dict:
    return {
        "base_model_name_or_path": model,
        "bias": "none",
        "fan_in_fan_out": False,
        "inference_mode": True,
        "init_lora_weights": True,
        "lora_alpha": rank,
        "lora_dropout": 0.0,
        "peft_type": "LORA",
        "r": rank,
        "target_modules": targets,
        "task_type": "CAUSAL_LM",
    }


def write_adapter_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def lora_noise_tensors(*args, **kwargs):
    from optimus.modeling.noise import lora_noise_tensors as materialize

    return materialize(*args, **kwargs)


def save_seed_adapter(
    adapter_dir: Path,
    *,
    model: str,
    candidate: Candidate,
    rank: int,
    targets: list[str],
    config,
    tensor_dtype: str,
    family_state: dict | None = None,
) -> None:
    if candidate.method != "lora":
        raise ValueError(f"LoRA adapter materialization requires lora perturbations, got {candidate.method!r}")
    import torch
    from safetensors.torch import save_file

    dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[tensor_dtype]
    tensors = {}
    for module, in_features, out_features in qwen_lora_shapes(config, targets):
        a, b = lora_noise_tensors(
            module,
            (rank, in_features),
            (out_features, rank),
            candidate,
            rank,
            family_state=family_state,
            state_key=module,
        )
        prefix = f"base_model.model.{module}"
        tensors[f"{prefix}.lora_A.weight"] = a.to(dtype).contiguous()
        tensors[f"{prefix}.lora_B.weight"] = b.to(dtype).contiguous()

    adapter_dir.mkdir(parents=True, exist_ok=True)
    write_adapter_json(adapter_dir / "adapter_config.json", adapter_config(model, rank, targets))
    save_file(tensors, str(adapter_dir / "adapter_model.safetensors"), metadata={"format": "pt"})

__all__ = [
    "AdapterSpec",
    "Candidate",
    "adapter_config",
    "lora_noise_tensors",
    "parse_targets",
    "save_seed_adapter",
]
