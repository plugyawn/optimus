from __future__ import annotations

from dataclasses import dataclass

import torch

from .lora_space import Candidate, canonical_module_name, stable_int


def dense_noise_tensor(
    module_name: str,
    shape: torch.Size | tuple[int, ...],
    candidate: Candidate,
    *,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Materialize a canonical CPU dense Gaussian perturbation tensor."""

    canonical_name = canonical_module_name(module_name)
    gen = torch.Generator(device="cpu")
    gen.manual_seed((candidate.seed + stable_int(canonical_name + ":dense_gaussian")) % (2**63 - 1))
    noise = torch.randn(tuple(shape), generator=gen, dtype=dtype)
    return (candidate.sign * candidate.sigma) * noise


@dataclass
class DensePatchEntry:
    name: str
    module: torch.nn.Module
    base_weight: torch.Tensor


class DenseGaussianPatcher:
    """Apply dense Gaussian candidates to selected linear weights and restore exactly."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_suffixes: tuple[str, ...],
        *,
        snapshot_device: str = "model",
    ):
        if snapshot_device not in {"model", "cpu"}:
            raise ValueError("snapshot_device must be 'model' or 'cpu'")
        self.entries: list[DensePatchEntry] = []
        for name, module in model.named_modules():
            weight = getattr(module, "weight", None)
            if weight is None or getattr(weight, "ndim", 0) != 2:
                continue
            if not any(name.endswith(suffix) for suffix in target_suffixes):
                continue
            base = weight.detach().clone()
            if snapshot_device == "cpu":
                base = base.cpu()
            self.entries.append(DensePatchEntry(name=name, module=module, base_weight=base))
        if not self.entries:
            raise ValueError(f"no dense target modules matched suffixes {target_suffixes}")

    @property
    def module_names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    @torch.no_grad()
    def clear(self) -> None:
        for entry in self.entries:
            weight = entry.module.weight
            weight.copy_(entry.base_weight.to(device=weight.device, dtype=weight.dtype))

    @torch.no_grad()
    def set_candidate(self, candidate: Candidate) -> None:
        for entry in self.entries:
            weight = entry.module.weight
            weight.copy_(entry.base_weight.to(device=weight.device, dtype=weight.dtype))
            noise = dense_noise_tensor(entry.name, weight.shape, candidate)
            weight.add_(noise.to(device=weight.device, dtype=weight.dtype))
