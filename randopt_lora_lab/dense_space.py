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
    noise_mode: str = "canonical",
) -> torch.Tensor:
    """Materialize a canonical CPU dense Gaussian perturbation tensor."""

    canonical_name = canonical_module_name(module_name)
    gen = torch.Generator(device="cpu")
    if noise_mode == "canonical":
        gen.manual_seed((candidate.seed + stable_int(canonical_name + ":dense_gaussian")) % (2**63 - 1))
    elif noise_mode == "paper":
        gen.manual_seed(int(candidate.seed))
    else:
        raise ValueError(f"unknown dense noise mode: {noise_mode}")
    noise = torch.randn(tuple(shape), generator=gen, dtype=dtype)
    return (candidate.sign * candidate.sigma) * noise


@dataclass
class DensePatchEntry:
    name: str
    parameter: torch.nn.Parameter
    base_weight: torch.Tensor


class DenseGaussianPatcher:
    """Apply dense Gaussian candidates to selected weights and restore exactly."""

    def __init__(
        self,
        model: torch.nn.Module,
        target_suffixes: tuple[str, ...],
        *,
        snapshot_device: str = "model",
        noise_mode: str = "canonical",
    ):
        if snapshot_device not in {"model", "cpu"}:
            raise ValueError("snapshot_device must be 'model' or 'cpu'")
        if noise_mode not in {"canonical", "paper"}:
            raise ValueError("noise_mode must be 'canonical' or 'paper'")
        self.noise_mode = noise_mode
        self.entries: list[DensePatchEntry] = []
        if any(suffix in {"all", "all_params", "*"} for suffix in target_suffixes):
            for name, parameter in model.named_parameters():
                if not parameter.is_floating_point():
                    continue
                base = parameter.detach().clone()
                if snapshot_device == "cpu":
                    base = base.cpu()
                self.entries.append(DensePatchEntry(name=name, parameter=parameter, base_weight=base))
        else:
            for name, module in model.named_modules():
                weight = getattr(module, "weight", None)
                if weight is None or getattr(weight, "ndim", 0) != 2:
                    continue
                if not any(name.endswith(suffix) for suffix in target_suffixes):
                    continue
                base = weight.detach().clone()
                if snapshot_device == "cpu":
                    base = base.cpu()
                self.entries.append(DensePatchEntry(name=name, parameter=weight, base_weight=base))
        if not self.entries:
            raise ValueError(f"no dense target modules matched suffixes {target_suffixes}")

    @property
    def module_names(self) -> list[str]:
        return [entry.name for entry in self.entries]

    @torch.no_grad()
    def clear(self) -> None:
        for entry in self.entries:
            weight = entry.parameter
            weight.copy_(entry.base_weight.to(device=weight.device, dtype=weight.dtype))

    @torch.no_grad()
    def set_candidate(self, candidate: Candidate) -> None:
        for entry in self.entries:
            weight = entry.parameter
            weight.copy_(entry.base_weight.to(device=weight.device, dtype=weight.dtype))
            noise = dense_noise_tensor(entry.name, weight.shape, candidate, noise_mode=self.noise_mode)
            weight.add_(noise.to(device=weight.device, dtype=weight.dtype))
