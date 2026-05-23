"""Model-specific adapter materialization helpers."""

from .lora import AdapterSpec, adapter_config, parse_targets, save_seed_adapter
from .qwen import SUPPORTED_QWEN_LORA_TARGETS, load_qwen_lora_config, normalize_qwen_text_config, qwen_lora_shapes

__all__ = [
    "AdapterSpec",
    "DenseGaussianPatcher",
    "DensePatchEntry",
    "MatrixSpec",
    "SUPPORTED_QWEN_LORA_TARGETS",
    "adapter_config",
    "best_rank_projection",
    "dense_noise_tensor",
    "lora_update",
    "load_qwen_lora_config",
    "low_rank_factors_from_dense",
    "normalize_dense_noise_mode",
    "normalize_qwen_text_config",
    "parse_targets",
    "qwen_lora_shapes",
    "randomized_low_rank_factors_from_dense",
    "save_seed_adapter",
    "spectral_projected_gaussian_factors",
]


def __getattr__(name: str):
    if name in {"DenseGaussianPatcher", "DensePatchEntry", "dense_noise_tensor", "normalize_dense_noise_mode"}:
        from . import dense

        return getattr(dense, name)
    if name in {
        "MatrixSpec",
        "best_rank_projection",
        "lora_update",
        "low_rank_factors_from_dense",
        "randomized_low_rank_factors_from_dense",
        "spectral_projected_gaussian_factors",
    }:
        from . import geometry

        return getattr(geometry, name)
    raise AttributeError(name)
