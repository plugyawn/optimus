"""Serving backends used for high-throughput candidate evaluation."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "AdapterSpec",
    "GenerationResult",
    "TransformersDenseGaussianBackend",
    "TransformersLoraBackend",
    "backend_contract",
    "build_adapter_specs",
    "candidate_panel",
    "import_vllm_lora_request",
    "make_vllm_prompt_inputs",
    "make_sampling_params",
    "sampling_params_contract",
    "score_mixed_rows",
    "score_rows",
    "timed_generate",
    "tokenizer_contract",
    "run_benchmark",
    "run_halving",
    "run_search",
    "run_vllm_search",
    "vllm_tokenizer_contract",
    "visible_token_count",
]

_EXPORT_MODULES = {
    "backend_contract": "optimus.serving.contracts",
    "make_vllm_prompt_inputs": "optimus.serving.prompting",
    "sampling_params_contract": "optimus.serving.contracts",
    "score_mixed_rows": "optimus.serving.runtime",
    "score_rows": "optimus.serving.runtime",
    "timed_generate": "optimus.serving.runtime",
    "tokenizer_contract": "optimus.serving.contracts",
    "GenerationResult": "optimus.serving.transformers",
    "TransformersDenseGaussianBackend": "optimus.serving.transformers",
    "TransformersLoraBackend": "optimus.serving.transformers",
    "visible_token_count": "optimus.serving.transformers",
    "run_benchmark": "optimus.serving.benchmark",
    "run_halving": "optimus.serving.halving",
    "run_search": "optimus.serving.search",
    "vllm_tokenizer_contract": "optimus.serving.contracts",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module(_EXPORT_MODULES.get(name, "optimus.serving.vllm"))
    return getattr(module, name)
