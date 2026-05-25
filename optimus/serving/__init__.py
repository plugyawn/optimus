"""Serving backends used for high-throughput candidate evaluation."""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "backend_contract",
    "make_vllm_prompt_inputs",
    "sampling_params_contract",
    "score_mixed_rows",
    "score_rows",
    "timed_generate",
    "tokenizer_contract",
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
    "visible_token_count": "optimus.serving.transformers",
    "vllm_tokenizer_contract": "optimus.serving.contracts",
}


def __getattr__(name: str) -> Any:
    if name not in __all__:
        raise AttributeError(name)
    module = importlib.import_module(_EXPORT_MODULES[name])
    return getattr(module, name)
