"""Compatibility re-exports for vLLM/HF prompt and sampling contracts."""

from optimus.serving.contracts import (
    ANSWER_STOP_TEXT,
    backend_contract,
    contract_max_tokens,
    resolve_vllm_tokenizer,
    sampling_kwargs,
    sampling_params_contract,
    sha256_ids,
    sha256_text,
    tokenizer_contract,
    tokenizer_input_ids,
    vllm_tokenizer_contract,
)

__all__ = [
    "ANSWER_STOP_TEXT",
    "backend_contract",
    "contract_max_tokens",
    "resolve_vllm_tokenizer",
    "sampling_kwargs",
    "sampling_params_contract",
    "sha256_ids",
    "sha256_text",
    "tokenizer_contract",
    "tokenizer_input_ids",
    "vllm_tokenizer_contract",
]
