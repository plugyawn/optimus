from __future__ import annotations

from collections.abc import Sequence

from .contracts import tokenizer_input_ids


def make_vllm_prompt_inputs(prompt_texts: Sequence[str], tokenizer, prompt_input: str):
    if prompt_input == "text":
        return list(prompt_texts)
    if prompt_input != "token_ids":
        raise ValueError(f"unknown vLLM prompt input mode: {prompt_input!r}")

    prompt_token_ids = [
        tokenizer_input_ids(tokenizer, text, add_special_tokens=True)
        for text in prompt_texts
    ]
    try:
        from vllm.inputs import TokensPrompt

        return [TokensPrompt(prompt_token_ids=list(ids)) for ids in prompt_token_ids]
    except Exception:
        return [list(ids) for ids in prompt_token_ids]
