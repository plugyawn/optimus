from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Any


ANSWER_STOP_TEXT = "</answer>"


def _jsonable(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return str(value)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_ids(ids: Sequence[int]) -> str:
    payload = ",".join(str(int(token_id)) for token_id in ids)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def tokenizer_input_ids(tokenizer, text: str, *, add_special_tokens: bool = True) -> list[int]:
    encoded = tokenizer(text, add_special_tokens=add_special_tokens)
    return [int(token_id) for token_id in encoded["input_ids"]]


def tokenizer_contract(tokenizer, prompts: Sequence[str], *, answer_stop_text: str = ANSWER_STOP_TEXT) -> dict:
    stop_ids = tokenizer_input_ids(tokenizer, answer_stop_text, add_special_tokens=False)
    prompt_rows = []
    for idx, prompt in enumerate(prompts):
        ids = tokenizer_input_ids(tokenizer, prompt, add_special_tokens=True)
        prompt_rows.append(
            {
                "index": idx,
                "text_sha256": sha256_text(prompt),
                "char_length": len(prompt),
                "token_ids": ids,
                "token_count": len(ids),
                "token_ids_sha256": sha256_ids(ids),
            }
        )
    token_lengths = [row["token_count"] for row in prompt_rows]
    return {
        "tokenizer_class": type(tokenizer).__name__,
        "padding_side": getattr(tokenizer, "padding_side", None),
        "pad_token": getattr(tokenizer, "pad_token", None),
        "pad_token_id": getattr(tokenizer, "pad_token_id", None),
        "eos_token": getattr(tokenizer, "eos_token", None),
        "eos_token_id": getattr(tokenizer, "eos_token_id", None),
        "bos_token": getattr(tokenizer, "bos_token", None),
        "bos_token_id": getattr(tokenizer, "bos_token_id", None),
        "unk_token": getattr(tokenizer, "unk_token", None),
        "unk_token_id": getattr(tokenizer, "unk_token_id", None),
        "chat_template_sha256": sha256_text(getattr(tokenizer, "chat_template", "") or ""),
        "answer_stop_text": answer_stop_text,
        "answer_stop_ids": stop_ids,
        "answer_stop_decoded": tokenizer.decode(stop_ids) if hasattr(tokenizer, "decode") else None,
        "add_special_tokens_for_prompts": True,
        "prompt_count": len(prompt_rows),
        "min_prompt_tokens": min(token_lengths) if token_lengths else 0,
        "max_prompt_tokens": max(token_lengths) if token_lengths else 0,
        "prompt_token_ids_sha256": sha256_ids([token_id for row in prompt_rows for token_id in row["token_ids"]]),
        "prompts": prompt_rows,
    }


def resolve_vllm_tokenizer(llm):
    if hasattr(llm, "get_tokenizer"):
        try:
            tokenizer = llm.get_tokenizer()
            if tokenizer is not None:
                return tokenizer
        except Exception:
            pass
    for attr in ["tokenizer", "_tokenizer"]:
        tokenizer = getattr(llm, attr, None)
        if tokenizer is not None:
            return tokenizer
    engine = getattr(llm, "llm_engine", None)
    for attr in ["tokenizer", "_tokenizer"]:
        tokenizer = getattr(engine, attr, None)
        if tokenizer is not None:
            return tokenizer
    return None


def vllm_tokenizer_contract(llm, prompts: Sequence[str], *, answer_stop_text: str = ANSWER_STOP_TEXT) -> dict:
    tokenizer = resolve_vllm_tokenizer(llm)
    if tokenizer is None:
        return {"available": False, "reason": "no tokenizer attribute or get_tokenizer result found"}
    try:
        contract = tokenizer_contract(tokenizer, prompts, answer_stop_text=answer_stop_text)
    except Exception as exc:
        return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}
    contract["available"] = True
    return contract


def sampling_kwargs(max_new_tokens: int, stop_at_answer: bool, *, answer_stop_text: str = ANSWER_STOP_TEXT) -> dict:
    kwargs: dict[str, Any] = {"max_tokens": int(max_new_tokens), "temperature": 0.0}
    if stop_at_answer:
        kwargs["stop"] = [answer_stop_text]
        kwargs["include_stop_str_in_output"] = True
    return kwargs


def sampling_params_contract(SamplingParams, max_new_tokens: int, stop_at_answer: bool, *, answer_stop_text: str = ANSWER_STOP_TEXT) -> dict:
    requested = sampling_kwargs(max_new_tokens, stop_at_answer, answer_stop_text=answer_stop_text)
    used = dict(requested)
    dropped_include_stop = False
    try:
        params = SamplingParams(**used)
    except TypeError:
        used.pop("include_stop_str_in_output", None)
        dropped_include_stop = True
        params = SamplingParams(**used)

    attrs = {}
    for name in [
        "max_tokens",
        "temperature",
        "top_p",
        "top_k",
        "stop",
        "stop_token_ids",
        "include_stop_str_in_output",
        "ignore_eos",
        "skip_special_tokens",
        "spaces_between_special_tokens",
    ]:
        if hasattr(params, name):
            attrs[name] = _jsonable(getattr(params, name))

    return {
        "sampling_params_class": f"{type(params).__module__}.{type(params).__name__}",
        "requested_kwargs": _jsonable(requested),
        "used_kwargs": _jsonable(used),
        "dropped_include_stop_str_in_output": dropped_include_stop,
        "actual_attrs": attrs,
    }


def contract_max_tokens(args) -> int:
    value = getattr(args, "max_new_tokens", None)
    if value is None:
        return 1
    return max(int(value), 1)


def backend_contract(tokenizer, prompts: Sequence[str], args, SamplingParams=None) -> dict:
    max_tokens = contract_max_tokens(args)
    contract = {
        "kind": "backend_prompt_decode_contract",
        "model": getattr(args, "model", None),
        "max_new_tokens": max_tokens,
        "stop_at_answer": bool(getattr(args, "stop_at_answer", False)),
        "hf_batch_size": getattr(args, "hf_batch_size", None),
        "hf_dtype": getattr(args, "hf_dtype", None),
        "vllm_dtype": getattr(args, "vllm_dtype", None),
        "tokenizer": tokenizer_contract(tokenizer, prompts),
    }
    if SamplingParams is not None:
        contract["vllm_sampling"] = sampling_params_contract(
            SamplingParams,
            max_tokens,
            bool(getattr(args, "stop_at_answer", False)),
        )
    else:
        contract["vllm_sampling"] = {
            "requested_kwargs": _jsonable(
                sampling_kwargs(
                    max_tokens,
                    bool(getattr(args, "stop_at_answer", False)),
                )
            )
        }
    return contract
