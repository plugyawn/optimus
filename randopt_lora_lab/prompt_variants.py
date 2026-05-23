"""Compatibility re-exports for Countdown prompt variants."""

from optimus.tasks.countdown import CountdownExample
from optimus.tasks.prompt_variants import (
    UPSTREAM_SYSTEM_MESSAGE,
    PromptFn,
    compact_tagged_prompt,
    direct_tagged_prompt,
    make_variant_prompts as _make_variant_prompts,
    prompt_fn as _prompt_fn,
    render_prompt_text,
    reordered_tagged_prompt,
    system_message_for_variant,
    tight_tagged_prompt,
    upstream_reasoning_prompt,
    xml_tagged_prompt,
)

PAPER_SYSTEM_MESSAGE = UPSTREAM_SYSTEM_MESSAGE
paper_reasoning_prompt = upstream_reasoning_prompt


def _normalize_legacy_prompt_variant(name: str) -> str:
    return "upstream" if name == "paper" else name


def prompt_fn(name: str) -> PromptFn:
    return _prompt_fn(_normalize_legacy_prompt_variant(name))


def make_variant_prompts(
    examples: list[CountdownExample],
    variant: str,
    *,
    tokenizer=None,
    use_chat_template: bool = False,
) -> list[str]:
    return _make_variant_prompts(
        examples,
        _normalize_legacy_prompt_variant(variant),
        tokenizer=tokenizer,
        use_chat_template=use_chat_template,
    )


__all__ = [
    "PAPER_SYSTEM_MESSAGE",
    "PromptFn",
    "UPSTREAM_SYSTEM_MESSAGE",
    "compact_tagged_prompt",
    "direct_tagged_prompt",
    "make_variant_prompts",
    "paper_reasoning_prompt",
    "prompt_fn",
    "render_prompt_text",
    "reordered_tagged_prompt",
    "system_message_for_variant",
    "tight_tagged_prompt",
    "upstream_reasoning_prompt",
    "xml_tagged_prompt",
]
