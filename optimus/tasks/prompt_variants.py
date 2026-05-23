from __future__ import annotations

from collections.abc import Callable

from .countdown import CountdownExample, prompt as default_prompt


PromptFn = Callable[[CountdownExample], str]

UPSTREAM_SYSTEM_MESSAGE = (
    "You are a helpful assistant. You first think about the reasoning process "
    "in your mind and then provide the user with the answer."
)


def upstream_reasoning_prompt(example: CountdownExample) -> str:
    nums = list(example.numbers)
    return (
        f"Using the numbers {nums}, create an equation that equals {example.target}. "
        "You can use basic arithmetic operations (+, -, *, /) and each number can only be used once. "
        "Show your work in <think> </think> tags. "
        "And return the final answer in <answer> </answer> tags, for example <answer> (1 + 2) / 3 </answer>."
    )


def tight_tagged_prompt(example: CountdownExample) -> str:
    nums = ", ".join(str(x) for x in example.numbers)
    return (
        "Output exactly this format: <answer>EXPRESSION</answer>. "
        "Use each given number exactly once. No reasoning. No other text. "
        f"Numbers: {nums}. Target: {example.target}."
    )


def compact_tagged_prompt(example: CountdownExample) -> str:
    nums = ", ".join(str(x) for x in example.numbers)
    return (
        f"Numbers: {nums}. Target: {example.target}. "
        "Reply only with <answer>one arithmetic expression</answer>. "
        "Use each number exactly once."
    )


def direct_tagged_prompt(example: CountdownExample) -> str:
    nums = ", ".join(str(x) for x in example.numbers)
    return (
        f"Make {example.target} from these numbers: {nums}. "
        "Use every number once. Put only the expression inside <answer></answer>."
    )


def reordered_tagged_prompt(example: CountdownExample) -> str:
    nums = ", ".join(str(x) for x in example.numbers)
    return (
        f"Numbers: {nums}. Target: {example.target}. "
        "Use the given numbers exactly once. "
        "Return only one arithmetic expression wrapped in <answer> </answer> tags. "
        "Do not include an equals sign, reasoning, or any other text."
    )


def xml_tagged_prompt(example: CountdownExample) -> str:
    nums = ", ".join(str(x) for x in example.numbers)
    return (
        "Write exactly one arithmetic expression and nothing else. "
        "Put it between <answer> and </answer>. "
        "The expression must use each provided number exactly once, must not contain an equals sign, "
        "must not include reasoning or any other text, "
        f"and must evaluate to the target. Provided numbers: {nums}. Target: {example.target}."
    )


def prompt_fn(name: str) -> PromptFn:
    if name == "default":
        return default_prompt
    if name == "upstream":
        return upstream_reasoning_prompt
    if name == "reordered":
        return reordered_tagged_prompt
    if name == "xml":
        return xml_tagged_prompt
    if name == "compact":
        return compact_tagged_prompt
    if name == "direct":
        return direct_tagged_prompt
    if name == "tight":
        return tight_tagged_prompt
    raise ValueError(f"unknown prompt variant: {name}")


def system_message_for_variant(name: str) -> str | None:
    if name == "upstream":
        return UPSTREAM_SYSTEM_MESSAGE
    return None


def render_prompt_text(
    user_content: str,
    *,
    variant: str,
    tokenizer=None,
    use_chat_template: bool = False,
) -> str:
    system_content = system_message_for_variant(variant)
    messages = []
    if system_content:
        messages.append({"role": "system", "content": system_content})
    messages.append({"role": "user", "content": user_content})
    if use_chat_template:
        if tokenizer is None:
            raise ValueError("use_chat_template=True requires a tokenizer")
        if getattr(tokenizer, "chat_template", None):
            return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    return "\n".join(message["content"] for message in messages) + ("\n" if system_content else "")


def make_variant_prompts(
    examples: list[CountdownExample],
    variant: str,
    *,
    tokenizer=None,
    use_chat_template: bool = False,
) -> list[str]:
    make_prompt = prompt_fn(variant)
    return [
        render_prompt_text(
            make_prompt(ex),
            variant=variant,
            tokenizer=tokenizer,
            use_chat_template=use_chat_template,
        )
        for ex in examples
    ]

__all__ = [
    "PromptFn",
    "UPSTREAM_SYSTEM_MESSAGE",
    "compact_tagged_prompt",
    "direct_tagged_prompt",
    "make_variant_prompts",
    "prompt_fn",
    "render_prompt_text",
    "reordered_tagged_prompt",
    "system_message_for_variant",
    "tight_tagged_prompt",
    "upstream_reasoning_prompt",
    "xml_tagged_prompt",
]
