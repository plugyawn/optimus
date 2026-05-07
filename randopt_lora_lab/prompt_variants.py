from __future__ import annotations

from collections.abc import Callable

from .countdown import CountdownExample, prompt as default_prompt


PromptFn = Callable[[CountdownExample], str]


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


def make_variant_prompts(examples: list[CountdownExample], variant: str) -> list[str]:
    make_prompt = prompt_fn(variant)
    return [make_prompt(ex) for ex in examples]
