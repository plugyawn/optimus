from __future__ import annotations

import ast
import json
import math
import operator as op
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


@dataclass(frozen=True)
class CountdownExample:
    id: int
    numbers: tuple[int, ...]
    target: int


def built_in_examples() -> list[CountdownExample]:
    raw = [
        ([89, 80, 70], 99),
        ([55, 53, 37, 40], 79),
        ([4, 20, 8], 48),
        ([3, 7, 8, 8], 24),
        ([25, 50, 75, 100], 24),
        ([2, 5, 9, 10], 37),
        ([6, 6, 5, 2], 17),
        ([9, 4, 4, 2], 18),
        ([1, 3, 4, 6], 24),
        ([7, 7, 3, 3], 24),
        ([10, 10, 4, 4], 96),
        ([8, 6, 3, 1], 25),
        ([12, 5, 2, 2], 26),
        ([11, 7, 6, 1], 30),
        ([13, 9, 4, 2], 47),
        ([15, 8, 3, 2], 50),
        ([21, 7, 5, 2], 17),
        ([100, 25, 5, 2], 52),
        ([14, 12, 6, 3], 30),
        ([18, 9, 4, 2], 34),
        ([16, 8, 4, 2], 30),
        ([19, 13, 5, 2], 45),
        ([17, 11, 6, 4], 40),
        ([22, 10, 7, 3], 75),
        ([24, 12, 6, 2], 54),
        ([30, 15, 5, 3], 48),
        ([40, 20, 10, 5], 35),
        ([50, 30, 10, 2], 70),
        ([60, 20, 5, 3], 45),
        ([70, 10, 7, 2], 68),
        ([80, 40, 8, 2], 50),
        ([90, 30, 9, 3], 93),
    ]
    return [CountdownExample(i, tuple(nums), target) for i, (nums, target) in enumerate(raw)]


def assert_unique_example_ids(examples: Iterable[CountdownExample], *, label: str = "examples") -> None:
    seen: set[int] = set()
    duplicates: list[int] = []
    for ex in examples:
        if ex.id in seen:
            duplicates.append(ex.id)
        seen.add(ex.id)
    if duplicates:
        sample = ", ".join(str(x) for x in duplicates[:8])
        raise ValueError(f"{label} contains duplicate example ids: {sample}")


def semantic_example_key(example: CountdownExample) -> tuple[tuple[int, ...], int]:
    return tuple(sorted(example.numbers)), example.target


def assert_unique_example_semantics(examples: Iterable[CountdownExample], *, label: str = "examples") -> None:
    seen: set[tuple[tuple[int, ...], int]] = set()
    duplicates: list[tuple[tuple[int, ...], int]] = []
    for ex in examples:
        key = semantic_example_key(ex)
        if key in seen:
            duplicates.append(key)
        seen.add(key)
    if duplicates:
        sample = ", ".join(f"{numbers}->{target}" for numbers, target in duplicates[:8])
        raise ValueError(f"{label} contains duplicate semantic Countdown examples: {sample}")


def unique_example_count(examples: Iterable[CountdownExample]) -> int:
    return len({ex.id for ex in examples})


def unique_semantic_example_count(examples: Iterable[CountdownExample]) -> int:
    return len({semantic_example_key(ex) for ex in examples})


def load_examples(
    path: str | None,
    n: int,
    seed: int,
    *,
    allow_repeat: bool = False,
    exclude_ids: set[int] | None = None,
) -> list[CountdownExample]:
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
        examples = []
        for i, row in enumerate(data):
            nums = tuple(int(x) for x in row["numbers"])
            examples.append(CountdownExample(int(row.get("id", i)), nums, int(row["target"])))
    else:
        examples = built_in_examples()
    assert_unique_example_ids(examples, label=path or "built-in Countdown examples")
    assert_unique_example_semantics(examples, label=path or "built-in Countdown examples")
    if exclude_ids:
        examples = [ex for ex in examples if ex.id not in exclude_ids]
    rng = random.Random(seed)
    examples = examples[:]
    if len(examples) < n:
        if not allow_repeat:
            raise ValueError(
                f"Requested {n} examples, but only {len(examples)} unique examples are available. "
                "Pass --allow-repeat-data only for smoke tests."
            )
        while len(examples) < n:
            examples.extend(examples)
    rng.shuffle(examples)
    return examples[:n]


def prompt(example: CountdownExample) -> str:
    nums = ", ".join(str(x) for x in example.numbers)
    return (
        "Return only one arithmetic expression using the given numbers exactly once, "
        "wrapped in <answer> </answer> tags. Do not include an equals sign, reasoning, "
        f"or any other text. Numbers: {nums}. Target: {example.target}."
    )


def extract_answer(text: str) -> str:
    matches = ANSWER_RE.findall(text)
    return matches[-1].strip() if matches else ""


def answer_spans(text: str) -> list[re.Match[str]]:
    return list(ANSWER_RE.finditer(text))


ALLOWED = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.USub: op.neg,
    ast.UAdd: op.pos,
}


def safe_eval_expr(expr: str) -> float:
    def rec(node):
        if isinstance(node, ast.Expression):
            return rec(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.UnaryOp) and type(node.op) in ALLOWED:
            return ALLOWED[type(node.op)](rec(node.operand))
        if isinstance(node, ast.BinOp) and type(node.op) in ALLOWED:
            return ALLOWED[type(node.op)](rec(node.left), rec(node.right))
        raise ValueError("bad expression")

    return float(rec(ast.parse(expr, mode="eval")))


def score_completion(text: str, example: CountdownExample, *, strict: bool = True) -> dict:
    spans = answer_spans(text)
    answer = spans[-1].group(1).strip() if spans else ""
    has_answer = bool(answer)
    answer_count = len(spans)
    missing_answer = answer_count == 0
    multiple_answers = answer_count > 1
    trailing_text = False
    if spans:
        before = text[: spans[0].start()].strip()
        after = text[spans[-1].end() :].strip()
        trailing_text = bool(before or after)
    malformed = bool(strict and (missing_answer or multiple_answers or trailing_text))
    exact = 0.0
    if answer and not malformed:
        if "=" in answer or not re.match(r"^[0-9+\-*/(). ]+$", answer):
            malformed = True
        else:
            used = sorted(int(x) for x in re.findall(r"\d+", answer))
            try:
                value = safe_eval_expr(answer)
                exact = float(used == sorted(example.numbers) and math.isclose(value, example.target, abs_tol=1e-6))
            except Exception:
                malformed = True
    return {
        "exact": exact,
        "has_answer": has_answer,
        "malformed": malformed,
        "answer_count": answer_count,
        "missing_answer": missing_answer,
        "multiple_answers": multiple_answers,
        "trailing_text": trailing_text,
        "answer": answer,
    }


def extract_numeric_vote(text: str, example: CountdownExample) -> dict:
    """Return the numeric Countdown vote for a valid formula completion.

    This mirrors the paper-style Countdown ensemble rule: candidates vote by
    evaluated numeric result, but only formulas using exactly the given numbers
    are allowed into the vote.
    """
    answer = extract_answer(text)
    if not answer:
        return {"valid_vote": False, "vote": "", "vote_reject": "missing_answer"}
    if "=" in answer or not re.match(r"^[0-9+\-*/(). ]+$", answer):
        return {"valid_vote": False, "vote": "", "vote_reject": "invalid_chars"}
    used = sorted(int(x) for x in re.findall(r"\d+", answer))
    if used != sorted(example.numbers):
        return {
            "valid_vote": False,
            "vote": "",
            "vote_reject": "wrong_numbers",
            "used_numbers": used,
        }
    try:
        value = safe_eval_expr(answer)
    except Exception:
        return {"valid_vote": False, "vote": "", "vote_reject": "eval_error"}
    if math.isclose(value, round(value), abs_tol=1e-9):
        vote = str(int(round(value)))
    else:
        vote = str(value)
    return {"valid_vote": True, "vote": vote, "vote_reject": ""}


def voted_answer_exact(vote: str, example: CountdownExample) -> float:
    if not vote:
        return 0.0
    try:
        return float(math.isclose(float(vote), float(example.target), abs_tol=1e-5))
    except ValueError:
        return 0.0


def prompts(examples: Iterable[CountdownExample]) -> list[str]:
    return [prompt(ex) for ex in examples]
