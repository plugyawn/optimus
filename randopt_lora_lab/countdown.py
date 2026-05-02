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


def load_examples(path: str | None, n: int, seed: int) -> list[CountdownExample]:
    if path and Path(path).exists():
        data = json.loads(Path(path).read_text())
        examples = []
        for i, row in enumerate(data):
            nums = tuple(int(x) for x in row["numbers"])
            examples.append(CountdownExample(int(row.get("id", i)), nums, int(row["target"])))
    else:
        examples = built_in_examples()
    rng = random.Random(seed)
    examples = examples[:]
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


def score_completion(text: str, example: CountdownExample) -> dict:
    answer = extract_answer(text)
    has_answer = bool(answer)
    malformed = False
    exact = 0.0
    if answer:
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
        "answer": answer,
    }


def prompts(examples: Iterable[CountdownExample]) -> list[str]:
    return [prompt(ex) for ex in examples]
