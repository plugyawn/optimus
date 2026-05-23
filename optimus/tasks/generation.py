from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from fractions import Fraction
from pathlib import Path


@dataclass(frozen=True)
class GeneratedCountdown:
    id: int
    numbers: list[int]
    target: int
    solution: str


def combine_expr(left: tuple[Fraction, str], right: tuple[Fraction, str], op: str) -> tuple[Fraction, str] | None:
    a, a_expr = left
    b, b_expr = right
    if op == "+":
        return a + b, f"({a_expr}+{b_expr})"
    if op == "-":
        return a - b, f"({a_expr}-{b_expr})"
    if op == "*":
        return a * b, f"({a_expr}*{b_expr})"
    if op == "/":
        if b == 0:
            return None
        return a / b, f"({a_expr}/{b_expr})"
    raise ValueError(op)


def random_solution(numbers: list[int], rng: random.Random) -> tuple[Fraction, str] | None:
    terms = [(Fraction(n), str(n)) for n in rng.sample(numbers, len(numbers))]
    while len(terms) > 1:
        idx = rng.randrange(len(terms) - 1)
        left = terms.pop(idx)
        right = terms.pop(idx)
        ops = ["+", "-", "*", "/"]
        rng.shuffle(ops)
        combined = None
        for op in ops:
            candidate = combine_expr(left, right, op)
            if candidate is not None and candidate[0].denominator == 1:
                combined = candidate
                break
        if combined is None:
            return None
        terms.insert(idx, combined)
    return terms[0]


def generate_examples(
    *,
    count: int,
    seed: int,
    numbers_per_example: int = 4,
    min_number: int = 1,
    max_number: int = 100,
    min_target: int = 1,
    max_target: int = 999,
    max_attempts: int = 1_000_000,
) -> list[GeneratedCountdown]:
    rng = random.Random(seed)
    examples: list[GeneratedCountdown] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    attempts = 0
    while len(examples) < count and attempts < max_attempts:
        attempts += 1
        numbers = [rng.randint(min_number, max_number) for _ in range(numbers_per_example)]
        solution = random_solution(numbers, rng)
        if solution is None:
            continue
        value, expr = solution
        if value.denominator != 1:
            continue
        target = int(value)
        if target < min_target or target > max_target:
            continue
        key = (tuple(sorted(numbers)), target)
        if key in seen:
            continue
        seen.add(key)
        examples.append(GeneratedCountdown(len(examples), numbers, target, expr))
    if len(examples) < count:
        raise RuntimeError(f"Generated {len(examples)} examples after {attempts} attempts; requested {count}.")
    return examples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate a deterministic solvable Countdown JSON dataset.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=1200)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--numbers-per-example", type=int, default=4)
    parser.add_argument("--min-number", type=int, default=1)
    parser.add_argument("--max-number", type=int, default=100)
    parser.add_argument("--min-target", type=int, default=1)
    parser.add_argument("--max-target", type=int, default=999)
    args = parser.parse_args(argv)
    examples = generate_examples(
        count=args.count,
        seed=args.seed,
        numbers_per_example=args.numbers_per_example,
        min_number=args.min_number,
        max_number=args.max_number,
        min_target=args.min_target,
        max_target=args.max_target,
    )
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps([asdict(ex) for ex in examples], indent=2, sort_keys=True) + "\n")
    print(json.dumps({"out": str(out), "count": len(examples), "seed": args.seed}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
