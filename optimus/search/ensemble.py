from __future__ import annotations

from collections import Counter

from optimus.tasks.countdown import (
    extract_numeric_vote,
    score_completion,
    voted_answer_exact,
)


def parse_float_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def parse_k_list(text: str) -> list[int]:
    return sorted({int(x) for x in text.split(",") if x.strip()})


def parse_ratio_list(text: str) -> list[float]:
    return [float(x) for x in text.split(",") if x.strip()]


def ensemble_ks_from_values(population: int, k_text: str = "", ratio_text: str = "") -> list[int]:
    ks = set(parse_k_list(k_text) if k_text else [])
    for ratio in parse_ratio_list(ratio_text) if ratio_text else []:
        ks.add(max(1, int(float(population) * ratio)))
    return sorted(ks)


def rows_by_candidate_and_example(rows: list[dict]) -> dict[str, dict[int, dict]]:
    out: dict[str, dict[int, dict]] = {}
    for row in rows:
        out.setdefault(str(row["candidate"]), {})[int(row["example_id"])] = row
    return out


def _mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def majority_vote_evaluation(
    candidate_order: list[str],
    rows: list[dict],
    examples,
    k_values: list[int],
    *,
    strict_rows: bool = False,
) -> tuple[list[dict], list[dict]]:
    by_candidate = rows_by_candidate_and_example(rows)
    result_rows = []
    per_prompt_rows = []
    for k in k_values:
        active = candidate_order[: min(k, len(candidate_order))]
        exact_values = []
        coverage_values = []
        valid_vote_counts = []
        for ex in examples:
            votes = []
            rejects = Counter()
            for candidate in active:
                row = by_candidate.get(candidate, {}).get(ex.id)
                if not row:
                    continue
                if strict_rows and score_completion(str(row.get("text", "")), ex, strict=True)["malformed"]:
                    rejects["strict_malformed"] += 1
                    continue
                vote = extract_numeric_vote(str(row.get("text", "")), ex)
                if vote["valid_vote"]:
                    votes.append(str(vote["vote"]))
                else:
                    rejects[str(vote["vote_reject"])] += 1
            counter = Counter(votes)
            final_vote = counter.most_common(1)[0][0] if counter else ""
            exact = voted_answer_exact(final_vote, ex)
            exact_values.append(exact)
            coverage_values.append(float(bool(counter)))
            valid_vote_counts.append(len(votes))
            per_prompt_rows.append(
                {
                    "k": k,
                    "example_id": ex.id,
                    "numbers": list(ex.numbers),
                    "target": ex.target,
                    "final_vote": final_vote,
                    "exact": exact,
                    "valid_vote_count": len(votes),
                    "missing_vote_count": max(len(active) - len(votes), 0),
                    "strict_rows": strict_rows,
                    "vote_counts": dict(counter),
                    "reject_counts": dict(rejects),
                }
            )
        denom = max(len(examples), 1)
        result_rows.append(
            {
                "k": k,
                "evaluated_candidates": len(active),
                "exact_mean": _mean([float(value) for value in exact_values]),
                "coverage_mean": _mean(coverage_values),
                "valid_votes_per_prompt": _mean([float(value) for value in valid_vote_counts]),
                "correct": int(sum(exact_values)),
                "total": denom,
                "strict_rows": strict_rows,
            }
        )
    return result_rows, per_prompt_rows


def anzo_anchor_prompts() -> list[str]:
    return [
        "Explain why the sky looks blue in one sentence.",
        "Write a short Python function that reverses a list.",
        "Summarize the benefits of unit tests.",
        "Draft a polite email declining a meeting.",
        "Explain what photosynthesis does.",
        "Give concise debugging advice for an import error.",
        "Compare quicksort and mergesort briefly.",
        "Extract names and dates from a short paragraph.",
    ]


__all__ = [
    "anzo_anchor_prompts",
    "ensemble_ks_from_values",
    "majority_vote_evaluation",
    "parse_float_list",
    "parse_k_list",
    "parse_ratio_list",
    "rows_by_candidate_and_example",
]
