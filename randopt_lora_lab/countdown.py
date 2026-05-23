"""Compatibility re-exports for Countdown task utilities."""

from optimus.tasks.countdown import (
    CountdownExample,
    answer_spans,
    assert_unique_example_ids,
    assert_unique_example_semantics,
    built_in_examples,
    extract_answer,
    extract_numeric_vote,
    load_examples,
    prompt,
    prompts,
    safe_eval_expr,
    score_completion,
    semantic_example_key,
    unique_example_count,
    unique_semantic_example_count,
    voted_answer_exact,
)

__all__ = [
    "CountdownExample",
    "answer_spans",
    "assert_unique_example_ids",
    "assert_unique_example_semantics",
    "built_in_examples",
    "extract_answer",
    "extract_numeric_vote",
    "load_examples",
    "prompt",
    "prompts",
    "safe_eval_expr",
    "score_completion",
    "semantic_example_key",
    "unique_example_count",
    "unique_semantic_example_count",
    "voted_answer_exact",
]
