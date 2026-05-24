"""Search-space construction and candidate-panel utilities."""

from optimus.core.candidates import SearchCandidate, candidate_panel, parse_candidate_key, read_candidate_file
from .ensemble import (
    anzo_anchor_prompts,
    ensemble_ks_from_values,
    majority_vote_evaluation,
    parse_float_list,
    parse_k_list,
    parse_ratio_list,
    rows_by_candidate_and_example,
)
from .selection import (
    base_protocol_valid,
    combine_candidate_conditions,
    condition_score,
    enrich_condition_rows,
    filter_condition_rows_by_variants,
    parse_prompt_variants,
    protocol_valid_variants,
)

__all__ = [
    "SearchCandidate",
    "anzo_anchor_prompts",
    "base_protocol_valid",
    "candidate_panel",
    "combine_candidate_conditions",
    "condition_score",
    "ensemble_ks_from_values",
    "enrich_condition_rows",
    "filter_condition_rows_by_variants",
    "majority_vote_evaluation",
    "parse_float_list",
    "parse_k_list",
    "parse_ratio_list",
    "parse_prompt_variants",
    "parse_candidate_key",
    "read_candidate_file",
    "rows_by_candidate_and_example",
    "protocol_valid_variants",
    "run_adaptive_search",
    "run_peft_search",
]


def run_adaptive_search(*args, **kwargs):
    from .adaptive import run_search

    return run_search(*args, **kwargs)


def run_peft_search(*args, **kwargs):
    from .peft import run_search

    return run_search(*args, **kwargs)
