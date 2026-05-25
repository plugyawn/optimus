"""Search-space construction, selection, and zeroth-order study utilities."""

from optimus.core.perturbations import PerturbationSpec, perturbation_panel, read_perturbation_file
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
from .zeroth_order import EvaluationRecord, SearchResult, ZerothOrderStudy, select_top_k, sorted_records

__all__ = [
    "EvaluationRecord",
    "PerturbationSpec",
    "SearchResult",
    "ZerothOrderStudy",
    "anzo_anchor_prompts",
    "base_protocol_valid",
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
    "perturbation_panel",
    "read_perturbation_file",
    "rows_by_candidate_and_example",
    "protocol_valid_variants",
    "run_adaptive_search",
    "select_top_k",
    "sorted_records",
]


def run_adaptive_search(*args, **kwargs):
    from .adaptive import run_search

    return run_search(*args, **kwargs)
