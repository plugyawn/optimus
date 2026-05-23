"""Compatibility entrypoint for Optimus vLLM LoRA search."""

from optimus.core.candidates import candidate_panel, read_candidate_file
from optimus.serving.search import (
    base_eval,
    build_activation_family_state,
    build_parser,
    diagnostic_payload,
    make_adapter_specs,
    main,
    mixed_eval,
    require_all_prompt_variants_valid_or_raise,
    reset_outputs,
    run_search,
    safe_name,
    selection_variants_or_raise,
    write_prompt_contracts,
)

__all__ = [
    "base_eval",
    "build_activation_family_state",
    "build_parser",
    "candidate_panel",
    "diagnostic_payload",
    "make_adapter_specs",
    "main",
    "mixed_eval",
    "read_candidate_file",
    "require_all_prompt_variants_valid_or_raise",
    "reset_outputs",
    "run_search",
    "safe_name",
    "selection_variants_or_raise",
    "write_prompt_contracts",
]


if __name__ == "__main__":
    raise SystemExit(main())
