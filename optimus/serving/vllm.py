from __future__ import annotations

from typing import Any

from optimus.core.candidates import candidate_panel
from optimus.modeling import AdapterSpec
from optimus.serving.runtime import import_vllm_lora_request, make_sampling_params


def build_adapter_specs(*args: Any, **kwargs: Any) -> Any:
    from optimus.serving.search import make_adapter_specs

    return make_adapter_specs(*args, **kwargs)


def run_vllm_search(argv: list[str] | None = None) -> Any:
    from optimus.serving.search import main

    return main(argv)

__all__ = [
    "AdapterSpec",
    "build_adapter_specs",
    "candidate_panel",
    "import_vllm_lora_request",
    "make_sampling_params",
    "run_vllm_search",
]
