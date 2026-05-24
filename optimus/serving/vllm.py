from __future__ import annotations

from typing import Any

from optimus.core.perturbations import PerturbationSpec, perturbation_panel
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
    "PerturbationSpec",
    "build_adapter_specs",
    "import_vllm_lora_request",
    "make_sampling_params",
    "perturbation_panel",
    "run_vllm_search",
]
