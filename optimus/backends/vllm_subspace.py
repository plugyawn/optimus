from __future__ import annotations

from typing import Any

from optimus.subspace.reference import run_reference_search


def run_eager_smoke(args: Any) -> dict:
    """Run the vLLM-labelled subspace smoke path.

    V1 keeps vLLM as the selected production backend and enforces the cache and
    candidate-routing contract before optimized hooks land. The current smoke
    path uses the same torch lazy evaluator as the reference route, but emits
    vLLM backend artifacts and fails closed on shared prefix caching.
    """

    if getattr(args, "enable_prefix_caching", None):
        raise ValueError("subspace vLLM smoke requires shared prefix caching to stay disabled")
    if (getattr(args, "prefix_cache_policy", None) or "disabled-for-search") != "disabled-for-search":
        raise ValueError("subspace vLLM smoke requires --prefix-cache-policy disabled-for-search")
    return run_reference_search(args, backend="vllm")


__all__ = ["run_eager_smoke"]
