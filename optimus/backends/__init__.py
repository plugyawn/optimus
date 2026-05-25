"""Backend integration contracts for Optimus search runtimes.

vLLM is the planned production execution substrate for subspace search, but the
runtime route fails closed until the Phase 5 backend lands. Transformers paths
are reference and parity infrastructure.
"""

from __future__ import annotations

from typing import Literal


BackendName = Literal["vllm", "transformers"]
MethodName = Literal["dense", "lora", "subspace"]

__all__ = ["BackendName", "MethodName"]
