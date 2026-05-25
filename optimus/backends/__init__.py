"""Backend integration contracts for Optimus search runtimes.

vLLM is the production execution substrate for subspace search. Transformers
paths are reference and parity infrastructure.
"""

from __future__ import annotations

from typing import Literal


BackendName = Literal["vllm", "transformers"]
MethodName = Literal["dense", "lora", "subspace"]

__all__ = ["BackendName", "MethodName"]
