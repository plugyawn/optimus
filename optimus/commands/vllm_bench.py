from __future__ import annotations

from collections.abc import Sequence

from ._help import VLLM_BENCH_HELP
from ._dispatch import run_module


def main(argv: Sequence[str] | None = None) -> None:
    run_module("optimus.serving.benchmark", argv, help_text=VLLM_BENCH_HELP)


if __name__ == "__main__":
    main()
