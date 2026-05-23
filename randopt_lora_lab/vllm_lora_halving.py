"""Compatibility entrypoint for Optimus staged vLLM LoRA search."""

from optimus.serving.halving import (
    build_parser,
    diagnostic_payload,
    main,
    reset_outputs,
    run_halving,
)

__all__ = [
    "build_parser",
    "diagnostic_payload",
    "main",
    "reset_outputs",
    "run_halving",
]


if __name__ == "__main__":
    raise SystemExit(main())
