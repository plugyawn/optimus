from __future__ import annotations

import sys
from typing import Any


_HELP = """\
usage: python -m randopt_lora_lab.adaptive {search} ...

Compatibility entry point for Optimus adaptive LoRA search.

Prefer importing from `optimus.search.adaptive`.
"""


__all__ = [
    "build_family_state",
    "candidate_a_noise",
    "candidate_score_rows",
    "col_scale_from_rows",
    "expand_prior_paths",
    "lora_a_modules",
    "main",
    "read_jsonl",
    "run_search",
    "top_basis",
]


def __getattr__(name: str) -> Any:
    if name in __all__:
        from optimus.search import adaptive

        return getattr(adaptive, name)
    raise AttributeError(name)


def main(argv: list[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(_HELP.rstrip())
        return
    from optimus.search.adaptive import main as adaptive_main

    adaptive_main(args)


if __name__ == "__main__":
    main()
