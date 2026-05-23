from __future__ import annotations

import runpy
import sys
from collections.abc import Sequence


def run_module(module: str, argv: Sequence[str] | None = None, *, help_text: str | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if help_text and any(arg in {"-h", "--help"} for arg in args):
        print(help_text.rstrip())
        return
    prog = sys.argv[0]
    old_argv = sys.argv[:]
    try:
        sys.argv = [prog, *args]
        runpy.run_module(module, run_name="__main__", alter_sys=False)
    finally:
        sys.argv = old_argv
