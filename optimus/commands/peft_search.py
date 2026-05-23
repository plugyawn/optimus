from __future__ import annotations

import sys
from collections.abc import Sequence

from ._help import PEFT_SEARCH_HELP


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(PEFT_SEARCH_HELP.rstrip())
        return
    from optimus.search.peft import main as peft_main

    peft_main(["search", *args])


if __name__ == "__main__":
    main()
