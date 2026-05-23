from __future__ import annotations

import sys
from collections.abc import Sequence

from ._help import AGGREGATE_LORA_HELP


def main(argv: Sequence[str] | None = None) -> None:
    args = list(sys.argv[1:] if argv is None else argv)
    if any(arg in {"-h", "--help"} for arg in args):
        print(AGGREGATE_LORA_HELP.rstrip())
        return
    from optimus.search.aggregation import main as aggregation_main

    aggregation_main(args)


if __name__ == "__main__":
    main()
