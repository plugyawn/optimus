from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from optimus.evaluation.lighteval_report import main as report_main

    raise SystemExit(report_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main(sys.argv[1:])
