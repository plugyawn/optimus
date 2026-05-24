from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from optimus.evaluation.lighteval import main as lighteval_main

    raise SystemExit(lighteval_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main()
