from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from optimus.evaluation.materialize import main as materialize_main

    raise SystemExit(materialize_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main()
