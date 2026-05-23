from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from optimus.evaluation.systems import main as systems_main

    raise SystemExit(systems_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main()
