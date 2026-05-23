from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from optimus.tasks.generation import main as generation_main

    raise SystemExit(generation_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main()
