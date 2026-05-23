from __future__ import annotations

from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> None:
    from optimus.evaluation.backend_parity import main as backend_parity_main

    raise SystemExit(backend_parity_main(list(argv) if argv is not None else None))


if __name__ == "__main__":
    main()
