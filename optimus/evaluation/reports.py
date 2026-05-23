from __future__ import annotations

from typing import Any


def build_systems_report(*args: Any, **kwargs: Any) -> Any:
    from .systems import main

    return main(*args, **kwargs)

__all__ = ["build_systems_report"]
