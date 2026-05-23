from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True)
class OptimusEvent:
    """Structured event emitted by search, serving, and reporting workflows."""

    name: str
    step: str
    payload: dict[str, Any] = field(default_factory=dict)


class OptimusHook(Protocol):
    def __call__(self, event: OptimusEvent) -> None: ...


class HookRegistry:
    """Minimal synchronous hook registry for research instrumentation."""

    def __init__(self, hooks: Iterable[OptimusHook] | None = None):
        self._hooks: list[OptimusHook] = list(hooks or [])

    def register(self, hook: OptimusHook) -> None:
        self._hooks.append(hook)

    def emit(self, name: str, step: str, **payload: Any) -> OptimusEvent:
        event = OptimusEvent(name=name, step=step, payload=dict(payload))
        for hook in list(self._hooks):
            hook(event)
        return event

    def __len__(self) -> int:
        return len(self._hooks)


def hook_registry(hooks: Iterable[OptimusHook | Callable[[OptimusEvent], None]] | None = None) -> HookRegistry:
    return HookRegistry(hooks)
