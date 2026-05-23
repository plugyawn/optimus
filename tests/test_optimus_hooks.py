from __future__ import annotations

from optimus.core.hooks import HookRegistry


def test_hook_registry_emits_structured_events():
    seen = []
    registry = HookRegistry([seen.append])

    event = registry.emit("candidate_screened", "screen", candidate="c0", exact=0.125)

    assert len(registry) == 1
    assert seen == [event]
    assert event.name == "candidate_screened"
    assert event.step == "screen"
    assert event.payload == {"candidate": "c0", "exact": 0.125}
