"""What holds the node half's loop, so a slow node says which work made it slow.

Every view of a node half runs on one loop, which each of ParaView's calls blocks.
Background work notes what it did (``span``), and the parts of it (``phase``);
renders note how long they drew (``rendered``). ``watch_held`` wakes a few times a
second; when the loop came back late it writes one line naming what held it.
Nothing is stopped or limited: it only says.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Iterator
from contextlib import contextmanager

#: Seconds the loop may be held before the log says what held it.
HELD_REPORT = 2.0
#: Seconds between looks at the loop.
HELD_TICK = 0.25


class Ledger:
    """The work the loop did since the last look, and how long each part took."""

    def __init__(self) -> None:
        self.spans: list[tuple[str, float, dict[str, float]]] = []
        self.render = 0.0
        self._open: dict[str, float] | None = None

    @contextmanager
    def span(self, label: str) -> Iterator[None]:
        outer, self._open = self._open, {}
        began = time.perf_counter()
        try:
            yield
        finally:
            parts, self._open = self._open or {}, outer
            self.spans.append((label, time.perf_counter() - began, parts))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        began = time.perf_counter()
        try:
            yield
        finally:
            if self._open is not None:
                took = time.perf_counter() - began
                self._open[name] = self._open.get(name, 0.0) + took

    def rendered(self, seconds: float) -> None:
        self.render += seconds

    def take(self) -> tuple[list[tuple[str, float, dict[str, float]]], float]:
        spans, render = self.spans, self.render
        self.spans, self.render = [], 0.0
        return spans, render


LEDGER = Ledger()


def report(
    held: float, spans: list[tuple[str, float, dict[str, float]]], render: float
) -> str:
    """One line: how long the loop was held, and by what."""
    named = []
    for label, took, parts in sorted(spans, key=lambda s: -s[1]):
        inside = ", ".join(
            f"{name} {seconds:.1f}"
            for name, seconds in sorted(parts.items(), key=lambda p: -p[1])
            if seconds >= 0.05
        )
        named.append(f"{label} {took:.1f} s" + (f" ({inside})" if inside else ""))
    other = held - sum(s[1] for s in spans)
    if render >= 0.05:
        named.append(f"renders {render:.1f} s")
    if other >= 0.5:
        named.append(f"other {other:.1f} s")
    return f"the node's loop was held {held:.1f} s: " + ("; ".join(named) or "unknown")


async def watch_held(tick: float = HELD_TICK, limit: float = HELD_REPORT) -> None:
    last = time.monotonic()
    while True:
        await asyncio.sleep(tick)
        now = time.monotonic()
        held = now - last - tick
        last = now
        spans, render = LEDGER.take()
        if held >= limit:
            print(report(held, spans, render), flush=True)
