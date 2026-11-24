"""What holds the node half's loop, so a slow node says which work made it slow.

Every view of a node half runs on one loop, which each of ParaView's calls blocks.
Background work notes what it did (``span``), and the parts of it (``phase``);
renders note how long they drew (``rendered``, or ``drawing`` around a render the
view's frame clock cannot time). ``watch_held`` wakes a few times a second; when the
loop came back late it writes one line naming what held it, and what no span named
as ``untraced``.
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
    """The work the loop did since the last look, and how long each part took.

    Each second is counted once: a span inside a span, and a render inside a span,
    are parts of the outer span, and a render timed by both :meth:`drawing` and the
    view's frame clock counts as one.
    """

    def __init__(self) -> None:
        self.spans: list[tuple[str, float, dict[str, float]]] = []
        self.render = 0.0
        self._open: dict[str, float] | None = None
        self._drawing = 0

    def _part(self, name: str, seconds: float) -> None:
        if self._open is not None:
            self._open[name] = self._open.get(name, 0.0) + seconds

    @contextmanager
    def span(self, label: str) -> Iterator[None]:
        outer, self._open = self._open, {}
        began = time.perf_counter()
        try:
            yield
        finally:
            parts, self._open = self._open or {}, outer
            took = time.perf_counter() - began
            if outer is not None:
                self._part(label, took)
            else:
                self.spans.append((label, took, parts))

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        began = time.perf_counter()
        try:
            yield
        finally:
            self._part(name, time.perf_counter() - began)

    @contextmanager
    def drawing(self) -> Iterator[None]:
        """Time a render the frame clock cannot see, such as a pvserver's."""
        began = time.perf_counter()
        self._drawing += 1
        try:
            yield
        finally:
            self._drawing -= 1
            if not self._drawing:
                self.rendered(time.perf_counter() - began)

    def rendered(self, seconds: float) -> None:
        if self._drawing:
            return  # the enclosing drawing() counts it
        if self._open is not None:
            self._part("render", seconds)
        else:
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
    if render >= 0.05:
        named.append(f"renders {render:.1f} s")
    # Work no span names, such as a browser's image request or ParaView's own
    # callbacks. The look spans the tick too, so what is named can pass ``held``.
    untraced = held - sum(s[1] for s in spans) - render
    if untraced >= 0.5:
        named.append(f"untraced {untraced:.1f} s")
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
