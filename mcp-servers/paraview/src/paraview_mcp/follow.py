"""Watching a growing source step by step, for the ``watch_steps`` tool.

The node half re-reads a growing source's time list; ``steps`` returns the
steps from an index on, each with when it was first seen. ``watch`` polls it
and reports every new step once, through ``report``: the tool sends that as
MCP progress, which a client running the call as a task reads as the task's
status, and a client calling it directly receives as progress notifications.
It stops once the descriptor's ``expected_steps`` are written, when no step
has come for ``idle_seconds``, after ``max_seconds``, or when the source is no
longer marked as growing.
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable
from typing import Any

from .present import _num

Fetch = Callable[[int], Awaitable[dict[str, Any]]]
Report = Callable[[int, int | None, str], Awaitable[None]]

POLL_SECONDS = 1.0
#: The steps kept in the watch's result; the progress messages carried each one.
KEEP = 200


def view_note(time_state: dict[str, Any]) -> str:
    """Where the view is, against the newest step."""
    steps = int(time_state.get("steps") or 0)
    if steps == 0:
        return "the data has no time steps yet"
    if time_state.get("catching_up"):
        return "the view is catching up from the first step, then follows it"
    if time_state.get("follow_latest"):
        return "the view follows it"
    return (
        f"the view stays parked at step {int(time_state.get('index', 0)) + 1}, "
        f"{time_state.get('newer_steps', 0)} newer step(s) exist"
    )


def _of(expected: int | None) -> str:
    return f" of {expected}" if expected else ""


def expected_steps(result: dict[str, Any]) -> int | None:
    """The steps the producer will write in total, from its view descriptor."""
    value = (result.get("follow") or {}).get("expected_steps")
    return int(value) if value else None


def step_message(
    step: dict[str, Any], time_state: dict[str, Any], expected: int | None = None
) -> str:
    return (
        f"Step {step['index'] + 1}{_of(expected)} at time {_num(step['time'])} is "
        f"written; {view_note(time_state)}."
    )


def start_message(result: dict[str, Any]) -> str:
    time_state = result.get("time") or {}
    steps = int(time_state.get("steps") or 0)
    growing = (
        "still being written"
        if result.get("growing")
        else "not marked as growing, so no new step is expected"
    )
    return (
        f"Watching scene {result.get('scene')}: {steps}"
        f"{_of(expected_steps(result))} step(s) so far, the source is {growing}; "
        f"{view_note(time_state)}."
    )


async def watch(
    fetch: Fetch,
    report: Report,
    *,
    after: int | None = None,
    idle_seconds: float = 300.0,
    max_seconds: float | None = None,
    poll_seconds: float | None = None,
    clock: Callable[[], float] = time.monotonic,
    sleep: Callable[[float], Awaitable[Any]] = asyncio.sleep,
) -> dict[str, Any]:
    """Report each step from index ``after`` on (default: the steps written from now)
    and return what was reported and why the watch ended.

    The progress reported is the number of steps written so far, against the total
    the view descriptor expects when it gives one; the watch also ends once that
    many are written. ``poll_seconds`` defaults to the descriptor's, else
    ``POLL_SECONDS``."""
    started = clock()
    result = await fetch(0 if after is None else max(0, after))
    following = len(result.get("steps") or []) if after is None else max(0, after)
    expected = expected_steps(result)
    poll = float(
        poll_seconds or (result.get("follow") or {}).get("poll_seconds") or POLL_SECONDS
    )
    await report(following, expected, start_message(result))
    reported: list[dict[str, Any]] = []
    count = 0
    last_new = started
    while True:
        fresh = [s for s in result.get("steps") or [] if s["index"] >= following]
        for step in fresh:
            count += 1
            following = step["index"] + 1
            reported.append(step)
            message = step_message(step, result.get("time") or {}, expected)
            await report(following, expected, message)
        if fresh:
            last_new = clock()
        now = clock()
        written = max(following, int((result.get("time") or {}).get("steps") or 0))
        if not result.get("growing"):
            reason = "not_growing"
        elif expected and written >= expected:
            reason = "complete"
        elif now - last_new >= idle_seconds:
            reason = "idle"
        elif max_seconds is not None and now - started >= max_seconds:
            reason = "max_seconds"
        else:
            await sleep(poll)
            result = await fetch(following)
            continue
        return {
            "scene": result.get("scene"),
            "ended": reason,
            "new_steps": count,
            "steps": reported[-KEEP:],
            "time": result.get("time") or {},
            "growing": bool(result.get("growing")),
            "expected_steps": expected,
            "seconds": round(now - started, 1),
        }


def watch_text(outcome: dict[str, Any]) -> str:
    """The agent-facing summary of a finished watch."""
    why = {
        "not_growing": "the source is not marked as growing",
        "complete": f"all {outcome.get('expected_steps')} expected steps are written",
        "idle": "no new step came for the idle time",
        "max_seconds": "the time limit was reached",
    }[outcome["ended"]]
    count = outcome["new_steps"]
    head = f"Watched scene {outcome['scene']} for {outcome['seconds']:g} s: "
    if count:
        first, last = outcome["steps"][0], outcome["steps"][-1]
        head += (
            f"{count} new step(s), step {first['index'] + 1} (time {_num(first['time'])}) "
            f"to step {last['index'] + 1} (time {_num(last['time'])})"
        )
    else:
        head += "no new step"
    time_state = outcome.get("time") or {}
    return f"{head}. It ended because {why}; {view_note(time_state)}."
