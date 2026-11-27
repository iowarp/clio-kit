"""Starting a node half: waiting on its log, and the errors it can end with."""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from .errors import ParaViewError

_JOIN_FAILED = "pvserver join failed: "


def _whole_lines(text: str, limit: int = 1500) -> str:
    """The end of ``text`` within ``limit`` characters, cut only between lines."""
    kept: list[str] = []
    size = 0
    for line in reversed(text.strip().splitlines()):
        if kept and size + len(line) + 1 > limit:
            break
        kept.append(line if len(line) <= limit else "…" + line[-limit:])
        size += len(line) + 1
    return "\n".join(reversed(kept))


def start_failure(where: str, log: str) -> ParaViewError:
    """The typed error for a node half whose log shows it stopped while starting."""
    for line in log.splitlines():
        if _JOIN_FAILED in line:
            target, _, reason = line.split(_JOIN_FAILED, 1)[1].partition(": ")
            return ParaViewError(
                "pvserver_join_failed",
                f"The ParaView node half started {where} but could not join the "
                f"pvserver at {target}: {reason or 'no reason given'}. Check that "
                "the pvserver is still running and that its forwarded port is open.",
            )
    return ParaViewError(
        "node_failed",
        f"The ParaView node half failed to start {where}:\n{_whole_lines(log)}",
    )


_JOINING = "joining pvserver at "


@dataclass
class StartLook:
    """One look at a node half that is starting."""

    record: dict[str, Any] | None
    #: The log's tail when the log shows that the start failed.
    failed: str | None
    alive: bool
    size: int
    last: str
    tail: str = ""


def wait_for_start(
    look: Callable[[], StartLook],
    code_hash: str,
    where: str,
    log_path: str,
    stall: float,
    cancel: threading.Event,
    progress: Callable[[str], None] | None = None,
    stop: Callable[[], None] | None = None,
    every: float = 1.0,
) -> dict[str, Any]:
    """Wait for a starting node half for as long as it shows progress.

    There is no overall limit. The start fails at once when the process ends or
    its log shows a failure, and only after ``stall`` seconds with no new log
    output; ``cancel`` ends the wait whenever it is set. Each new log line is
    passed to ``progress``, so the wait is visible.
    """
    began = changed = told = time.monotonic()
    size = -1
    said = ""
    dead = 0
    while True:
        if cancel.is_set():
            if stop is not None:
                stop()
            raise ParaViewError(
                "cancelled", f"Stopped starting the ParaView node half {where}."
            )
        seen = look()
        if seen.record is not None and seen.record.get("code_hash") == code_hash:
            return seen.record
        if seen.failed:
            raise start_failure(where, seen.failed)
        now = time.monotonic()
        if seen.size != size:
            size, changed = seen.size, now
        # A launcher may hand over to a child; take the process for gone only
        # when it has also stopped writing.
        dead = 0 if seen.alive else dead + 1
        if dead >= 3 and now - changed >= 2 * every:
            raise ParaViewError(
                "node_failed",
                f"The ParaView node half stopped while starting {where}. Its log "
                f"({log_path}) ends with:\n{_whole_lines(seen.tail or seen.last)}",
            )
        quiet = now - changed
        if progress is not None:
            line = seen.last.strip()
            if line != said or now - told >= 5.0:
                said, told = line, now
                progress(
                    f"Starting ParaView {where}, {now - began:.0f} s"
                    + (f": {line}" if line else "")
                )
        if quiet >= stall:
            if stop is not None:
                stop()
            last = seen.last.strip()
            if last.startswith(_JOINING):
                # The address stays in the node half's log, out of the text the
                # view shows on its own.
                raise ParaViewError(
                    "pvserver_join_stalled",
                    f"ParaView could not reach its server: it waited {quiet:.0f} s "
                    "with no answer. The server may be busy with another viewer, "
                    "or its connection may lead nowhere. Check the server, then "
                    "open the view again.",
                )
            raise ParaViewError(
                "node_start_stalled",
                f"The ParaView node half {where} showed no progress for {quiet:.0f} "
                f"s, so it was stopped. Its log ({log_path}) ends with:\n"
                f"{_whole_lines(seen.tail or last) or '(nothing)'}",
            )
        cancel.wait(every)


def gone_error(
    note: str, tail: str, where: str, scene: str | None = None
) -> ParaViewError:
    """The error for a linked node half that has gone, from what it left behind.

    ``note`` is the ``<slot>.lost.json`` it writes when its pvserver closes the
    connection; ``tail`` is the end of its log, for any other end.
    """
    try:
        lost = json.loads(note) if note.startswith("{") else None
    except ValueError:
        lost = None
    if isinstance(lost, dict) and not scene and isinstance(lost.get("scene"), str):
        scene = lost["scene"]
    again = (
        f" The scene {scene} is saved: open_view(scene={scene!r}) starts a new node "
        "half and opens it again."
        if scene
        else " open_view starts a new node half."
    )
    if isinstance(lost, dict):
        return ParaViewError(
            "pvserver_lost",
            f"The ParaView view {where} has stopped: {lost.get('reason')}. "
            "Nothing more can be drawn or read through it, and its live view "
            "shows that it stopped." + again + " The pvserver must be running "
            "again for that.",
        )
    return ParaViewError(
        "node_lost",
        f"The ParaView node half {where} has stopped, so its live view shows "
        "nothing more."
        + (f" Its log ends with:\n{_whole_lines(tail)}\n" if tail else "")
        + again,
    )
