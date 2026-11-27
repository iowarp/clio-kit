"""How the views of one node half share its loop and its pvserver: the turns
background work takes, and the pages it holds for."""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field

END_INTERACTION_EVENT = 45  # vtkCommand::EndInteractionEvent
#: Seconds after a view's page last fetched a file before background work resumes.
PAGE_QUIET = 1.5
#: Seconds a stretch of page fetches may hold background work at most.
PAGE_HOLD = 10.0
#: Seconds between looks at whether a background task's turn has come.
TURN_POLL = 0.05
#: Seconds the loop is left free between two stages of one piece of work. It must be
#: a real wait: ``sleep(0)`` gives the loop a single pass, while a timer or a socket
#: needs several (the timer fires, its future resolves, its task wakes; bytes arrive,
#: the reader is fed, the handler wakes), so the held-loop watch, commands and pages
#: only got in after the next stage had blocked the loop again.
STAGE_BREAK = 0.02
#: Seconds a background task may wait for its turn before the node log says so.
TURN_SLOW = 30.0
#: Seconds at most the loop is left free after a piece of background work. Commands
#: and loading pages get in however long the gap (background work skips its turn for
#: them), so a gap as long as the work only made every other view wait twice as long.
TURN_GAP = 2.0
#: How many of the latest turn waits say whether the node is busy.
TURN_RECENT = 6
#: Seconds a view counts as following after its last look at a growing source.
FOLLOWING_SECONDS = 60.0
#: Seconds the browser has to say it drew a scene it was sent before the next one
#: goes anyway (a page that went, a bundle that cannot say).
SCENE_WAIT = 60.0
#: Seconds after a page connects before it gets a fresh picture.
PAGE_FRESH = 1.0
#: Seconds between a following view's looks at whether a view sharing its reader
#: has re-read it.
GROWTH_TICK = 0.25
#: Seconds at most background work holds for a page loading until it draws.
PAGE_DUE = 15.0


@dataclass
class Shared:
    """What every view of the node half shares: they run on one loop, which each of
    ParaView's calls blocks, and on one pvserver."""

    #: Commands that arrived and have not started their work yet.
    waiting: int = 0
    #: Why the pvserver went, once it has; nothing is drawn after that.
    lost: str = ""
    #: When a view's page last fetched a file; background work holds while a page
    #: loads.
    last_fetch: float = 0.0
    #: When the current stretch of page fetches began.
    fetch_began: float = 0.0
    #: Until when the loop is left free after the last background work of any view.
    free_at: float = 0.0
    #: Background work waiting for its turn, in the order it asked.
    queue: deque[object] = field(default_factory=deque)
    #: The latest turn waits, in seconds.
    recent_waits: deque[float] = field(
        default_factory=lambda: deque(maxlen=TURN_RECENT)
    )
    #: Views following a growing source with a page open, by view, with when they
    #: last looked at it.
    following: dict[int, float] = field(default_factory=dict)
    #: Views whose page is loading, or connected and not yet drawn, with since when:
    #: background work holds for them (PAGE_DUE at most), so a page that a resize or
    #: a host remount reloaded links and draws at once rather than behind every
    #: view's follow.
    pages_due: dict[int, float] = field(default_factory=dict)

    def page_due(self, view: object) -> None:
        now = time.monotonic()
        since = self.pages_due.get(id(view))
        # A page that never drew holds nothing more until a while after its cap.
        if since is None or now - since >= 4 * PAGE_DUE:
            self.pages_due[id(view)] = now

    def page_drawn(self, view: object) -> None:
        self.pages_due.pop(id(view), None)

    def page_loading(self) -> bool:
        now = time.monotonic()
        return any(now - since < PAGE_DUE for since in self.pages_due.values())

    async def turn(self) -> float:
        """Wait for background work's turn: every view's follow and catch-up share
        the loop, so each leaves it free for a while after its work (TURN_GAP at
        most), and commands and pages get in however many views are open. Turns go
        in the order they were asked for, so one view's catch-up cannot keep another
        view's follow waiting. Returns the seconds it waited."""
        me = object()
        self.queue.append(me)
        asked = time.monotonic()
        try:
            while True:
                wait = self.free_at - time.monotonic()
                if wait > 0:
                    await asyncio.sleep(wait)
                elif self.queue[0] is me:
                    waited = time.monotonic() - asked
                    if waited > TURN_SLOW:
                        print(
                            f"background work waited {waited:.0f} s for its turn",
                            flush=True,
                        )
                    self.recent_waits.append(waited)
                    return waited
                else:
                    await asyncio.sleep(TURN_POLL)
        finally:
            self.queue.remove(me)

    def worked(self, began: float) -> float:
        """Note background work that began at ``began``; return how long it took."""
        now = time.monotonic()
        spent = now - began
        self.free_at = max(self.free_at, now + min(spent, TURN_GAP))
        return spent

    def busy_waits(self) -> float:
        """The average of the latest turn waits, in seconds (0 with none yet)."""
        waits = self.recent_waits
        return sum(waits) / len(waits) if waits else 0.0

    def follows(self, view: object, on: bool) -> None:
        """Note whether ``view`` follows a growing source with a page open."""
        if on:
            self.following[id(view)] = time.monotonic()
        else:
            self.following.pop(id(view), None)

    def others_following(self, view: object) -> int:
        """How many other views followed a growing source in the last
        FOLLOWING_SECONDS."""
        now = time.monotonic()
        return sum(
            1
            for key, seen in self.following.items()
            if key != id(view) and now - seen < FOLLOWING_SECONDS
        )
