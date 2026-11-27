"""A view's background work: following a growing source, and catching a
late-opened one up from its first step."""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING, Any

from .held import LEDGER
from .sources import REPLAY_ALONE, REPLAY_HOLD, REPLAY_SECONDS, REPLAY_SLOW
from .turns import GROWTH_TICK, PAGE_HOLD, PAGE_QUIET, STAGE_BREAK, TURN_POLL, Shared

if TYPE_CHECKING:
    from .engine import Engine


class GrowthWork:
    """The follow and the catch-up of a view (``NodeApp``)."""

    engine: Engine
    server: Any
    shared: Shared
    name: str
    lost: str
    waiting: int
    sockets: int
    growth_busy: bool

    # -- background work ---------------------------------------------------------
    async def watch_growth(self, interval: float = 2.0) -> None:
        """Re-read growing sources; report new steps to the view."""
        spent = 0.0
        while True:
            # ParaView's calls block this loop, which also serves the view's page and
            # its link; over a slow link a check takes seconds, so the loop stays free
            # at least as long as the last check took. A view whose shared reader
            # another view just re-read moves at once.
            wait = max(interval, spent)
            waited = 0.0
            while waited < wait and not self._behind():
                await asyncio.sleep(GROWTH_TICK)
                waited += GROWTH_TICK
            await self.shared.turn()
            if self.lost:
                self.shared.follows(self, False)
                return
            if self.waiting or self._loading() or self.engine.opening:
                spent = 0.0
                continue
            spent = await self._grow_staged()
            self.shared.follows(self, self._following())

    async def _grow_staged(self) -> float:
        """One growth check a stage at a time (``growth_stages``). A step's disk
        look, re-read, update, colour ranges and picture took 8 to 10 s together
        over a slow link and held every view's loop that long; between stages the
        other views take their turns and commands get in, and each stage is a span
        of its own in the held-loop log. Commands and catch-up on this view wait
        for the last stage. Returns the seconds of work, waits left out."""
        spent = 0.0
        self.growth_busy = True
        try:
            with self.server.state:
                stages = self.engine.growth_stages()
            try:
                while not self.lost:
                    began = time.monotonic()
                    ended = False
                    try:
                        with LEDGER.span(f"{self.name} growth"), self.server.state:
                            try:
                                next(stages)
                            except StopIteration:
                                ended = True
                    except Exception as exc:  # noqa: BLE001 - keep watching; say why
                        print(f"growth check failed: {exc}", flush=True)
                        ended = True
                    spent += self.shared.worked(began)
                    if ended:
                        break
                    if self.shared.queue:
                        await self.shared.turn()
                    else:
                        await asyncio.sleep(STAGE_BREAK)
            finally:
                with self.server.state:
                    stages.close()
        finally:
            self.growth_busy = False
        return spent

    async def _after_growth(self) -> None:
        """A growth check part way through its stages finishes before this view's
        scene changes again."""
        while self.growth_busy and not self.lost:
            await asyncio.sleep(TURN_POLL)

    def _behind(self) -> bool:
        try:
            return bool(self.engine.behind())
        except Exception:  # noqa: BLE001 - a scene mid-change is looked at next tick
            return False

    def _following(self) -> bool:
        """Whether this view follows a growing source with a page open."""
        scene = self.engine.scene
        if scene is None or self.sockets <= 0:
            return False
        growing = bool((scene.get("dataset") or {}).get("growing"))
        return growing and bool(scene["time"].get("follow_latest", True))

    def _loading(self) -> bool:
        """Whether a view's page is loading: one connected and has not drawn yet
        (PAGE_DUE at most), or one fetched a file in the last PAGE_QUIET seconds, in
        a stretch of fetches under PAGE_HOLD seconds long. Pages that reload over and
        over do not hold background work for good."""
        now = time.monotonic()
        shared = self.shared
        if shared.page_loading():
            return True
        return (
            now - shared.last_fetch < PAGE_QUIET
            and now - shared.fetch_began < PAGE_HOLD
        )

    async def _replaying(self) -> None:
        """Catch a late-opened growing source up from its first step while a view
        watches; with no view for REPLAY_ALONE seconds it jumps to the latest."""
        engine = self.engine
        try:
            while self._loading():
                await asyncio.sleep(PAGE_QUIET)
            # On a node busy with other views every catch-up step waits this long
            # again, so a long wait for the first means the view stays on the
            # latest step and follows it rather than crawl up from step 1. So does
            # another view already following: the catch-up's steps would share the
            # loop with it, and slow both.
            waited = await self.shared.turn()
            others = self.shared.others_following(self)
            busy = self.shared.busy_waits()
            if others or waited > REPLAY_SLOW or busy > REPLAY_SLOW:
                engine.disarm_replay()
                why = (
                    f"{others} other view{'s' if others > 1 else ''} already "
                    "follow this node"
                    if others
                    else f"the node is busy (waited {waited:.0f} s for a turn)"
                    if waited > REPLAY_SLOW
                    else f"the node is busy (turns waited {busy:.0f} s lately)"
                )
                print(
                    f"{why}; following the latest step without a catch-up", flush=True
                )
                return
            await self._after_growth()
            began = time.monotonic()
            with LEDGER.span(f"{self.name} catch-up"), self.server.state:
                if not engine.start_replay():
                    return
            # The first step is the costly one (the reader's first read of a later
            # step), so it is what says whether the link can replay at all.
            spent = self.shared.worked(began)
            print(f"catching up from step 1 of {len(engine.timesteps())}", flush=True)
            await asyncio.sleep(max(REPLAY_HOLD, spent))
            alone = 0.0
            held = 0.0
            going = True
            while going:
                pause = max(REPLAY_SECONDS, spent)
                await asyncio.sleep(pause)
                # Time spent waiting for the turn holds the catch-up as much as
                # commands and loading pages do.
                waited = await self.shared.turn()
                held += waited
                # Held by commands or loading pages for REPLAY_ALONE in all, the
                # catch-up goes to the latest step rather than wait on for its turn.
                if (self.waiting or self._loading()) and held < REPLAY_ALONE:
                    held += pause
                    continue
                alone = 0.0 if self.sockets > 0 else alone + pause + waited
                if 0 < alone < REPLAY_ALONE:
                    continue
                # A step that takes longer than REPLAY_SLOW to show, or to get its
                # turn, cannot replay at a watchable pace: go to the latest instead.
                slow = (
                    spent > REPLAY_SLOW or waited > REPLAY_SLOW or held >= REPLAY_ALONE
                )
                await self._after_growth()
                began = time.monotonic()
                with LEDGER.span(f"{self.name} catch-up"), self.server.state:
                    going = engine.replay_tick(finish=alone >= REPLAY_ALONE or slow)
                if slow:
                    if spent > REPLAY_SLOW:
                        why = f"a step took {spent:.1f} s to show over this link"
                    elif waited > REPLAY_SLOW:
                        why = f"a step waited {waited:.0f} s for its turn"
                    else:
                        why = f"the catch-up was held {held:.0f} s"
                    print(f"{why}; skipped the rest of the catch-up", flush=True)
                spent = self.shared.worked(began)
            print(f"caught up; following step {len(engine.timesteps())}", flush=True)
        except Exception as exc:  # noqa: BLE001 - the view follows the latest step
            print(f"catch-up failed: {exc}", flush=True)
            with self.server.state:
                engine.replay_tick(finish=True)
