"""The node half's command routes: a front's call runs on its view's scene, on
its own when it has an id, with its progress for the front that asks."""

from __future__ import annotations

import asyncio
import contextvars
import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from .engine import MUTATING, STEPWISE, CommandError
from .held import LEDGER
from .sources import is_pattern
from .turns import STAGE_BREAK, TURN_POLL, Shared

if TYPE_CHECKING:
    from .engine import Engine

#: Seconds a growing source may show no change on disk before open_view stops
#: waiting for it to become readable.
SOURCE_STALL = 90.0
#: Times a growing source's open is tried again, each after the source changed,
#: before its error stands.
OPEN_TRIES = 5
#: Seconds a finished call's answer is kept, so a front whose copy was lost on the
#: way gets it again instead of running the call twice.
KEEP_ANSWER = 600.0
#: The longest a front may ask a call's answer to be waited for in one request.
MOST_WAIT = 60.0


@dataclass
class Job:
    """One call a front sent with its own id: it runs on its own, and the front
    asks how it goes until it ends. A call takes as long as its work does."""

    started: float
    progress: dict[str, Any] = field(default_factory=dict)
    task: asyncio.Future[None] | None = None
    #: The answer, as sent: body and status.
    answer: tuple[bytes, int] | None = None
    ended: float | None = None


#: The job the running call belongs to, for its progress.
_JOB: contextvars.ContextVar[Job | None] = contextvars.ContextVar("job", default=None)


def say_progress(**progress: Any) -> None:
    """How the running call goes, for the front that asks."""
    job = _JOB.get()
    if job is not None:
        job.progress = progress


class CommandRoutes:
    """The command routes of a view (``NodeApp``)."""

    engine: Engine
    server: Any
    shared: Shared
    stage: Any
    view: Any
    name: str
    lost: str
    waiting: int
    animating: bool
    jobs: dict[str, Job]
    _source_seen: tuple[Any, float] | None
    _json: Callable[..., Any]
    _went_during_call: Callable[[], Any]
    shown: Callable[[], dict[str, Any]]

    if TYPE_CHECKING:

        async def _after_growth(self) -> None: ...

    async def _command(self, request: Any) -> Any:
        try:
            body = await request.json()
            if not isinstance(body, dict):
                raise TypeError("the body is not an object")
        except (ValueError, TypeError):
            return self._json(
                {"error": {"code": "bad_request", "message": "Malformed command."}}, 400
            )
        if body.get("job"):
            return await self._job(str(body["job"]), body)
        return await self._execute(body)

    async def _job(self, job_id: str, body: dict[str, Any]) -> Any:
        """Start the call with this id, or ask how it goes: its answer once it has
        one, else after ``wait`` seconds a pending note with its progress."""
        from aiohttp import web

        now = time.monotonic()
        for known, old in list(self.jobs.items()):
            if old.ended is not None and now - old.ended > KEEP_ANSWER:
                del self.jobs[known]
        job = self.jobs.get(job_id)
        if job is None:
            job = self.jobs[job_id] = Job(started=now)
            job.task = asyncio.ensure_future(self._run_job(job, body))
        try:
            wait = min(MOST_WAIT, max(0.0, float(body.get("wait") or 0)))
        except (TypeError, ValueError):
            wait = 0.0
        if job.answer is None and wait and job.task is not None:
            await asyncio.wait({job.task}, timeout=wait)
        if job.answer is not None:
            payload, status = job.answer
            return web.Response(
                body=payload, status=status, content_type="application/json"
            )
        return self._json(
            {
                "pending": True,
                "seconds": round(time.monotonic() - job.started, 1),
                "progress": job.progress,
            }
        )

    async def _run_job(self, job: Job, body: dict[str, Any]) -> None:
        _JOB.set(job)
        try:
            response = await self._execute(body)
            job.answer = (bytes(response.body), int(response.status))
        except asyncio.CancelledError:
            job.answer = self._error_bytes(
                "cancelled", "The call was cancelled before it ended.", 409
            )
        except Exception as exc:  # noqa: BLE001 - the front gets the failure
            job.answer = self._error_bytes(
                "node_error", f"{type(exc).__name__}: {exc}", 500
            )
        finally:
            job.ended = time.monotonic()

    @staticmethod
    def _error_bytes(code: str, message: str, status: int) -> tuple[bytes, int]:
        payload = {"error": {"code": code, "message": message}}
        return json.dumps(payload).encode(), status

    async def _cancel(self, request: Any) -> Any:
        """Stop a running call: it ends at its next step with a ``cancelled`` error."""
        try:
            job = self.jobs.get(str((await request.json())["job"]))
        except (ValueError, KeyError, TypeError):
            return self._json(
                {"error": {"code": "bad_request", "message": "Malformed cancel."}}, 400
            )
        if job is None:
            return self._json({"ok": False, "known": False})
        if job.answer is None and job.task is not None:
            job.task.cancel()
        return self._json({"ok": True, "known": True, "ended": job.answer is not None})

    async def _execute(self, body: dict[str, Any]) -> Any:
        # Until its work starts, a command comes first: the follow and the catch-up
        # skip their turns, so each of the command's awaits returns at once rather
        # than after a step's round trips to a pvserver over a slow link.
        self.waiting += 1
        counted = True

        def started() -> None:
            nonlocal counted
            if counted:
                counted = False
                self.waiting -= 1

        try:
            return await self._run_command(body, started)
        finally:
            started()

    async def _run_command(self, body: dict[str, Any], started: Any) -> Any:
        try:
            name = str(body["command"])
            args = dict(body.get("args") or {})
            # The front says whose call it is; anything else is the agent's.
            by = "person" if body.get("by") == "person" else "agent"
        except (ValueError, KeyError, TypeError):
            return self._json(
                {"error": {"code": "bad_request", "message": "Malformed command."}}, 400
            )
        app = self
        found: dict[str, Any] = {"stage": self.stage}
        if self.stage is not None and not self.lost:
            try:
                app = await self.stage.route(name, args, by, found)
            except CommandError as exc:
                return self._json(
                    {"error": {"code": exc.code, "message": exc.message}}, 422
                )
            except Exception as exc:  # noqa: BLE001 - a view that could not start
                return self._json(
                    {
                        "error": {
                            "code": "node_error",
                            "message": "A view for this scene could not start: "
                            f"{type(exc).__name__}: {exc}",
                        }
                    },
                    500,
                )
        return await app._run_on(name, args, by, started, found)

    async def _run_on(
        self,
        name: str,
        args: dict[str, Any],
        by: str,
        started: Any,
        found: dict[str, Any] | None = None,
    ) -> Any:
        """Run one command on this view's scene."""
        if self.lost:
            scene = self.engine.scene
            again = (
                f" open_view(scene={scene['scene']!r}) opens it again once a "
                "pvserver runs."
                if scene
                else ""
            )
            return self._json(
                {
                    "error": {
                        "code": "pvserver_lost",
                        "message": f"This ParaView view has stopped: {self.lost}. "
                        "Nothing more is drawn or read through it." + again,
                    }
                },
                409,
            )
        try:
            await self._await_source(name, args)
            if name in MUTATING:
                await self._after_animation()
                await self._after_growth()
            started()
            if name == "steps":
                args = {**args, "check": self._steps_check()}
            if name in STEPWISE:
                result = await self._stepwise(name, args)
            else:
                result = await self._run_engine(name, args, by)
        except CommandError as exc:
            went = self._went_during_call()
            if went is not None:
                return went
            return self._json(
                {"error": {"code": exc.code, "message": exc.message}}, 422
            )
        except Exception as exc:  # noqa: BLE001 - report the failure to the front
            went = self._went_during_call()
            if went is not None:
                return went
            return self._json(
                {
                    "error": {
                        "code": "node_error",
                        "message": f"{type(exc).__name__}: {exc}",
                    }
                },
                500,
            )
        if name in ("set_camera", "fit_camera"):
            self.view.push_camera()
        if name == "open_view" and isinstance(result, dict) and self.engine.scene:
            # A small test frame: what the opened view really draws. It renders, so
            # it waits for its turn behind the other views' follows.
            if self.shared.queue:
                await self.shared.turn()
            try:
                result["drawn"] = self.engine.drawn()
            except Exception as exc:  # noqa: BLE001 - reported, not fatal
                result["drawn"] = {
                    "fraction": None,
                    "blank": None,
                    "note": f"the test frame failed: {type(exc).__name__}: {exc}",
                }
        if name in ("capture", "open_view") and isinstance(result, dict):
            result["shown"] = self.shown()
        if name == "open_view" and isinstance(result, dict):
            # The port this scene's view is served on.
            result["service"] = {"port": int(self.server.port)}
            result["already_open"] = bool((found or {}).get("already_open"))
        stage = (found or {}).get("stage")
        if name in ("open_view", "scene_state") and isinstance(result, dict) and stage:
            result["views"] = stage.views()
        went = self._went_during_call()
        if went is not None:
            return went
        return self._json({"result": result})

    def _remote_data(self) -> bool:
        return bool(getattr(getattr(self.engine, "disk", None), "remote", False))

    def _steps_check(self) -> bool:
        """Whether an agent's ``steps`` call checks the growing source itself. With
        the data on a joined pvserver a check is several round trips over what may
        be a slow link, and the follow checks anyway: the call says what the follow
        last read, and when. Checking there too, a watch polling every second kept
        the loop busy outside the turns, and every view's follow and catch-up but
        one waited for good."""
        return not self._remote_data()

    async def _run_engine(self, name: str, args: dict[str, Any], by: str) -> Any:
        """Run a command on the engine. Opening a growing source that its writer has
        not made readable yet (a step half written, a reader that sees no data yet) is
        tried again while the source goes on changing; once it stops changing, the
        error stands."""
        growing = name == "open_view" and args.get("growing") and args.get("path")
        tries = 0
        while True:
            while self.engine.opening and not self.lost:
                # A view opening a stage at a time: the scene is whole once it ends.
                await asyncio.sleep(TURN_POLL)
            try:
                if name == "open_view":
                    return await self._open_staged(args, by)
                with self.server.state:
                    return self.engine.run(name, args, by=by)
            except CommandError as exc:
                # An error that comes back after the source has changed several
                # times is not about a step half written.
                if (
                    not growing
                    or self.lost
                    or exc.code == "bad_arguments"
                    or tries >= OPEN_TRIES
                    or self._source_seen is None
                ):
                    raise
                path = self.engine.disk.expand(str(args["path"]))
                tries += 1
                print(f"opening {path} failed ({tries}): {exc}", flush=True)
                say_progress(
                    note=f"the source is not readable yet ({exc.message}); trying "
                    "again once it changes"
                )
                while not self._source_changed(path, exc):
                    await asyncio.sleep(1.0)

    async def _open_staged(self, args: dict[str, Any], by: str) -> Any:
        """open_view a stage at a time: each stage (the reader, its display, a slice,
        a colour map) can take seconds over a slow link, and between them the other
        views take their turns, so they go on following while this one opens."""
        engine = self.engine
        with self.server.state:
            stages = engine.staged(args, by)
        try:
            while True:
                if self.lost:
                    raise CommandError(
                        "pvserver_lost",
                        f"This ParaView view stopped while it opened: {self.lost}.",
                    )
                began = time.monotonic()
                ended, result, label = False, None, ""
                with LEDGER.span(f"{self.name} open"), self.server.state:
                    try:
                        label = next(stages)
                    except StopIteration as done:
                        ended, result = True, done.value
                if ended:
                    return result
                say_progress(note=f"opening the view: {label}")
                if self.shared.queue:
                    # Other views wait to follow their sources: they go first.
                    self.shared.worked(began)
                    await self.shared.turn()
                else:
                    await asyncio.sleep(STAGE_BREAK)
        finally:
            with self.server.state:
                stages.close()

    async def _after_animation(self) -> None:
        """While a video is rendered in the scene's view, changes to the scene wait
        for it: between its frames they would move the camera or what it shows."""
        while self.animating:
            say_progress(note="waiting for the video being rendered")
            await asyncio.sleep(0.2)

    async def _stepwise(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        """A ``STEPWISE`` command a step at a time, giving the loop back between
        steps: a long series or video must not hold up the view's follow, its step
        reports or health. Each step is reported as progress; a cancel ends it at
        the next step, and the command undoes what it set up."""
        engine = self.engine
        animating = name == "export_animation"
        self.animating = self.animating or animating
        try:
            with self.server.state:
                steps = engine.series(name, args)
            try:
                while True:
                    if self.lost:
                        raise CommandError(
                            "pvserver_lost",
                            f"This ParaView view stopped during {name}: {self.lost}.",
                        )
                    with self.server.state:
                        progress, result = engine.advance(steps)
                    if progress is None:
                        return dict(result)
                    say_progress(done=progress[0], total=progress[1])
                    await asyncio.sleep(STAGE_BREAK)
            finally:
                with self.server.state:
                    steps.close()
        finally:
            if animating:
                self.animating = False

    async def _await_source(self, name: str, args: dict[str, Any]) -> None:
        """Before a growing source is opened, wait until it can be read: until it is
        there and, for a pattern, until one file it matches has stopped changing. The
        wait lets the node go on answering. It ends when the source is readable; once it is
        not, and nothing in it or beside it has changed for ``SOURCE_STALL`` seconds,
        open_view says why instead of opening a view with nothing in it."""
        self._source_seen = None
        if name != "open_view" or not args.get("growing") or not args.get("path"):
            return
        path = self.engine.disk.expand(str(args["path"]))
        scene = self.engine.scene
        if (
            scene
            and (scene.get("dataset") or {}).get("path") == path
            and not args.get("new")
        ):
            return
        pattern = is_pattern(path)
        last: Any = object()
        changed = start = time.monotonic()
        said = None
        name_of = os.path.basename(path.rstrip("/")) or path
        while True:
            reason, key = self.engine.disk.source_readiness(path, pattern)
            now = time.monotonic()
            if key != last:
                last, changed = key, now
            if reason is None:
                break
            if now - changed >= SOURCE_STALL:
                if reason:
                    raise CommandError(
                        "source_not_ready",
                        f"{path} cannot be read yet: {reason}, and nothing in it or "
                        f"beside it has changed for {SOURCE_STALL:g} s, so the run "
                        "writing it may be queued, starting, or stopped. Check the "
                        "run, then call open_view again: it waits again while the "
                        "source changes.",
                    )
                break
            if reason and reason != said:
                said = reason
                print(f"waiting for {path}: {reason}", flush=True)
            say_progress(
                note=f"waiting for {name_of}: {reason or 'letting it settle'} "
                f"({now - start:.0f} s)"
            )
            await asyncio.sleep(0.5)
        if said:
            print(f"waited {time.monotonic() - start:.1f}s for {path}", flush=True)
        self._source_seen = (key, changed)

    def _source_changed(self, path: str, failure: CommandError) -> bool:
        """Whether a growing source that failed to open has changed on disk since;
        once it has stopped changing, the failure stands, with why."""
        seen = self._source_seen
        if seen is None:
            raise failure
        last, changed = seen
        _reason, key = self.engine.disk.source_readiness(path, is_pattern(path))
        now = time.monotonic()
        if key != last:
            self._source_seen = (key, now)
            return True
        if now - changed >= SOURCE_STALL:
            raise CommandError(
                failure.code,
                f"{failure.message} Nothing in {path} has changed for "
                f"{SOURCE_STALL:g} s since, so this stopped trying. Check the run "
                "that writes it, then call open_view again.",
            )
        return False
