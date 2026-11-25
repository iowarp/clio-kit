"""The trame application and the node half's command routes, on one port.

Every request, the trame page and its WebSocket included, must carry the
node half's token. The front sends it as a bearer token; the host's service
route adds it on the way through, so the browser never holds it.
"""

from __future__ import annotations

import asyncio
import contextvars
import hmac
import json
import os
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import manifest, vtkjs_patch
from .bridge import BRIDGE_JS, token_from
from .changes import lines as change_lines
from .delivery import fresh_stream
from .engine import MUTATING, STEPWISE, CommandError, Engine
from .held import LEDGER
from .panel import MODE_LABELS, build
from .serializers import register_serializers
from .sources import (
    REPLAY_ALONE,
    REPLAY_HOLD,
    REPLAY_SECONDS,
    REPLAY_SLOW,
    SETTLE,
    is_pattern,
)
from .theme import FALLBACK_CSS, THEME_JS, apply_to_view
from .workbench import Workbench

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


class NodeApp:
    """Binds one scene's engine to trame and serves the command routes.

    A node half has one of these per open scene, each on a port of its own, so
    views of different scenes (another variable, another dataset) stay live side by
    side. The first one, the front's, takes the commands and hands each to its
    scene's app (``stage``)."""

    def __init__(
        self,
        engine: Engine,
        token: str,
        server: Any = None,
        shared: Shared | None = None,
    ) -> None:
        from trame.app import get_server

        self.engine = engine
        self.token = token
        #: How the node log names this view.
        self.name = "view 1"
        self.server = server or get_server("clio-paraview", client_type="vue3")
        self.shared = shared or Shared()
        #: The node half's views, when there may be more than this one.
        self.stage: Any = None
        self.last_link = time.monotonic()
        self.sockets = 0
        #: Calls sent with their own id, by id.
        self.jobs: dict[str, Job] = {}
        #: Whether a video is being rendered in the scene's view.
        self.animating = False
        #: Whether a growth check is part way through its stages.
        self.growth_busy = False
        self._replay: asyncio.Future[None] | None = None
        self.view: Any = None
        self.link_note = ""
        #: The pvserver socket and its address, when there is one to watch.
        self.watched: int | None = None
        self.pvserver = ""
        #: Pictures the viewers have shown, as their pages report them.
        self.frames_shown = 0
        self.last_frame: float | None = None
        self.last_frame_size = ""
        #: What a growing source just opened had on disk, and when that last changed.
        self._source_seen: tuple[Any, float] | None = None
        #: When a scene went to the browser that it has not said it drew, and whether
        #: a newer one waits for that.
        self._scene_sent: float | None = None
        self._scene_waiting = False
        self._build()
        engine.listeners.append(self._on_scene_change)

    @property
    def waiting(self) -> int:
        return self.shared.waiting

    @waiting.setter
    def waiting(self, value: int) -> None:
        self.shared.waiting = value

    @property
    def lost(self) -> str:
        return self.shared.lost

    @lost.setter
    def lost(self, value: str) -> None:
        self.shared.lost = value

    # -- trame -----------------------------------------------------------------
    def _build(self) -> None:
        from trame.widgets import client
        from trame.widgets import paraview as pv_widgets

        register_serializers(self._legend_for)
        state, ctrl = self.server.state, self.server.controller
        state.trame__title = "ParaView"
        state.update(
            clio_bridge=None,
            clio_label="No scene yet",
            clio_path="",
            clio_steps=0,
            clio_index=0,
            clio_follow=True,
            clio_catching=False,
            clio_newer=0,
            clio_mode_note="",
            clio_theme="light",
            clio_view_colors=None,
            viewMode="local",
            clio_drag=None,
            clio_lost="",
        )
        # A value slider's drag in progress stays in the browser; release applies it.
        state.client_only("clio_drag")
        state.change("clio_view_colors")(self._ui_theme)
        bench = self.bench = Workbench(self.engine, state, after=self._after_person)
        for name, handler in {
            "clio_set_time": bench.set_time,
            "clio_latest": bench.latest,
            "clio_step": bench.step,
            "clio_play": bench.play,
            "clio_select": self._ui_select,
            "clio_visible": bench.set_visible,
            "clio_remove": bench.remove,
            "clio_display": bench.display,
            "clio_camera": bench.camera,
            "clio_filter_form": bench.open_form,
            "clio_filter_apply": bench.apply_form,
            "clio_filter_edit": bench.edit_form,
            "clio_set_mode": self._ui_set_mode,
            "clio_pick": self._ui_pick,
            "clio_redraw": self._ui_redraw,
            "clio_drawn": self._ui_drawn,
            "clio_scene_loaded": self._ui_scene_loaded,
        }.items():
            ctrl.trigger(name)(handler)
        ctrl.on_server_bind.add(self._bind_routes)

        def make_view() -> Any:
            client.Style(FALLBACK_CSS)
            client.Script(THEME_JS)
            client.Script(BRIDGE_JS)
            return pv_widgets.VtkRemoteLocalView(
                self.engine.view,
                namespace="view",
                mode=("viewMode", "local"),
                interactive_ratio=0.5,
                interactive_quality=50,
                interactor_events=("['EndAnimation']",),
                EndAnimation=(self._ui_camera, "[trame.refs.view.getCamera()]"),
            )

        self.layout, self.view = build(self.server, make_view)
        ctrl.view_update = self.view.update
        # In remote mode the camera moves on the server; report it after a drag.
        interactor = self.engine.view.GetInteractor()
        if interactor is not None:
            interactor.AddObserver(END_INTERACTION_EVENT, self._remote_camera)

    def _after_person(self, name: str) -> None:
        """After a person's command commits: move the browser's camera if it moved."""
        if name in ("set_camera", "fit_camera") and self.view is not None:
            self.view.push_camera()

    def _bind_routes(self, wslink_server: Any) -> None:
        from aiohttp import web

        app = wslink_server.app

        @web.middleware
        async def guard(request: Any, handler: Callable[..., Any]) -> Any:
            return await self._guard(request, handler)

        app.middlewares.append(guard)
        app.router.add_get("/clio/health", self._health)
        app.router.add_post("/clio/ping", self._ping)
        app.router.add_post("/clio/command", self._command)
        app.router.add_post("/clio/cancel", self._cancel)
        # trame-client asks a launcher for its session before connecting. There
        # is none: an empty answer makes it connect to the WebSocket beside its
        # own page. (A sessionURL parameter would be lowercased by trame-client,
        # which breaks a host route whose capability is case-sensitive.)
        app.router.add_post("/paraview/", self._no_launcher)

    @staticmethod
    def _json(payload: Any, status: int = 200) -> Any:
        from aiohttp import web

        return web.json_response(payload, status=status, dumps=json.dumps)

    # aiohttp middleware: the token gate in front of every route.
    async def _guard(self, request: Any, handler: Callable[..., Any]) -> Any:
        from aiohttp import web

        presented = token_from(request)
        if presented is None or not hmac.compare_digest(presented, self.token):
            return web.Response(status=401, text="A valid token is required.")
        self.last_link = time.monotonic()
        websocket = request.headers.get("Upgrade", "").lower() == "websocket"
        if websocket:
            self.sockets += 1
            # Its first picture comes next; background work holds until it draws.
            self.shared.page_due(self)
            self.shared.last_fetch = time.monotonic()
            # A page that opens (a new view, a frame the host remounted, a link that
            # came back) gets a fresh picture: the scene a page that went never said
            # it drew is not waited for, and a remote stream that page left is reset.
            asyncio.get_running_loop().call_later(PAGE_FRESH, self._new_page)
            # A scene's catch-up runs once, for its first viewer; a view that opens
            # later shows the step the others show.
            if self.sockets > 1:
                self.engine.disarm_replay()
            elif self.engine.replay_armed and (
                self._replay is None or self._replay.done()
            ):
                self._replay = asyncio.ensure_future(self._replaying())
        # The view frames trame from a sandboxed, opaque-origin page, so its
        # module scripts and fetches are cross-origin.
        cross_origin = not request.path.startswith("/clio/") and not websocket
        if cross_origin:
            now = time.monotonic()
            shared = self.shared
            if now - shared.last_fetch >= PAGE_QUIET:
                shared.fetch_began = now
            shared.last_fetch = now
            if self.sockets <= 0:
                # A page of this view is loading (a resize or remount reloads it).
                shared.page_due(self)
        try:
            response = await handler(request)
            if cross_origin:
                response.headers["Access-Control-Allow-Origin"] = "*"
            return response
        except web.HTTPException as exc:
            if cross_origin:
                exc.headers["Access-Control-Allow-Origin"] = "*"
            raise
        finally:
            if websocket:
                self.sockets -= 1
                self.last_link = time.monotonic()

    async def _health(self, _request: Any) -> Any:
        from . import NODE_PROTOCOL

        scene = self.engine.scene
        return self._json(
            {
                "ok": True,
                "protocol": NODE_PROTOCOL,
                "scene": scene["scene"] if scene else None,
                "session": self.engine.session,
                **self.shown(),
            }
        )

    def shown(self) -> dict[str, Any]:
        """What the live view really shows: its viewers and the pictures they drew."""
        return {
            "lost": self.lost or None,
            "viewers": self.sockets,
            "frames_shown": self.frames_shown,
            "last_frame_age": None
            if self.last_frame is None
            else round(time.time() - self.last_frame, 1),
            "last_frame_size": self.last_frame_size or None,
        }

    def pvserver_lost(self, reason: str) -> None:
        """The pvserver went: nothing more is drawn, and the view says why."""
        self.lost = reason
        with self.server.state:
            self.server.state.clio_lost = (
                f"ParaView stopped: {reason}. The scene is saved, and the agent "
                "can open it again once the server runs."
            )

    def _went_during_call(self) -> Any:
        """A refusal when the pvserver closed while the call ran, else ``None``.

        ParaView returns from a call whose pvserver went (a send fails, a render
        falls back to an empty frame), so what it returned is not what the pvserver
        holds; the watcher leaves its note within seconds.
        """
        if self.watched is None or self.lost:
            return None
        from . import liveness

        if not liveness.closed(self.watched):
            return None
        return self._json(
            {
                "error": {
                    "code": "pvserver_lost",
                    "message": f"The pvserver at {self.pvserver} closed the "
                    "connection while this call ran, so nothing it returned is "
                    "real. Nothing more is drawn or read through this view.",
                }
            },
            409,
        )

    async def _no_launcher(self, _request: Any) -> Any:
        return self._json(None)

    async def _ping(self, _request: Any) -> Any:
        return self._json({"ok": True})

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
                while not self._source_changed(path, args.get("reader"), exc):
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
        there and, for an ADIOS2 BP5 folder, until a whole step is on disk. The wait
        lets the node go on answering. It ends when the source is readable; once it is
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
        reader = args.get("reader")
        last: Any = object()
        changed = start = time.monotonic()
        said = None
        name_of = os.path.basename(path.rstrip("/")) or path
        while True:
            reason, clear, key = self.engine.disk.source_readiness(
                path, reader, pattern
            )
            now = time.monotonic()
            if key != last:
                last, changed = key, now
            if reason is None and (clear or now - changed >= SETTLE):
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

    def _source_changed(self, path: str, reader: Any, failure: CommandError) -> bool:
        """Whether a growing source that failed to open has changed on disk since;
        once it has stopped changing, the failure stands, with why."""
        seen = self._source_seen
        if seen is None:
            raise failure
        last, changed = seen
        _reason, _clear, key = self.engine.disk.source_readiness(
            path, reader, is_pattern(path)
        )
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

    # -- reacting to the scene -------------------------------------------------
    def _on_scene_change(self, reason: str) -> None:
        engine = self.engine
        state = self.server.state
        scene = engine.scene
        if scene is not None:
            dataset = scene.get("dataset") or {}
            path = str(dataset.get("path") or "")
            state.clio_label = (
                f"ParaView session {engine.session}"
                if engine.session and not path
                else os.path.basename(path.rstrip("/")) or scene["scene"]
            )
            state.clio_path = path
            mode = scene["view"].get("mode", "local")
            state.viewMode = mode
            size = scene["view"].get("geometry_mb", 0)
            switched = mode == "remote" and not scene["view"].get("pinned")
            state.clio_mode_note = (
                ""
                if not switched
                else "Switched: a volume is drawn on the ParaView machine"
                if scene["view"].get("reason") == "volume"
                else "Switched: an image slice is drawn on the ParaView machine"
                if scene["view"].get("reason") == "slice"
                else "Switched: a growing source is drawn on the ParaView machine"
                if scene["view"].get("reason") == "growing"
                else f"Switched: geometry {size:.1f} MB, over {engine.threshold_mb:g} MB"
            )
        time_state = engine.time_summary()
        state.clio_steps = time_state.get("steps", 0)
        state.clio_index = time_state.get("index", 0)
        state.clio_follow = time_state.get("follow_latest", True)
        state.clio_catching = time_state.get("catching_up", False)
        state.clio_newer = time_state.get("newer_steps", 0)
        with LEDGER.phase("panel"):
            self.bench.refresh(reason)
        if reason not in ("selection", "camera"):
            with LEDGER.phase("picture"):
                self._update_view()
        moved = reason in ("open_view", "fit_camera", "set_camera", "rollback") or (
            reason in ("growth", "replay") and engine.reframed
        )
        if moved and self.view is not None:
            self.view.push_camera()
        state.clio_bridge = self.bridge()

    def bridge(self) -> dict[str, Any] | None:
        """The key state the view reports to its host."""
        scene = self.engine.scene
        if scene is None:
            return None
        pipeline = []
        for o in manifest.pipeline(scene):
            display = o.get("display") or {}
            color = display.get("color_by") or [None, None]
            pipeline.append(
                {
                    "id": o["id"],
                    "label": o.get("label"),
                    "type": o.get("type"),
                    "input": o.get("input"),
                    "properties": o.get("properties") or {},
                    "visible": display.get("visible", True),
                    "representation": display.get("representation"),
                    "color_by": color[1],
                    "color_component": display.get("color_component"),
                    "colormap": display.get("colormap"),
                    "opacity": display.get("opacity", 1.0),
                    "scalar_bar": bool(display.get("scalar_bar")),
                }
            )
        selection = scene.get("selection")
        changes = self.engine.changes.recent()
        return {
            "scene": scene["scene"],
            "dataset": (scene.get("dataset") or {}).get("path"),
            "growing": bool((scene.get("dataset") or {}).get("growing")),
            "time": self.engine.time_summary(),
            "selection": None
            if not selection
            else {
                "object": selection.get("object"),
                "point_id": selection.get("point_id"),
                "point": selection.get("point"),
                "values": selection.get("point_values"),
            },
            "pipeline": pipeline,
            "representation": [p for p in pipeline if p["visible"]],
            "active_object": self.server.state.clio_active,
            "camera": scene.get("camera"),
            "view_mode": scene["view"].get("mode"),
            "changes": changes,
            "changes_text": change_lines(changes),
        }

    # -- the person's actions in the view -----------------------------------------
    def _ui_theme(self, clio_view_colors: Any = None, **_: Any) -> None:
        """Draw the render view in the host's tone (how it is drawn, not scene state)."""
        if not clio_view_colors or len(clio_view_colors) != 2:
            return
        engine = self.engine
        background, foreground = clio_view_colors
        sources = list(engine.proxies.values())
        applied = apply_to_view(
            engine.simple, engine.view, background, foreground, sources
        )
        if applied:
            self._update_view()

    def _update_view(self) -> None:
        """Send the scene to the view. ParaView brings its legends up to date
        (title, text colours, range) only when it renders, so render first when the
        browser draws, or the browser gets a legend ParaView has not set up yet.

        A view drawn on the ParaView machine gets only its new image: the geometry
        the browser would draw is not needed, and bringing it over is the slow part
        of each step on a slow link. Switching to the browser sends it then."""
        if self.view is None or self.lost:
            return
        with LEDGER.drawing():
            if self.server.state.viewMode != "remote":
                if self._scene_held():
                    self._scene_waiting = True
                    return
                self._send_scene()
            else:
                self.view.update_image()

    def _scene_held(self) -> bool:
        """Whether the next scene waits for the browser to draw the last one sent.
        Each scene's arrays are fetched after it lands, and over a slow link a page
        sent a scene per step fetched them all at once and drew none."""
        sent = self._scene_sent
        if sent is None or not vtkjs_patch.scene_signal or self.sockets <= 0:
            return False
        if time.monotonic() - sent < SCENE_WAIT:
            return True
        print(
            f"the browser has not said it drew the scene sent {SCENE_WAIT:g} s ago; "
            "sending the newest",
            flush=True,
        )
        return False

    def _send_scene(self) -> None:
        self._scene_waiting = False
        self.engine.simple.Render(self.engine.view)
        self.view.update()
        if vtkjs_patch.scene_signal and self.sockets > 0:
            self._scene_sent = time.monotonic()

    def _ui_scene_loaded(self, ok: Any = True, error: Any = "") -> None:
        """A page drew the scene it was sent, or could not: send the newest, if one
        waits."""
        if not ok:
            print(f"the browser could not draw a scene: {error}", flush=True)
        self._scene_sent = None
        self.shared.page_drawn(self)
        if (
            self._scene_waiting
            and not self.lost
            and self.view is not None
            and self.server.state.viewMode != "remote"
        ):
            self._send_scene()

    def _new_page(self) -> None:
        """A fresh picture for a page that just connected."""
        if self.lost or self.view is None or self.sockets <= 0:
            return
        if self.server.state.viewMode == "remote":
            fresh_stream(self.server, self.engine.view, 0, 0)
            self.view.update_image()
            return
        self._scene_sent = None
        if self._scene_waiting:
            with self.server.state:
                self._send_scene()

    def _legend_for(self, renderer: Any) -> tuple[bool, str | None]:
        """Whether ``renderer`` is this view's, and the field whose legend the browser
        shows there: the selected object's, if its legend is on."""
        try:
            mine = self.engine.view.GetClientSideObject().GetRenderWindow()
            window = renderer.GetRenderWindow() if renderer is not None else None
            if window is None or mine.__this__ != window.__this__:
                return False, None
        except Exception:  # noqa: BLE001 - not a renderer of this view
            return False, None
        sel = self.server.state.clio_sel or {}
        return True, sel.get("color") if sel.get("scalar_bar") else None

    def _ui_select(self, object_id: str) -> None:
        self.bench.select(object_id)
        self.server.state.clio_bridge = self.bridge()
        self._update_view()  # the browser's one legend follows the selection

    def _ui_redraw(self, width: Any = 0, height: Any = 0, fresh: Any = False) -> None:
        # Local mode draws in the browser; remote mode needs ParaView to render again,
        # at the asking page's size, from a stream a page that went may have left stuck.
        if self.lost:
            return
        if self.view is not None and self.server.state.viewMode == "remote":
            if fresh:
                try:
                    w, h = int(width), int(height)
                except (TypeError, ValueError):
                    w = h = 0
                fresh_stream(self.server, self.engine.view, w, h)
                # A page with no picture at this size (a resize) draws before any
                # view's background work runs again.
                self.shared.page_due(self)
            # The picture alone: the geometry is the browser's to draw, and sending
            # it is the slow part over a slow link.
            self.view.update_image()

    def _ui_drawn(self, size: str = "", count: Any = 1) -> None:
        self.shared.page_drawn(self)
        try:
            self.frames_shown += max(1, min(int(count), 10_000))
        except (TypeError, ValueError):
            self.frames_shown += 1
        self.last_frame = time.time()
        self.last_frame_size = str(size)[:32]

    def _ui_set_mode(self, mode: str) -> None:
        if self.engine.scene is None or mode not in MODE_LABELS:
            return
        if mode != self.engine.scene["view"].get("mode", "local"):
            self.engine.set_view_mode(mode)

    def _ui_camera(self, camera: dict[str, Any] | None) -> None:
        if not camera or self.engine.scene is None:
            return
        self.engine.browser_camera(camera)
        self.server.state.clio_bridge = self.bridge()

    def _remote_camera(self, *_args: Any) -> None:
        scene = self.engine.scene
        if scene is None or scene["view"].get("mode") != "remote":
            return
        self.engine.person_moved_camera()
        with self.server.state:
            self.server.state.clio_bridge = self.bridge()

    def _ui_pick(
        self,
        x: float,
        y: float,
        width: float,
        height: float,
        camera: dict[str, Any] | None = None,
    ) -> None:
        engine = self.engine
        if engine.scene is None:
            return
        if camera and engine.scene["view"].get("mode") == "local":
            engine.browser_camera(camera)
        world = engine.pick_display(x, y, width, height)
        if world is not None:
            engine.browser_pick(world)

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

    def idle_for(self) -> float:
        """Seconds since the last request, or zero while a view is connected."""
        if self.sockets > 0:
            return 0.0
        return time.monotonic() - self.last_link
