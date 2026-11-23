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
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from . import manifest
from .changes import lines as change_lines
from .engine import MUTATING, STEPWISE, CommandError, Engine
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


TOKEN_HEADERS = ("X-Clio-Service-Token",)
END_INTERACTION_EVENT = 45  # vtkCommand::EndInteractionEvent
#: Seconds after a view's page last fetched a file before background work resumes.
PAGE_QUIET = 1.5

# Posts the bridge state and the link state to the page that frames trame.
BRIDGE_JS = r"""
(function () {
  function post(message) {
    try { window.parent.postMessage(message, '*'); } catch (e) { /* no parent */ }
  }
  var down = null;
  // A click that did not move is a pick; ParaView finds what is under it.
  document.addEventListener('pointerdown', function (e) {
    down = e.button === 0 ? { x: e.clientX, y: e.clientY } : null;
  }, true);
  document.addEventListener('pointerup', function (e) {
    var start = down;
    down = null;
    if (!start || !window.trame || !e.target.closest || !e.target.closest('.clio-view')) return;
    if (Math.abs(e.clientX - start.x) + Math.abs(e.clientY - start.y) > 4) return;
    var canvas = Array.prototype.find.call(document.querySelectorAll('.clio-view canvas'), function (c) {
      var r = c.getBoundingClientRect();
      return e.clientX >= r.left && e.clientX <= r.right && e.clientY >= r.top && e.clientY <= r.bottom;
    });
    if (!canvas) return;
    var box = canvas.getBoundingClientRect();
    var camera = null;
    try { camera = window.trame.refs.view.getCamera(); } catch (err) { camera = null; }
    window.trame.trigger('clio_pick', [e.clientX - box.left, box.height - (e.clientY - box.top),
      box.width, box.height, camera]);
  }, true);
  function start() {
    var trame = window.trame;
    if (!trame || !trame.state || !trame.state.watch) { setTimeout(start, 50); return; }
    trame.state.watch(['clio_bridge'], function (bridge) {
      if (bridge) post({ type: 'clio.paraview.state', state: bridge });
    });
    var client = trame.client;
    if (client && client.onConnectionClose) {
      client.onConnectionClose(function () { post({ type: 'clio.paraview.link', up: false }); });
      client.onConnectionError(function () { post({ type: 'clio.paraview.link', up: false }); });
    }
    post({ type: 'clio.paraview.link', up: true });
    // The view page answers with the host's theme and colours (theme.py).
    post({ type: 'clio.paraview.ready' });
    // A frame that mounts or resizes (a tab moved, the canvas maximized) asks for a
    // fresh picture: in remote mode nothing else would draw it until the next drag.
    // The remote view subscribes to images only once it has mounted and sized itself,
    // so ask again when the view's box changes and a little after the page starts.
    var redraw = null;
    function ask() {
      clearTimeout(redraw);
      redraw = setTimeout(function () { trame.trigger('clio_redraw'); }, 500);
    }
    [300, 1500, 4000].forEach(function (ms) { setTimeout(ask, ms); });
    window.addEventListener('resize', ask);
    (function watch() {
      var box = document.querySelector('.clio-view');
      if (!box || !window.ResizeObserver) { setTimeout(watch, 200); return; }
      new ResizeObserver(ask).observe(box);
    })();
    // A remote picture can land well after the page starts (a pvserver over a
    // forwarded link), and again after each resize; say when a full-size one shows
    // for the current size, so a host that looked early and saw nothing looks again.
    var drawnAt = '';
    var toldAt = 0;
    var untold = 0;
    (function watchPicture() {
      var box = document.querySelector('.clio-view');
      if (!box) { setTimeout(watchPicture, 200); return; }
      box.addEventListener('load', function (e) {
        var img = e.target;
        if (!img || img.tagName !== 'IMG') return;
        var rect = img.getBoundingClientRect();
        var size = Math.round(rect.width) + 'x' + Math.round(rect.height);
        if (rect.width < 1 || img.naturalWidth < rect.width * 0.9) return;
        // The node half counts the pictures its viewers really show, told at
        // most every two seconds while a drag streams them.
        untold += 1;
        var now = Date.now();
        if (size !== drawnAt || now - toldAt > 2000) {
          trame.trigger('clio_drawn', [size, untold]);
          toldAt = now;
          untold = 0;
        }
        if (size === drawnAt) return;
        drawnAt = size;
        post({ type: 'clio.paraview.drawn' });
      }, true);
    })();
    // The pvserver went: say so in place of the picture, and stop reconnecting.
    trame.state.watch(['clio_lost'], function (text) {
      if (text) post({ type: 'clio.paraview.lost', text: text });
    });
    if (trame.state.get && trame.state.get('clio_lost')) {
      post({ type: 'clio.paraview.lost', text: trame.state.get('clio_lost') });
    }
  }
  start();
})();
"""


def token_from(request: Any) -> str | None:
    """The token a request carries, from the bearer header or the CLIO header."""
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    for name in TOKEN_HEADERS:
        if request.headers.get(name):
            return request.headers[name].strip()
    return None


class NodeApp:
    """Binds the engine to trame and serves the command routes."""

    def __init__(self, engine: Engine, token: str, server: Any = None) -> None:
        from trame.app import get_server

        self.engine = engine
        self.token = token
        self.server = server or get_server("clio-paraview", client_type="vue3")
        self.last_link = time.monotonic()
        #: When the view's page last fetched a file; background work holds while a
        #: page loads, as each of ParaView's calls blocks the loop serving it.
        self.last_fetch = 0.0
        self.sockets = 0
        #: Commands that arrived and have not started their work yet.
        self.waiting = 0
        #: Calls sent with their own id, by id.
        self.jobs: dict[str, Job] = {}
        #: Whether a video is being rendered in the scene's view.
        self.animating = False
        self._replay: asyncio.Future[None] | None = None
        self.view: Any = None
        self.link_note = ""
        #: Why the pvserver went, once it has; nothing is drawn after that.
        self.lost = ""
        #: The pvserver socket and its address, when there is one to watch.
        self.watched: int | None = None
        self.pvserver = ""
        #: Pictures the viewers have shown, as their pages report them.
        self.frames_shown = 0
        self.last_frame: float | None = None
        self.last_frame_size = ""
        self._build()
        engine.listeners.append(self._on_scene_change)

    # -- trame -----------------------------------------------------------------
    def _build(self) -> None:
        from trame.widgets import client
        from trame.widgets import paraview as pv_widgets

        register_serializers(self._legend_title)
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
            if self.engine.replay_armed and (
                self._replay is None or self._replay.done()
            ):
                self._replay = asyncio.ensure_future(self._replaying())
        # The view frames trame from a sandboxed, opaque-origin page, so its
        # module scripts and fetches are cross-origin.
        cross_origin = not request.path.startswith("/clio/") and not websocket
        if cross_origin:
            self.last_fetch = time.monotonic()
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
                f"ParaView stopped: {reason}. Nothing more is drawn here; the scene "
                "is saved, and the agent can open it again once the pvserver runs."
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
            started()
            if name in STEPWISE:
                result = await self._stepwise(name, args)
            else:
                with self.server.state:
                    result = self.engine.run(name, args, by=by)
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
            # A small test frame: what the opened view really draws.
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
        went = self._went_during_call()
        if went is not None:
            return went
        return self._json({"result": result})

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
                    await asyncio.sleep(0)
            finally:
                with self.server.state:
                    steps.close()
        finally:
            if animating:
                self.animating = False

    async def _await_source(self, name: str, args: dict[str, Any]) -> None:
        """Before a growing source is opened, wait until it can be read: until it is
        there and, for an ADIOS2 BP5 folder, until a whole step is on disk. The wait
        lets the node go on answering. It ends when the source is readable, or once
        nothing on disk has changed for ``SOURCE_STALL`` seconds; open_view then says
        why the source cannot be read."""
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
                break
            if reason and reason != said:
                said = reason
                print(f"waiting for {path}: {reason}", flush=True)
            await asyncio.sleep(0.5)
        if said:
            print(f"waited {time.monotonic() - start:.1f}s for {path}", flush=True)

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
        self.bench.refresh(reason)
        if reason not in ("selection", "camera"):
            self._update_view()
        moved = reason in ("open_view", "fit_camera", "set_camera", "rollback")
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
        if self.server.state.viewMode != "remote":
            self.engine.simple.Render(self.engine.view)
            self.view.update()
        else:
            self.view.update_image()

    def _legend_title(self) -> str | None:
        """The field whose legend the browser shows: the selected object's, if on."""
        sel = self.server.state.clio_sel or {}
        return sel.get("color") if sel.get("scalar_bar") else None

    def _ui_select(self, object_id: str) -> None:
        self.bench.select(object_id)
        self.server.state.clio_bridge = self.bridge()
        self._update_view()  # the browser's one legend follows the selection

    def _ui_redraw(self) -> None:
        # Local mode draws in the browser; remote mode needs ParaView to render again.
        if self.lost:
            return
        if self.view is not None and self.server.state.viewMode == "remote":
            self.view.update()

    def _ui_drawn(self, size: str = "", count: Any = 1) -> None:
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
        scene["camera"] = self.engine.read_camera()
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
            # at least as long as the last check took.
            await asyncio.sleep(max(interval, spent))
            if self.lost:
                return
            if self.waiting or self._loading():
                spent = 0.0
                continue
            began = time.monotonic()
            try:
                with self.server.state:
                    self.engine.check_growth()
            except Exception as exc:  # noqa: BLE001 - keep watching; say why
                print(f"growth check failed: {exc}", flush=True)
            spent = time.monotonic() - began

    def _loading(self) -> bool:
        """Whether a view's page fetched a file in the last PAGE_QUIET seconds."""
        return time.monotonic() - self.last_fetch < PAGE_QUIET

    async def _replaying(self) -> None:
        """Catch a late-opened growing source up from its first step while a view
        watches; with no view for REPLAY_ALONE seconds it jumps to the latest."""
        engine = self.engine
        try:
            while self._loading():
                await asyncio.sleep(PAGE_QUIET)
            with self.server.state:
                if not engine.start_replay():
                    return
            print(f"catching up from step 1 of {len(engine.timesteps())}", flush=True)
            await asyncio.sleep(REPLAY_HOLD)
            alone = 0.0
            going = True
            spent = 0.0
            while going:
                pause = max(REPLAY_SECONDS, spent)
                await asyncio.sleep(pause)
                if self.waiting or self._loading():
                    continue
                alone = 0.0 if self.sockets > 0 else alone + pause
                if 0 < alone < REPLAY_ALONE:
                    continue
                # A step that takes longer than REPLAY_SLOW to show is a link too
                # slow to replay at a watchable pace: go to the latest step instead.
                slow = spent > REPLAY_SLOW
                began = time.monotonic()
                with self.server.state:
                    going = engine.replay_tick(finish=alone >= REPLAY_ALONE or slow)
                if slow:
                    print(
                        f"a step took {spent:.1f} s to show over this link; "
                        "skipped the rest of the catch-up",
                        flush=True,
                    )
                spent = time.monotonic() - began
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
