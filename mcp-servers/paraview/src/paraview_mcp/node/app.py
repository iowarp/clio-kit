"""The trame application and the node half's command routes, on one port.

Every request, the trame page and its WebSocket included, must carry the
node half's token. The front sends it as a bearer token; the host's service
route adds it on the way through, so the browser never holds it.
"""

from __future__ import annotations

import asyncio
import hmac
import json
import os
import time
from collections.abc import Callable
from typing import Any

from . import manifest, vtkjs_patch
from .bridge import BRIDGE_JS, token_from
from .changes import lines as change_lines
from .commands import CommandRoutes, Job
from .delivery import fresh_stream
from .engine import Engine
from .growth import GrowthWork
from .held import LEDGER
from .panel import MODE_LABELS, build
from .serializers import register_serializers
from .theme import FALLBACK_CSS, THEME_JS, apply_to_view
from .turns import (
    END_INTERACTION_EVENT,
    PAGE_FRESH,
    PAGE_QUIET,
    SCENE_WAIT,
    Shared,
)
from .workbench import Workbench


class NodeApp(CommandRoutes, GrowthWork):
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

    def idle_for(self) -> float:
        """Seconds since the last request, or zero while a view is connected."""
        if self.sockets > 0:
            return 0.0
        return time.monotonic() - self.last_link
