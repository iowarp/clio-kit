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

from .changes import lines as change_lines
from .engine import CommandError, Engine
from .panel import MODE_LABELS, build
from .serializers import register_serializers
from .theme import FALLBACK_CSS, THEME_JS, apply_to_view
from .workbench import Workbench

TOKEN_HEADERS = ("X-Clio-Service-Token",)
END_INTERACTION_EVENT = 45  # vtkCommand::EndInteractionEvent

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
        self.sockets = 0
        self.view: Any = None
        self.link_note = ""
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
            clio_newer=0,
            clio_mode_note="",
            clio_theme="light",
            clio_view_colors=None,
            viewMode="local",
        )
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
            "clio_set_mode": self._ui_set_mode,
            "clio_pick": self._ui_pick,
            "clio_redraw": self._ui_redraw,
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
        # The view frames trame from a sandboxed, opaque-origin page, so its
        # module scripts and fetches are cross-origin.
        cross_origin = not request.path.startswith("/clio/") and not websocket
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
            }
        )

    async def _no_launcher(self, _request: Any) -> Any:
        return self._json(None)

    async def _ping(self, _request: Any) -> Any:
        return self._json({"ok": True})

    async def _command(self, request: Any) -> Any:
        try:
            body = await request.json()
            name = str(body["command"])
            args = dict(body.get("args") or {})
            # The front says whose call it is; anything else is the agent's.
            by = "person" if body.get("by") == "person" else "agent"
        except (ValueError, KeyError, TypeError):
            return self._json(
                {"error": {"code": "bad_request", "message": "Malformed command."}}, 400
            )
        try:
            with self.server.state:
                result = self.engine.run(name, args, by=by)
        except CommandError as exc:
            return self._json(
                {"error": {"code": exc.code, "message": exc.message}}, 422
            )
        except Exception as exc:  # noqa: BLE001 - report the failure to the front
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
        return self._json({"result": result})

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
                else f"Switched: geometry {size:.1f} MB, over {engine.threshold_mb:g} MB"
            )
        time_state = engine.time_summary()
        state.clio_steps = time_state.get("steps", 0)
        state.clio_index = time_state.get("index", 0)
        state.clio_follow = time_state.get("follow_latest", True)
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
        for o in scene["objects"]:
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
        browser draws, or the browser gets a legend ParaView has not set up yet."""
        if self.view is None:
            return
        if self.server.state.viewMode != "remote":
            self.engine.simple.Render(self.engine.view)
        self.view.update()

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
        if self.view is not None and self.server.state.viewMode == "remote":
            self.view.update()

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
        while True:
            await asyncio.sleep(interval)
            try:
                with self.server.state:
                    self.engine.check_growth()
            except Exception as exc:  # noqa: BLE001 - keep watching; say why
                print(f"growth check failed: {exc}", flush=True)

    def idle_for(self) -> float:
        """Seconds since the last request, or zero while a view is connected."""
        if self.sockets > 0:
            return 0.0
        return time.monotonic() - self.last_link
