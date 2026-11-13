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

from .engine import CommandError, Engine

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
  }
  start();
})();
"""

STYLE = """
html, body, #app { margin: 0; height: 100%; font: 13px system-ui, sans-serif; }
.clio-root { display: flex; flex-direction: column; height: 100vh; }
.clio-bar { display: flex; flex-wrap: wrap; gap: 8px 14px; align-items: center;
  padding: 6px 10px; background: #f4f5f7; color: #1d2330; border-bottom: 1px solid #d5d8de; }
.clio-bar button { font: inherit; padding: 3px 8px; border: 1px solid #9aa3b2;
  border-radius: 4px; background: #fff; color: inherit; cursor: pointer; }
.clio-bar input[type=range] { width: 160px; }
.clio-view { flex: 1; position: relative; min-height: 120px; }
.clio-note { color: #4a5263; }
@media (prefers-color-scheme: dark) {
  .clio-bar { background: #1f232b; color: #e6e9ef; border-color: #363c48; }
  .clio-bar button { background: #2a2f39; border-color: #4b5262; }
  .clio-note { color: #b5bccb; }
}
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


def register_composite_serializers() -> None:
    """Let trame-vtk send ParaView 6's partitioned datasets to the browser.

    ParaView 6 representations hand their mappers a
    vtkPartitionedDataSetCollection; trame-vtk 2.11 has no serializer for it,
    so local rendering would show an empty scene. trame-vtk's own
    merge-to-polydata serializer handles any composite dataset.
    """
    from trame_vtk.modules.vtk.serializers import initialize_serializers
    from trame_vtk.modules.vtk.serializers.data import merge_to_polydata_serializer
    from trame_vtk.modules.vtk.serializers.registry import (
        register_instance_serializer,
    )

    initialize_serializers()
    for name in ("vtkPartitionedDataSetCollection", "vtkPartitionedDataSet"):
        register_instance_serializer(name, merge_to_polydata_serializer)


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
        from trame.ui.html import DivLayout
        from trame.widgets import client, html
        from trame.widgets import paraview as pv_widgets

        register_composite_serializers()
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
            clio_mode_text="",
            clio_mode_note="",
            viewMode="local",
        )
        ctrl.trigger("clio_set_time")(self._ui_time)
        ctrl.trigger("clio_latest")(self._ui_latest)
        ctrl.trigger("clio_toggle_mode")(self._ui_toggle_mode)
        ctrl.trigger("clio_pick")(self._ui_pick)
        ctrl.on_server_bind.add(self._bind_routes)

        with DivLayout(self.server) as layout:
            client.Style(STYLE)
            client.Script(BRIDGE_JS)
            with html.Div(classes="clio-root"):
                with html.Div(classes="clio-bar"):
                    html.Strong("{{ clio_label }}", title=("clio_path",))
                    with html.Span(v_if="clio_steps > 1"):
                        html.Span("Step {{ clio_index + 1 }} of {{ clio_steps }}")
                        html.Input(
                            type="range",
                            min=0,
                            max=("clio_steps - 1",),
                            value=("clio_index",),
                            aria_label="Time step",
                            change="trigger('clio_set_time', [Number($event.target.value)])",
                        )
                        html.Button(
                            "Follow latest",
                            v_if="!clio_follow",
                            click="trigger('clio_latest')",
                        )
                        html.Span(
                            "Following the latest step",
                            v_if="clio_follow",
                            classes="clio-note",
                        )
                        html.Span(
                            "{{ clio_newer }} newer step(s) available",
                            v_if="!clio_follow && clio_newer > 0",
                            classes="clio-note",
                        )
                    html.Span("{{ clio_mode_text }}")
                    html.Span("{{ clio_mode_note }}", classes="clio-note")
                    html.Button(
                        "{{ viewMode === 'local' ? 'Render in ParaView' "
                        ": 'Render in this browser' }}",
                        click="trigger('clio_toggle_mode')",
                    )
                with html.Div(classes="clio-view"):
                    self.view = pv_widgets.VtkRemoteLocalView(
                        self.engine.view,
                        namespace="view",
                        mode=("viewMode", "local"),
                        interactive_ratio=0.5,
                        interactive_quality=50,
                        interactor_events=("['EndAnimation']",),
                        EndAnimation=(self._ui_camera, "[trame.refs.view.getCamera()]"),
                    )
        self.layout = layout
        ctrl.view_update = self.view.update
        # In remote mode the camera moves on the server; report it after a drag.
        interactor = self.engine.view.GetInteractor()
        if interactor is not None:
            interactor.AddObserver(END_INTERACTION_EVENT, self._remote_camera)

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
        except (ValueError, KeyError, TypeError):
            return self._json(
                {"error": {"code": "bad_request", "message": "Malformed command."}}, 400
            )
        try:
            with self.server.state:
                result = self.engine.run(name, args)
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
            state.clio_mode_text = (
                "Drawn in this browser" if mode == "local" else "Drawn by ParaView"
            )
            if mode == "remote" and not scene["view"].get("pinned"):
                state.clio_mode_note = (
                    f"geometry {size:.1f} MB is above the "
                    f"{engine.threshold_mb:g} MB remote render threshold"
                )
            elif scene["view"].get("pinned"):
                state.clio_mode_note = "chosen by hand"
            else:
                state.clio_mode_note = ""
        time_state = engine.time_summary()
        state.clio_steps = time_state.get("steps", 0)
        state.clio_index = time_state.get("index", 0)
        state.clio_follow = time_state.get("follow_latest", True)
        state.clio_newer = time_state.get("newer_steps", 0)
        if self.view is not None and reason not in ("selection", "camera"):
            self.view.update()
        moved = reason in ("open_view", "fit_camera", "set_camera", "rollback")
        if moved and self.view is not None:
            self.view.push_camera()
        state.clio_bridge = self.bridge()

    def bridge(self) -> dict[str, Any] | None:
        """The key state the view reports to its host."""
        scene = self.engine.scene
        if scene is None:
            return None
        shown = [
            {
                "id": o["id"],
                "label": o.get("label"),
                "type": o.get("type"),
                "representation": o.get("display", {}).get("representation"),
                "color_by": (o.get("display", {}).get("color_by") or [None, None])[1],
            }
            for o in scene["objects"]
            if o.get("display", {}).get("visible", True)
        ]
        selection = scene.get("selection")
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
            "representation": shown,
            "camera": scene.get("camera"),
            "view_mode": scene["view"].get("mode"),
        }

    # -- the person's actions in the view -----------------------------------------
    def _ui_time(self, index: int) -> None:
        steps = len(self.engine.timesteps())
        try:
            self.engine.run(
                "set_timestep",
                {"index": int(index), "latest": int(index) == steps - 1 or None},
            )
        except CommandError:
            self._on_scene_change("rollback")

    def _ui_latest(self) -> None:
        try:
            self.engine.run("set_timestep", {"latest": True})
        except CommandError:
            self._on_scene_change("rollback")

    def _ui_toggle_mode(self) -> None:
        if self.engine.scene is None:
            return
        current = self.engine.scene["view"].get("mode", "local")
        self.engine.set_view_mode("remote" if current == "local" else "local")

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
