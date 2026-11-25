"""The node half's views: one trame app, render view and port per open scene.

A view follows its own scene: its variable, its time step, its catch-up and how it
is drawn. Opening another scene (another dataset, or another variable of the same
one) starts a view beside the open ones rather than replacing what they show. The
first app, the front's, takes every command and hands it to its scene's app.
"""

from __future__ import annotations

import asyncio
import re
import socket
import time
from collections.abc import Callable
from typing import Any

from . import manifest
from .app import NodeApp
from .engine import Engine
from .errors import CommandError


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", text).strip("-").lower()[:24]


def _port(value: Any) -> int:
    try:
        port = int(value)
    except (TypeError, ValueError):
        return 0
    return port if 0 < port < 65536 else 0


def _free(port: int) -> bool:
    """Whether a loopback port can be served on now."""
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


class Stage:
    """Every view of the node half, and which one the agent's commands go to."""

    def __init__(
        self,
        front: NodeApp,
        make_engine: Callable[[], Engine],
        growth_interval: float = 2.0,
    ) -> None:
        self.front = front
        self.apps = [front]
        #: The view the agent opened last: commands other than open_view go to it.
        self.current = front
        self.make_engine = make_engine
        self.growth_interval = growth_interval
        self._adding: asyncio.Lock | None = None
        self._tasks: list[Any] = []
        front.stage = self
        front.engine.alone = self.alone

    def alone(self) -> bool:
        return len(self.apps) == 1

    def scene_for(self, args: dict[str, Any]) -> str | None:
        """The scene id open_view opens, or ``None`` for the current view's."""
        if args.get("scene"):
            return str(args["scene"])
        if not args.get("path"):
            return None
        path = self.front.engine.disk.expand(str(args["path"]))
        scene_id = manifest.scene_id_for(path)
        display = (args.get("descriptor") or {}).get("display") or {}
        array = display.get("array")
        if array:
            # Each variable of a dataset is a scene of its own, in a view of its own.
            scene_id = f"{scene_id}-{_slug(str(array))}"
        return scene_id

    async def route(
        self,
        name: str,
        args: dict[str, Any],
        by: str = "agent",
        found: dict[str, Any] | None = None,
    ) -> NodeApp:
        """The app a command is for. open_view picks the view of its scene, else a
        view with no scene yet, else starts one. The agent's open_view makes it the
        current view; a view reopening itself for the person (a host replays its
        open_view) leaves the agent's commands where they were. ``found`` is told
        whether the scene was already open in a view (``already_open``).

        scene_state with a scene reads that scene's view and leaves the current one
        as it is, so the agent can look at every view without opening any."""
        if name == "scene_state" and args.get("scene"):
            return self._showing(str(args.pop("scene")))
        if name != "open_view":
            return self.current
        # A view the front reopens after its pvserver went asks for its old port.
        view_port = args.pop("view_port", None)
        scene_id = self.scene_for(args)
        if scene_id is None:
            return self.current
        if args.get("path") and not args.get("scene"):
            args["scene"] = scene_id
        app = self._find(scene_id)
        if found is not None:
            found["already_open"] = bool(
                app is not None and app.engine.scene is not None and not args.get("new")
            )
        if app is None:
            if self._adding is None:
                self._adding = asyncio.Lock()
            async with self._adding:
                app = self._find(scene_id) or await self.add(_port(view_port))
        if by == "agent" or self.current.engine.scene is None:
            self.current = app
        return app

    def _showing(self, scene_id: str) -> NodeApp:
        for app in self.apps:
            scene = app.engine.scene
            if scene is not None and scene["scene"] == scene_id:
                return app
        open_ids = ", ".join(v["scene"] for v in self.views()) or "none"
        raise CommandError(
            "scene_not_open",
            f"No open view shows scene {scene_id!r}. Open views: {open_ids}.",
        )

    def views(self) -> list[dict[str, Any]]:
        """Every open view as it was last drawn: its scene, dataset, step and mode,
        and whether the agent's commands go to it. Read from each view's last state,
        so it costs no ParaView call and never waits on a busy view."""
        out: list[dict[str, Any]] = []
        for app in self.apps:
            scene = app.engine.scene
            if scene is None:
                continue
            bridge = app.server.state.clio_bridge or {}
            out.append(
                {
                    "scene": scene["scene"],
                    "current": app is self.current,
                    "dataset": (scene.get("dataset") or {}).get("path"),
                    "color_by": [
                        p.get("color_by")
                        for p in bridge.get("representation") or []
                        if p.get("color_by")
                    ],
                    "time": bridge.get("time"),
                    "view_mode": bridge.get("view_mode"),
                }
            )
        return out

    def _find(self, scene_id: str) -> NodeApp | None:
        """The view showing a scene, else one that shows none yet."""
        for app in self.apps:
            scene = app.engine.scene
            if scene is not None and scene["scene"] == scene_id:
                return app
        for app in self.apps:
            if app.engine.scene is None:
                return app
        return None

    async def add(self, port: int = 0) -> NodeApp:
        """Start one more view, on a port of its own (``port``, else any free one),
        with the node half's token."""
        from trame.app import get_server

        front = self.front
        server = get_server(f"clio-paraview-{len(self.apps)}", client_type="vue3")
        engine = self.make_engine()
        engine.alone = self.alone
        app = NodeApp(engine, front.token, server, shared=front.shared)
        app.watched, app.pvserver = front.watched, front.pvserver
        began = time.monotonic()
        if port and not _free(port):
            print(f"port {port} is taken; this view takes another", flush=True)
            port = 0
        task = server.start(
            host="127.0.0.1",
            port=port,
            open_browser=False,
            show_connection_info=False,
            disable_logging=True,
            timeout=0,
            exec_mode="task",
        )
        self._tasks.append(task)
        await server.ready
        self.apps.append(app)
        app.name = f"view {len(self.apps)}"
        self._tasks.append(
            asyncio.ensure_future(app.watch_growth(self.growth_interval))
        )
        print(
            f"view {len(self.apps)} listening on 127.0.0.1:{server.port} "
            f"({time.monotonic() - began:.1f} s)",
            flush=True,
        )
        return app

    def idle_for(self) -> float:
        """Seconds since any view's last request, or zero while one is watched."""
        return min(app.idle_for() for app in self.apps)

    def pvserver_lost(self, reason: str) -> None:
        for app in self.apps:
            app.pvserver_lost(reason)

    @property
    def engine(self) -> Engine:
        return self.current.engine

    def engines(self) -> list[Engine]:
        return [app.engine for app in self.apps]
