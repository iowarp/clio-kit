"""The person's ParaView controls in the view: their state and their actions.

The panel (``panel.py``) shows a pipeline browser, an Add filter palette, the
display and camera controls and the data's information. Every action that
changes the scene runs the same engine command the agent's tools send
(``Engine.run`` with ``by="person"``): one command, committed with its
manifest or rolled back. Nothing here changes ParaView any other way.

When any command commits, from the person or the agent, the panel's state is
rebuilt from the scene, so the agent's changes show in the panel at once.
"""

from __future__ import annotations

import asyncio
from typing import Any

from . import data, filters, manifest
from .errors import CommandError

PRESETS = (
    "Cool to Warm",
    "Viridis",
    "Inferno",
    "Plasma",
    "Magma",
    "Cividis",
    "Turbo",
    "Jet",
    "Rainbow Uniform",
    "Blue Orange (divergent)",
    "Black-Body Radiation",
    "Grayscale",
    "X Ray",
)
SHOWN_AS = ("Surface", "Surface With Edges", "Wireframe", "Points", "Outline", "Volume")
COMPONENT_NAMES = ("X", "Y", "Z")
PLAY_SECONDS = 0.4
DISPLAY_FIELDS = {
    "representation",
    "color_by",
    "component",
    "colormap",
    "opacity",
    "visible",
    "scalar_bar",
    "rescale",
}


def _short(number: float) -> str:
    return f"{number:.4g}"


def _depths(objects: list[dict[str, Any]]) -> dict[str, int]:
    depth: dict[str, int] = {}
    for item in objects:
        parent = item.get("input")
        depth[item["id"]] = depth.get(parent, -1) + 1 if parent else 0
    return depth


def _color_items(info: dict[str, Any]) -> list[dict[str, str]]:
    items = [{"value": "", "title": "Solid color"}]
    for array in info.get("arrays") or []:
        where = "point" if array["association"] == "POINTS" else "cell"
        items.append({"value": array["name"], "title": f"{array['name']} ({where})"})
    return items


def _component_items(count: int) -> list[dict[str, Any]]:
    if count <= 1:
        return []
    names = COMPONENT_NAMES if count == 3 else tuple(str(i) for i in range(count))
    return [{"value": "Magnitude", "title": "Magnitude"}] + [
        {"value": index, "title": name} for index, name in enumerate(names)
    ]


def selected_view(item: dict[str, Any], info: dict[str, Any]) -> dict[str, Any]:
    """What the Display and Info panels show for one scene object."""
    display = item.get("display") or {}
    color_by = display.get("color_by")
    field = color_by[1] if color_by else ""
    count = next(
        (a["components"] for a in info.get("arrays") or [] if a["name"] == field), 1
    )
    component = display.get("color_component")
    bounds = info.get("bounds")
    return {
        "id": item["id"],
        "label": item.get("label") or item.get("type"),
        "type": item.get("type"),
        "kind": item["kind"],
        "visible": display.get("visible", True),
        "representation": display.get("representation"),
        "color": field,
        "color_items": _color_items(info),
        "component": "Magnitude" if component is None else component,
        "component_items": _component_items(count) if field else [],
        "colormap": display.get("colormap") or ("Cool to Warm" if field else None),
        "scalar_bar": bool(display.get("scalar_bar")),
        "opacity": float(display.get("opacity", 1.0)),
        "data_type": info.get("data_type"),
        "points": f"{info.get('points', 0):,}",
        "cells": f"{info.get('cells', 0):,}",
        "memory": f"{info.get('memory_kb', 0) / 1024:.2f} MB",
        "bounds": [
            {"axis": axis, "low": _short(bounds[i]), "high": _short(bounds[i + 1])}
            for axis, i in (("X", 0), ("Y", 2), ("Z", 4))
        ]
        if bounds
        else [],
        "arrays": [
            {
                "name": a["name"],
                "where": "point" if a["association"] == "POINTS" else "cell",
                "components": a["components"],
                "range": f"{_short(a['range'][0])} to {_short(a['range'][1])}",
            }
            for a in info.get("arrays") or []
        ],
        "properties": [
            {"key": k, "value": str(v)}
            for k, v in (item.get("properties") or {}).items()
        ],
    }


class Workbench:
    """Binds the panel's state and actions to the engine."""

    def __init__(self, engine: Any, state: Any, after: Any = None) -> None:
        self.engine = engine
        self.state = state
        self.after = after or (lambda _name: None)
        self.infos: dict[str, dict[str, Any]] = {}
        self._play: asyncio.Task[None] | None = None
        state.update(
            clio_panel="",
            clio_objects=[],
            clio_active=None,
            clio_sel=None,
            clio_offered=filters.offered(None),
            clio_filter="",
            clio_form_title="",
            clio_form_fields=[],
            clio_form={},
            clio_error="",
            clio_opacity=1.0,
            clio_playing=False,
            clio_parallel=False,
            clio_presets=list(PRESETS),
            clio_shown_as=list(SHOWN_AS),
        )

    # -- the one way to change the scene -------------------------------------
    def run(self, name: str, args: dict[str, Any]) -> dict[str, Any] | None:
        """Run an engine command as the person; show a refusal in the panel."""
        try:
            result = self.engine.run(name, args, by="person")
        except CommandError as exc:
            self.state.clio_error = f"{exc.message} [{exc.code}]"
            self.refresh("rollback")
            return None
        self.state.clio_error = ""
        self.after(name)
        return result

    # -- the panel's state from the scene ---------------------------------------
    def refresh(self, reason: str) -> None:
        """Rebuild the panel from the scene after any change, whoever made it."""
        scene = self.engine.scene
        state = self.state
        if scene is None:
            state.update(clio_objects=[], clio_active=None, clio_sel=None)
            return
        objects = scene["objects"]
        self.infos = {
            o["id"]: data.describe(self.engine.proxies[o["id"]])
            for o in objects
            if o["id"] in self.engine.proxies
        }
        ids = [o["id"] for o in objects]
        active = state.clio_active
        if reason == "create_filter" and objects:
            active = objects[-1]["id"]
        elif reason == "selection" and scene.get("selection"):
            active = scene["selection"].get("object") or active
        if active not in ids:
            active = ids[-1] if ids else None
        depths = _depths(objects)
        state.clio_objects = [
            {
                "id": o["id"],
                "label": o.get("label") or o.get("type"),
                "type": o.get("type"),
                "depth": depths[o["id"]],
                "visible": (o.get("display") or {}).get("visible", True),
                "removable": o is not objects[0] and o["kind"] != "session",
            }
            for o in objects
        ]
        state.clio_parallel = bool(
            (scene.get("camera") or {}).get("parallel_projection")
        )
        self.select(active)

    def select(self, object_id: str | None) -> None:
        """Make an object the one the panels act on (the view's state only)."""
        scene = self.engine.scene
        item = manifest.find_object(scene, object_id) if scene and object_id else None
        state = self.state
        state.clio_active = item["id"] if item else None
        info = self.infos.get(item["id"]) if item else None
        state.clio_sel = selected_view(item, info) if item and info else None
        state.clio_opacity = state.clio_sel["opacity"] if state.clio_sel else 1.0
        state.clio_offered = filters.offered(info)
        if state.clio_filter and info is not None:
            self.open_form(state.clio_filter)

    # -- actions ---------------------------------------------------------------
    def set_visible(self, object_id: str, visible: bool) -> None:
        self.run("set_representation", {"object": object_id, "visible": bool(visible)})

    def remove(self, object_id: str) -> None:
        self.run("remove_scene_object", {"object": object_id})

    def display(self, field: str, value: Any) -> None:
        active = self.state.clio_active
        if not active or field not in DISPLAY_FIELDS:
            return
        if field == "color_by":
            value = value or "none"
        elif field == "opacity":
            value = float(value)
        elif field in ("visible", "scalar_bar", "rescale"):
            value = bool(value)
        self.run("set_representation", {"object": active, field: value})

    def camera(self, kind: str, value: Any = None) -> None:
        if kind == "fit":
            self.run("fit_camera", {})
        elif kind == "direction":
            self.run("set_camera", {"direction": str(value)})
        elif kind == "parallel":
            self.run("set_camera", {"parallel_projection": bool(value)})

    def open_form(self, name: str) -> None:
        state = self.state
        active = state.clio_active
        info = self.infos.get(active) if active else None
        if info is None or name not in {e["name"] for e in filters.CATALOG}:
            state.clio_filter = ""
            return
        fields = filters.form(name, info)
        state.clio_filter = name
        state.clio_form_title = f"{filters.title(name)} on {state.clio_sel['label']}"
        state.clio_form_fields = fields
        state.clio_form = {f["key"]: f["value"] for f in fields}

    def apply_form(self, values: dict[str, Any] | None) -> None:
        state = self.state
        name = state.clio_filter
        if not name or not state.clio_active:
            return
        try:
            properties = filters.properties(state.clio_form_fields, values or {})
        except filters.FormError as exc:
            state.clio_error = str(exc)
            return
        made = self.run(
            "create_filter",
            {"filter": name, "input": state.clio_active, "properties": properties},
        )
        if made is not None:
            state.clio_filter = ""

    # -- time ------------------------------------------------------------------
    def set_time(self, index: int) -> None:
        steps = len(self.engine.timesteps())
        self.run(
            "set_timestep", {"index": int(index), "latest": int(index) == steps - 1}
        )

    def step(self, delta: int) -> None:
        steps = len(self.engine.timesteps())
        if steps > 1:
            index = int(self.state.clio_index or 0) + int(delta)
            self.set_time(max(0, min(steps - 1, index)))

    def latest(self) -> None:
        self.run("set_timestep", {"latest": True})

    def play(self, on: bool) -> None:
        if not on:
            if self._play is not None:
                self._play.cancel()
            self._play = None
            self.state.clio_playing = False
            return
        if self._play is None or self._play.done():
            self._play = asyncio.ensure_future(self._playing())

    async def _playing(self) -> None:
        state = self.state
        try:
            with state:
                state.clio_playing = True
                steps = len(self.engine.timesteps())
                if int(state.clio_index or 0) >= steps - 1:
                    self.set_time(0)
            while True:
                await asyncio.sleep(PLAY_SECONDS)
                with state:
                    steps = len(self.engine.timesteps())
                    index = int(state.clio_index or 0)
                    if steps <= 1 or index >= steps - 1 or state.clio_error:
                        break
                    self.set_time(index + 1)
        finally:
            with state:
                state.clio_playing = False
            self._play = None
