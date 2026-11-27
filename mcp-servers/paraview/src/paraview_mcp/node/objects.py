"""The engine's scene objects: add a filter, change its settings, remove an object,
and change how one is drawn."""

from __future__ import annotations

import difflib
from collections.abc import Callable
from typing import Any

from . import colors, data, manifest, ranges
from .errors import CommandError
from .scripts import forget_made
from .sources import set_property as _set_property

REPRESENTATIONS = (
    "Surface",
    "Surface With Edges",
    "Wireframe",
    "Points",
    "Outline",
    "Volume",
    "Slice",
    "Feature Edges",
)


class SceneObjects:
    """Mixed into the engine: create_filter, set_filter_property,
    remove_scene_object and set_representation."""

    simple: Any
    view: Any
    proxies: dict[str, Any]

    # Provided by the engine and its other parts.
    require_scene: Callable[[], dict[str, Any]]
    _make: Callable[[dict[str, Any]], Any]
    _show: Callable[..., None]
    update_view_mode: Callable[[], str]
    object_summary: Callable[..., dict[str, Any]]
    summary: Callable[..., dict[str, Any]]

    def cmd_create_filter(
        self,
        filter: str,
        input: str | None = None,
        properties: dict[str, Any] | None = None,
        label: str | None = None,
        hide_input: bool = False,
    ) -> dict[str, Any]:
        from paraview import servermanager

        scene = self.require_scene()
        names = [n for n in dir(servermanager.filters) if not n.startswith("_")]
        if filter not in names:
            close = difflib.get_close_matches(filter, names, n=6)
            raise CommandError(
                "unknown_filter",
                f"ParaView has no filter {filter!r}."
                + (f" Close matches: {', '.join(close)}." if close else ""),
            )
        parent = manifest.find_object(scene, input) if input else None
        if input and parent is None:
            raise CommandError("unknown_object", f"No scene object {input!r}.")
        if parent is None:
            parent = manifest.pipeline(scene)[-1]
        object_id = f"o{scene['next_id']}"
        scene["next_id"] += 1
        item = {
            "id": object_id,
            "kind": "filter",
            "type": filter,
            "input": parent["id"],
            "properties": properties or {},
            "label": label or f"{filter}{scene['next_id'] - 1}",
        }
        proxy = self._make(item)
        scene["objects"].append(item)
        self.proxies[object_id] = proxy
        self._show(item, proxy, apply=False)
        if hide_input:
            parent.setdefault("display", {})["visible"] = False
            self.simple.Hide(self.proxies[parent["id"]], self.view)
        self.update_view_mode()
        return {"object": self.object_summary(item), "scene": self.summary()}

    def cmd_set_filter_property(
        self, object: str, properties: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Change properties of a filter already in the scene, such as a contour value."""
        scene = self.require_scene()
        item = manifest.find_object(scene, object)
        if item is None:
            raise CommandError("unknown_object", f"No scene object {object!r}.")
        if item["kind"] != "filter":
            raise CommandError(
                "not_a_filter",
                f"{item['id']} is the dataset, not a filter; only filters have settings to change.",
            )
        if not properties:
            raise CommandError("bad_arguments", "Give at least one property to change.")
        proxy = self.proxies[item["id"]]
        kept = item.get("properties") or {}
        before = {key: kept.get(key) for key in properties}
        for key, value in properties.items():
            _set_property(proxy, item["type"], key, value)
        proxy.UpdatePipeline(self.view.ViewTime)
        item["properties"] = {**kept, **properties}
        self.update_view_mode()
        return {
            "object": self.object_summary(item),
            "before": before,
            "after": dict(properties),
            "scene": self.summary(),
        }

    def cmd_remove_scene_object(self, object: str) -> dict[str, Any]:
        scene = self.require_scene()
        item = manifest.find_object(scene, object)
        if item is None:
            raise CommandError("unknown_object", f"No scene object {object!r}.")
        if item is scene["objects"][0]:
            raise CommandError(
                "cannot_remove_dataset",
                "The dataset is the scene's root. Open another dataset instead.",
            )
        gone = manifest.descendants(scene, item["id"])
        forget_made(scene, gone)
        for object_id in reversed(gone):
            proxy = self.proxies.pop(object_id)
            self.simple.Hide(proxy, self.view)
            self.simple.Delete(proxy)
        scene["objects"] = [o for o in scene["objects"] if o["id"] not in gone]
        if scene.get("selection") and scene["selection"].get("object") in gone:
            scene["selection"] = None
        self.update_view_mode()
        return {"removed": gone, "scene": self.summary()}

    def cmd_set_representation(
        self,
        object: str,
        representation: str | None = None,
        color_by: str | None = None,
        colormap: str | None = None,
        opacity: float | None = None,
        visible: bool | None = None,
        component: int | str | None = None,
        scalar_bar: bool | None = None,
        rescale: bool | None = None,
        color_range: list[float] | None = None,
        solid_color: str | None = None,
    ) -> dict[str, Any]:
        scene = self.require_scene()
        item = manifest.find_object(scene, object)
        if item is None:
            raise CommandError("unknown_object", f"No scene object {object!r}.")
        proxy = self.proxies[item["id"]]
        display = item.setdefault("display", {})
        span = colors.checked_range(color_range, rescale)
        if color_by is not None:
            display["color_component"] = None
        if representation is not None:
            if representation not in REPRESENTATIONS:
                raise CommandError(
                    "unknown_representation",
                    f"{representation!r} is not one of: {', '.join(REPRESENTATIONS)}.",
                )
            display["representation"] = representation
        if color_by is not None:
            before = display.get("color_by")
            display["color_by"] = (
                None
                if color_by in ("", "none", "solid")
                else [data.find_association(proxy, color_by), color_by]
            )
            if display["color_by"] != before:
                display.update(color_range=None, auto_range=None, range_mode=None)
            if display["color_by"]:
                display["solid_color"] = None
        if solid_color is not None:
            display.update(solid_color=colors.parse_color(solid_color), color_by=None)
        if colormap is not None:
            display["colormap"] = colormap
            colors.share(scene, display.get("color_by"), "colormap", colormap)
        if opacity is not None:
            display["opacity"] = max(0.0, min(1.0, float(opacity)))
        if visible is not None:
            display["visible"] = bool(visible)
        if component is not None:
            display["color_component"] = data.color_component(
                proxy, display.get("color_by"), component
            )
        if scalar_bar is not None:
            display["scalar_bar"] = bool(scalar_bar)
            colors.share(scene, display.get("color_by"), "scalar_bar", bool(scalar_bar))
        self._show(item, proxy, apply=True)
        if colormap is not None and display.get("colormap"):
            # The field's other objects hold the preset ParaView applied, not the
            # name as it was asked for.
            colors.share(
                scene, display.get("color_by"), "colormap", display["colormap"]
            )
        if rescale and display.get("color_by"):
            ranges.full(self, item, proxy)
        if span is not None:
            colors.set_range(self.simple, scene, display.get("color_by"), span)
        self.update_view_mode()
        return {"object": self.object_summary(item)}
