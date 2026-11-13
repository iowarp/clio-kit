"""The engine's commands that read the scene and never change it."""

from __future__ import annotations

import base64
import os
import tempfile
from pathlib import Path
from typing import Any

from . import data, manifest
from .errors import CommandError, _expand


class SceneReading:
    """Mixed into the engine: selection, measurement, capture, export, state."""

    simple: Any
    view: Any
    root: Path
    scene: dict[str, Any] | None
    proxies: dict[str, Any]
    session: str | None
    changes: Any

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

    def time_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def read_camera(self) -> dict[str, Any]:
        raise NotImplementedError

    def _probe_visible(self, world: list[float]) -> dict[str, Any] | None:
        raise NotImplementedError

    def cmd_inspect_selection(
        self, position: list[float] | None = None
    ) -> dict[str, Any]:
        scene = self.require_scene()
        if position is not None:
            return {"selection": self._probe_visible(position), "source": "position"}
        picked = scene.get("selection")
        if not picked:
            return {
                "selection": None,
                "message": "Nothing is picked in the view. Click a point in the "
                "view, or give a position.",
            }
        fresh = self._probe_visible(picked["world_position"])
        return {"selection": fresh, "source": "view"}

    def cmd_measure_field(
        self,
        field: str,
        object: str | None = None,
        bins: int = 16,
        component: int | None = None,
    ) -> dict[str, Any]:
        scene = self.require_scene()
        item = manifest.find_object(scene, object) if object else None
        if object and item is None:
            raise CommandError("unknown_object", f"No scene object {object!r}.")
        if item is None:
            carrying = [
                o
                for o in scene["objects"]
                if field in self.proxies[o["id"]].PointData.keys()
                or field in self.proxies[o["id"]].CellData.keys()
            ]
            item = carrying[0] if carrying else scene["objects"][0]
        stats = data.measure(self.proxies[item["id"]], field, bins, component)
        return {
            "object": item["id"],
            "label": item.get("label"),
            "time": self.time_summary().get("time"),
            **stats,
        }

    def _render_png(self, target: Path, width: int, height: int, **extra: Any) -> bytes:
        target.parent.mkdir(parents=True, exist_ok=True)
        self.simple.SaveScreenshot(
            str(target), self.view, ImageResolution=[int(width), int(height)], **extra
        )
        return target.read_bytes()

    def cmd_capture(self, width: int = 1280, height: int = 800) -> dict[str, Any]:
        scene = self.require_scene()
        folder = manifest.scene_dir(self.root, scene["scene"]) / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(suffix=".png", dir=folder)
        os.close(handle)
        png = self._render_png(Path(name), width, height)
        return {
            "png_base64": base64.b64encode(png).decode("ascii"),
            "path": name,
            "camera": self.read_camera(),
            "time": self.time_summary().get("time"),
            "width": width,
            "height": height,
        }

    def cmd_export_artifact(
        self,
        path: str | None = None,
        width: int = 1920,
        height: int = 1080,
        transparent: bool = False,
    ) -> dict[str, Any]:
        scene = self.require_scene()
        folder = manifest.scene_dir(self.root, scene["scene"]) / "exports"
        target = Path(_expand(path)) if path else folder / "figure.png"
        if target.suffix.lower() != ".png":
            raise CommandError("bad_path", "A figure is written as a .png file.")
        extra = {"TransparentBackground": 1} if transparent else {}
        png = self._render_png(target, width, height, **extra)
        return {
            "path": str(target),
            "bytes": len(png),
            "png_base64": base64.b64encode(png).decode("ascii"),
            "width": width,
            "height": height,
        }

    def cmd_export_scene(self, path: str | None = None) -> dict[str, Any]:
        scene = self.require_scene()
        written = manifest.write(self.root, scene)
        state_path = written.with_name("scene.pvsm")
        if path:
            target = Path(_expand(path))
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(written.read_text("utf-8"), "utf-8")
            written = target
            state_path = target.with_suffix(".pvsm")
        self.simple.SaveState(str(state_path))
        return {
            "manifest_path": str(written),
            "state_path": str(state_path),
            "manifest": scene,
        }

    def cmd_scene_state(self) -> dict[str, Any]:
        if self.scene is None:
            return {
                "scene": None,
                "known_scenes": manifest.known_scenes(self.root),
                "session": self.session,
            }
        return {**self.summary(), "changes": self.changes.recent()}

    # -- summaries ------------------------------------------------------------
    def object_summary(self, item: dict[str, Any]) -> dict[str, Any]:
        proxy = self.proxies.get(item["id"])
        out = {
            "id": item["id"],
            "label": item.get("label"),
            "kind": item["kind"],
            "type": item.get("type"),
            "input": item.get("input"),
            "properties": item.get("properties") or {},
            "display": item.get("display", {}),
        }
        if item["kind"] == "reader":
            out["path"] = item["path"]
        if proxy is not None:
            out["data"] = data.describe(proxy)
        return out

    def summary(self, restored: bool = False) -> dict[str, Any]:
        scene = self.require_scene()
        return {
            "scene": scene["scene"],
            "restored": restored,
            "dataset": scene.get("dataset"),
            "session": self.session,
            "objects": [self.object_summary(o) for o in scene["objects"]],
            "time": self.time_summary(),
            "view": dict(scene["view"]),
            "camera": self.read_camera(),
            "selection": scene.get("selection"),
        }
