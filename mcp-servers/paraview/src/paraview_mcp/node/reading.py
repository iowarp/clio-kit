"""The engine's commands that read the scene and never change it."""

from __future__ import annotations

import base64
import os
import tempfile
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

from . import data, manifest
from .errors import CommandError, _expand

#: The most steps one measure_over_time returns; a longer run takes a stride.
MAX_SERIES = 1000


class SceneReading:
    """Mixed into the engine: selection, measurement, capture, export, state."""

    simple: Any
    view: Any
    root: Path
    scene: dict[str, Any] | None
    proxies: dict[str, Any]
    session: str | None
    changes: Any
    frames: Any

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

    def time_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def read_camera(self) -> dict[str, Any]:
        raise NotImplementedError

    def _probe_visible(self, world: list[float]) -> dict[str, Any] | None:
        raise NotImplementedError

    def timesteps(self) -> list[float]:
        raise NotImplementedError

    def _apply_time(self, scene: dict[str, Any]) -> None:
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

    def _carrier(self, field: str, object: str | None) -> dict[str, Any]:
        """The scene object to measure: ``object``, else the first that has ``field``."""
        scene = self.require_scene()
        item = manifest.find_object(scene, object) if object else None
        if object and item is None:
            raise CommandError("unknown_object", f"No scene object {object!r}.")
        if item is None:
            carrying = [
                o
                for o in manifest.pipeline(scene)
                if field in self.proxies[o["id"]].PointData.keys()
                or field in self.proxies[o["id"]].CellData.keys()
            ]
            item = carrying[0] if carrying else scene["objects"][0]
        return item

    def cmd_measure_field(
        self,
        field: str,
        object: str | None = None,
        bins: int = 16,
        component: int | None = None,
    ) -> dict[str, Any]:
        item = self._carrier(field, object)
        stats = data.measure(self.proxies[item["id"]], field, bins, component)
        return {
            "object": item["id"],
            "label": item.get("label"),
            "time": self.time_summary().get("time"),
            **stats,
        }

    def cmd_measure_over_time(
        self,
        field: str,
        object: str | None = None,
        component: int | None = None,
        start: int | None = None,
        end: int | None = None,
        stride: int = 1,
    ) -> dict[str, Any]:
        """Statistics of ``field`` at each time step from ``start`` to ``end``."""
        steps = self.measuring_over_time(field, object, component, start, end, stride)
        while True:
            try:
                next(steps)
            except StopIteration as done:
                return done.value

    def measuring_over_time(
        self,
        field: str,
        object: str | None = None,
        component: int | None = None,
        start: int | None = None,
        end: int | None = None,
        stride: int = 1,
    ) -> Generator[tuple[int, int], None, dict[str, Any]]:
        """measure_over_time one step at a time: yields how many steps are measured
        and how many there are, so the node half can go on answering and following
        between steps, and returns the series. Each step is measured at its own time;
        the view stays on its step throughout."""
        self.require_scene()
        item = self._carrier(field, object)
        proxy = self.proxies[item["id"]]
        steps: list[float | None] = list(self.timesteps()) or [None]
        last = len(steps) - 1
        first = max(0, min(int(start or 0), last))
        final = last if end is None else max(first, min(int(end), last))
        picked = list(range(first, final + 1, max(1, int(stride))))[-MAX_SERIES:]
        series = []
        for done, index in enumerate(picked, 1):
            stats = data.measure(
                proxy, field, 1, component, time=steps[index], histogram=False
            )
            stats.pop("histogram", None)
            series.append({"index": index, "time": steps[index], **stats})
            yield done, len(picked)
        head = series[0] if series else {}
        return {
            "object": item["id"],
            "label": item.get("label"),
            "field": field,
            "association": head.get("association"),
            "component": head.get("component"),
            "steps": len(steps) if steps != [None] else 0,
            "series": series,
        }

    def _render_png(self, target: Path, width: int, height: int, **extra: Any) -> bytes:
        target.parent.mkdir(parents=True, exist_ok=True)
        self.simple.SaveScreenshot(
            str(target), self.view, ImageResolution=[int(width), int(height)], **extra
        )
        return target.read_bytes()

    def drawn(
        self, width: int = 320, height: int = 200, picture: Path | None = None
    ) -> dict[str, Any]:
        """How much of a frame of this size the scene really draws (drawn.py)."""
        from paraview import servermanager

        from . import drawn

        scene = self.require_scene()
        reps = [
            rep
            for rep in (
                servermanager.GetRepresentation(proxy, self.view)
                for proxy in self.proxies.values()
            )
            if rep is not None
        ]
        folder = manifest.scene_dir(self.root, scene["scene"]) / "captures"
        return drawn.measure(
            lambda path: self.simple.SaveScreenshot(
                str(path), self.view, ImageResolution=[int(width), int(height)]
            ),
            reps,
            folder,
            picture,
        )

    def cmd_capture(self, width: int = 1280, height: int = 800) -> dict[str, Any]:
        scene = self.require_scene()
        folder = manifest.scene_dir(self.root, scene["scene"]) / "captures"
        folder.mkdir(parents=True, exist_ok=True)
        handle, name = tempfile.mkstemp(suffix=".png", dir=folder)
        os.close(handle)
        started = time.perf_counter()
        png = self._render_png(Path(name), width, height)
        seconds = round(time.perf_counter() - started, 2)
        return {
            "png_base64": base64.b64encode(png).decode("ascii"),
            "path": name,
            "camera": self.read_camera(),
            "time": self.time_summary().get("time"),
            "seconds": seconds,
            "view": dict(scene["view"]),
            "width": width,
            "height": height,
            "drawn": self.drawn(width, height, Path(name)),
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
            "by": item.get("by"),
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
            "objects": [self.object_summary(o) for o in manifest.pipeline(scene)],
            "scripts": [
                {
                    "id": o["id"],
                    "label": o.get("label"),
                    "lines": len(str(o.get("code") or "").splitlines()),
                    "made": [
                        m
                        for m in o.get("made") or []
                        if m not in (o.get("removed") or [])
                    ],
                }
                for o in scene["objects"]
                if o.get("kind") == "script"
            ],
            "time": self.time_summary(),
            "view": {**scene["view"], "frame_seconds": self.frames.seconds},
            "camera": self.read_camera(),
            "selection": scene.get("selection"),
        }
