"""The engine's commands that read the scene and never change it."""

from __future__ import annotations

import base64
import math
import os
import tempfile
import time
from collections.abc import Generator
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
        # Only pipeline objects carry data: a script step may share its label with
        # what it made.
        item = (
            next(
                (
                    o
                    for o in manifest.pipeline(scene)
                    if object in (o["id"], o.get("label"))
                ),
                None,
            )
            if object
            else None
        )
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
        # Every picked step, however many: a series is never cut short.
        picked = list(range(first, final + 1, max(1, int(stride))))
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

    def screenshot(
        self, target: Path | str, width: int, height: int, **extra: Any
    ) -> None:
        """Save one frame of exactly ``width`` x ``height``, rendered once.

        SaveScreenshot never enlarges a view: when the target is bigger than the
        view a browser last sized, it renders the picture in tiles, one render
        each, and a viewer a few pixels wide turns one frame into thousands of
        renders (over a remote pvserver, one round trip each). The offscreen view
        takes the target size for this frame and gets its own size back after.
        """
        size = [max(1, int(width)), max(1, int(height))]
        before = [int(v) for v in self.view.ViewSize]
        self.view.ViewSize = size
        try:
            self.simple.SaveScreenshot(
                str(target), self.view, ImageResolution=size, **extra
            )
        finally:
            self.view.ViewSize = before

    def _render_png(self, target: Path, width: int, height: int, **extra: Any) -> bytes:
        target.parent.mkdir(parents=True, exist_ok=True)
        self.screenshot(target, width, height, **extra)
        return target.read_bytes()

    def drawn(
        self,
        width: int = 320,
        height: int = 200,
        picture: Path | None = None,
        **extra: Any,
    ) -> dict[str, Any]:
        """How much of a frame of this size the scene really draws (drawn.py), and
        when it draws nothing, why."""
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
        result = drawn.measure(
            lambda path: self.screenshot(path, width, height, **extra),
            reps,
            folder,
            picture,
        )
        if result.get("blank") is True and not result.get("note"):
            try:
                result["note"] = self.blank_why()
            except Exception as exc:  # noqa: BLE001 - the blank verdict stands
                result["note"] = f"the cause was not found ({type(exc).__name__})"
        return result

    def blank_why(self) -> str:
        """Why the visible objects draw no pixel: an empty output at this time, a
        transparent display, or a camera that is inside the data or does not face it."""
        from paraview import servermanager

        scene = self.require_scene()
        empty: list[str] = []
        clear: list[str] = []
        boxes: list[list[float]] = []
        for item in scene.get("objects") or []:
            proxy = self.proxies.get(item["id"])
            rep = (
                servermanager.GetRepresentation(proxy, self.view)
                if proxy is not None
                else None
            )
            if rep is None or not rep.Visibility:
                continue
            name = str(item.get("label") or item["id"])
            info = proxy.GetDataInformation()
            if int(info.GetNumberOfPoints()) == 0:
                empty.append(name)
                continue
            if "Opacity" in rep.ListProperties() and float(rep.Opacity) <= 0.0:
                clear.append(name)
                continue
            boxes.append([float(v) for v in info.GetBounds()])
        reasons = []
        if empty:
            reasons.append(f"{', '.join(empty)} has no points at this time")
        if clear:
            reasons.append(f"{', '.join(clear)} is fully transparent")
        if boxes:
            reasons.append(self._camera_misses(boxes))
        return "; ".join(r for r in reasons if r) or (
            "the visible objects have points in front of the camera, yet no pixel "
            "differs from the background: check their colour against the background "
            "and whether ParaView is still running"
        )

    def _camera_misses(self, boxes: list[list[float]]) -> str:
        """How the camera misses the data in ``boxes`` (xmin, xmax, ymin, ...), if it does."""
        low = [min(b[2 * i] for b in boxes) for i in range(3)]
        high = [max(b[2 * i + 1] for b in boxes) for i in range(3)]
        centre = [(a + b) / 2 for a, b in zip(low, high)]
        radius = max(sum((b - a) ** 2 for a, b in zip(low, high)) ** 0.5 / 2, 1e-12)
        camera = self.read_camera()
        position = camera["position"]
        look = [f - p for f, p in zip(camera["focal_point"], position)]
        length = sum(v * v for v in look) ** 0.5
        if length == 0:
            return "the camera's position is its focal point, so it looks nowhere"
        look = [v / length for v in look]
        up = camera["view_up"]
        if abs(sum(a * b for a, b in zip(look, up))) > 0.999 * (
            sum(v * v for v in up) ** 0.5 or 1.0
        ):
            return "the camera's view up is along its view direction"
        if all(a <= p <= b for a, p, b in zip(low, position, high)):
            return "the camera sits inside the data's bounds"
        to_centre = [c - p for c, p in zip(centre, position)]
        distance = sum(v * v for v in to_centre) ** 0.5
        if camera["parallel_projection"]:
            across = [
                t - sum(a * b for a, b in zip(to_centre, look)) * l
                for t, l in zip(to_centre, look)
            ]
            off = sum(v * v for v in across) ** 0.5
            if off > radius + float(camera["parallel_scale"]) * 2:
                return "the data lies outside the camera's parallel view"
            return ""
        ahead = sum(a * b for a, b in zip(to_centre, look))
        if ahead < -radius:
            return "the data is behind the camera"
        angle = math.degrees(math.acos(max(-1.0, min(1.0, ahead / distance))))
        spread = math.degrees(math.asin(min(1.0, radius / distance)))
        if angle - spread > float(camera["view_angle"]):
            return "the data lies outside the camera's field of view"
        return ""

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
            "drawn": self.drawn(width, height, target, **extra),
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
