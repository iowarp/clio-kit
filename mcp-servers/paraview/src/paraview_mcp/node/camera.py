"""The engine's camera: read it, set it, look along an axis, fit it, follow the browser."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from . import manifest
from .errors import CommandError

# A camera direction is the way the camera looks, and the view up that goes with
# it, as ParaView's "Set view direction to +X" buttons choose them.
DIRECTIONS = {
    "+x": ((1, 0, 0), (0, 0, 1)),
    "-x": ((-1, 0, 0), (0, 0, 1)),
    "+y": ((0, 1, 0), (0, 0, 1)),
    "-y": ((0, -1, 0), (0, 0, 1)),
    "+z": ((0, 0, 1), (0, 1, 0)),
    "-z": ((0, 0, -1), (0, 1, 0)),
}

#: How far the data a followed view shows may move, as a share of its size, before
#: the camera is fitted to it again.
REFRAME_SHARE = 0.02

#: Coordinates beyond this size are not data but garbage (uninitialised memory,
#: VTK's empty box at 1e299).
LARGEST = 1e30

#: The camera a view falls back to when neither the data nor the last camera can
#: place it.
HOME_CAMERA = {
    "position": [0.0, 0.0, 1.0],
    "focal_point": [0.0, 0.0, 0.0],
    "view_up": [0.0, 1.0, 0.0],
}


def _real(value: Any) -> bool:
    """Whether a coordinate is a real number of a sane size: finite, below
    ``LARGEST`` and not denormal (a freed buffer reads as values near 1e-310)."""
    try:
        v = abs(float(value))
    except (TypeError, ValueError):
        return False
    return v == 0.0 or sys.float_info.min <= v <= LARGEST


def sane_bounds(bounds: Any) -> bool:
    """Whether a box (xmin, xmax, ymin, ...) is real data: six real numbers, each
    minimum at most its maximum."""
    values = list(bounds or [])
    if len(values) != 6 or not all(_real(v) for v in values):
        return False
    return all(float(values[2 * i]) <= float(values[2 * i + 1]) for i in range(3))


def sane_camera(camera: dict[str, Any] | None) -> bool:
    """Whether a camera can draw: real coordinates, a position apart from its focal
    point and a view up of some length."""
    if not camera:
        return False
    parts = [
        list(camera.get(key) or []) for key in ("position", "focal_point", "view_up")
    ]
    if any(len(part) != 3 or not all(_real(v) for v in part) for part in parts):
        return False
    position, focal, up = ([float(v) for v in part] for part in parts)
    look = sum((f - p) ** 2 for f, p in zip(focal, position))
    return look > 0.0 and sum(v * v for v in up) > 0.0


def same_camera(a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
    """Whether two cameras are the same, up to rounding on the way to the browser."""
    if not a or not b:
        return a == b
    for key in ("position", "focal_point", "view_up"):
        left, right = list(a.get(key) or []), list(b.get(key) or [])
        if len(left) != len(right):
            return False
        scale = max([1e-300, *(abs(float(v)) for v in left + right)])
        if any(abs(float(x) - float(y)) > 1e-6 * scale for x, y in zip(left, right)):
            return False
    return bool(a.get("parallel_projection")) == bool(b.get("parallel_projection"))


def _moved(framed: list[float], shown: list[float]) -> bool:
    """Whether the box ``shown`` (xmin, xmax, ymin, ...) left the box ``framed``
    by more than ``REFRAME_SHARE`` of its size."""
    size = sum((framed[2 * i + 1] - framed[2 * i]) ** 2 for i in range(3)) ** 0.5
    return max(abs(a - b) for a, b in zip(framed, shown)) > REFRAME_SHARE * size


def _check_camera(camera: dict[str, Any]) -> None:
    """Refuse a camera that draws nothing: one at its own focal point, or one whose
    view up lies along the way it looks."""
    for key in ("position", "focal_point", "view_up"):
        values = list(camera[key])
        if len(values) != 3 or not all(_real(v) for v in values):
            raise CommandError(
                "bad_camera",
                f"The camera {key.replace('_', ' ')} needs three finite numbers "
                f"of a sane size, not {values}.",
            )
    look = [
        float(f) - float(p) for f, p in zip(camera["focal_point"], camera["position"])
    ]
    up = [float(v) for v in camera["view_up"]]
    length = sum(v * v for v in look) ** 0.5
    if length == 0:
        raise CommandError(
            "bad_camera", "The camera position is its focal point; move one of them."
        )
    up_length = sum(v * v for v in up) ** 0.5
    if up_length == 0 or abs(sum(a * b for a, b in zip(look, up))) > 0.999 * (
        length * up_length
    ):
        raise CommandError(
            "bad_camera",
            "The view up lies along the way the camera looks; pick a view up "
            "across it.",
        )


class CameraCommands:
    """Mixed into the engine: the camera commands and the browser's camera reports.

    While a view follows the latest step of a growing source and nobody has placed
    the camera, the camera is fitted again whenever the data it shows moves, such
    as a moving window or a body that travels, keeping the way it looks. A camera
    the person turns, or the agent sets or moves in a scene script, stays where it
    was put; fitting the camera (Reset camera) or looking along an axis lets it
    follow the data again. Whether it is held is saved with the scene.

    Fits count only boxes that are real data (finite, of a sane size, minimum at
    most maximum), so an object with broken cells cannot blank the view, and a
    camera that cannot draw is never kept: the last good camera comes back.
    """

    view: Any
    root: Path
    scene: dict[str, Any] | None
    proxies: dict[str, Any]
    changes: Any
    #: The scene and the box of shown data the camera was last fitted to.
    _framed: tuple[str, list[float]] | None
    #: Whether the last followed step fitted the camera again.
    reframed: bool

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

    def camera_held(self) -> bool:
        return bool(self.scene is not None and self.scene["view"].get("camera_held"))

    def hold_camera(self, held: bool) -> None:
        """Keep the camera where it is (``held``), or let it follow the data."""
        if self.scene is not None:
            self.scene["view"]["camera_held"] = bool(held)
        self._framed = None

    def shown_bounds(self) -> list[float] | None:
        """The box (xmin, xmax, ymin, ...) round the visible objects with points."""
        from paraview import servermanager

        boxes: list[list[float]] = []
        for item in (self.scene or {}).get("objects") or []:
            proxy = self.proxies.get(item["id"])
            if proxy is None:
                continue
            rep = servermanager.GetRepresentation(proxy, self.view)
            if rep is None or not rep.Visibility:
                continue
            proxy.UpdatePipeline(self.view.ViewTime)
            info = proxy.GetDataInformation()
            if int(info.GetNumberOfPoints()) == 0:
                continue
            box = [float(v) for v in info.GetBounds()]
            # An object whose box is garbage (broken cells, a freed buffer) is left
            # out, so it cannot throw the camera off the real data.
            if sane_bounds(box):
                boxes.append(box)
        if not boxes:
            return None
        return [(min if i % 2 == 0 else max)(b[i] for b in boxes) for i in range(6)]

    def fit_view(self, before: dict[str, Any] | None = None) -> None:
        """Fit the camera to the shown data, keeping the way it looks. Only real
        boxes count; a fit that still leaves a camera that cannot draw goes back
        to ``before`` (the camera before the fit) or the home camera."""
        if before is None:
            before = self.read_camera()
        bounds = self.shown_bounds()
        if bounds is not None:
            self.view.ResetCamera(*bounds)
        elif not self.shown_points():
            self.view.ResetCamera(False)
        self.keep_camera_sane(before)

    def shown_points(self) -> bool:
        """Whether a visible object has points, real box or not."""
        from paraview import servermanager

        for item in (self.scene or {}).get("objects") or []:
            proxy = self.proxies.get(item["id"])
            if proxy is None:
                continue
            rep = servermanager.GetRepresentation(proxy, self.view)
            if (
                rep is not None
                and rep.Visibility
                and int(proxy.GetDataInformation().GetNumberOfPoints()) > 0
            ):
                return True
        return False

    def keep_camera_sane(self, fallback: dict[str, Any] | None = None) -> bool:
        """If the camera cannot draw (garbage coordinates, at its own focal point),
        put back ``fallback`` when that can draw, or fit the real data, or the home
        camera. Returns whether it had to."""
        if sane_camera(self.read_camera()):
            return False
        if fallback is not None and sane_camera(fallback):
            self.apply_camera(fallback)
            return True
        self.apply_camera(HOME_CAMERA)
        bounds = self.shown_bounds()
        if bounds is not None:
            self.view.ResetCamera(*bounds)
        if not sane_camera(self.read_camera()):
            self.apply_camera(HOME_CAMERA)
        return True

    def _following_free(self) -> bool:
        scene = self.scene
        return (
            scene is not None
            and bool((scene.get("dataset") or {}).get("growing"))
            and bool(scene["time"].get("follow_latest", True))
            and not self.camera_held()
        )

    def prime_frame(self) -> None:
        """Note the shown data's box before a followed step moves, if not yet noted."""
        scene = self.scene
        if scene is None or not self._following_free():
            return
        if self._framed is None or self._framed[0] != scene["scene"]:
            bounds = self.shown_bounds()
            self._framed = None if bounds is None else (scene["scene"], bounds)

    def follow_frame(self) -> bool:
        """Fit the camera again if the shown data moved since it was last fitted.
        Returns whether it did."""
        scene = self.scene
        if scene is None or not self._following_free():
            return False
        bounds = self.shown_bounds()
        if bounds is None:
            return False
        framed = self._framed
        if framed is None or framed[0] != scene["scene"]:
            self._framed = (scene["scene"], bounds)
            return False
        if not _moved(framed[1], bounds):
            return False
        self.fit_view()
        self._framed = (scene["scene"], bounds)
        return True

    def read_camera(self) -> dict[str, Any]:
        camera = self.view.GetActiveCamera()
        return {
            "position": list(camera.GetPosition()),
            "focal_point": list(camera.GetFocalPoint()),
            "view_up": list(camera.GetViewUp()),
            "view_angle": float(camera.GetViewAngle()),
            "parallel_scale": float(camera.GetParallelScale()),
            "parallel_projection": bool(self.view.CameraParallelProjection),
        }

    def apply_camera(self, camera: dict[str, Any]) -> None:
        self.view.CameraPosition = list(camera["position"])
        self.view.CameraFocalPoint = list(camera["focal_point"])
        self.view.CameraViewUp = list(camera["view_up"])
        if camera.get("view_angle"):
            self.view.CameraViewAngle = float(camera["view_angle"])
        if camera.get("parallel_scale"):
            self.view.CameraParallelScale = float(camera["parallel_scale"])
        if camera.get("parallel_projection") is not None:
            self.view.CameraParallelProjection = int(
                bool(camera["parallel_projection"])
            )

    def browser_camera(self, camera: dict[str, Any]) -> None:
        """Take the camera the browser reports after an interaction."""
        if self.scene is None or not camera:
            return
        reported = {
            "position": camera.get("position"),
            "focal_point": camera.get("focalPoint", camera.get("focal_point")),
            "view_up": camera.get("viewUp", camera.get("view_up")),
            "view_angle": camera.get("viewAngle"),
            "parallel_scale": camera.get("parallelScale"),
            "parallel_projection": camera.get("parallelProjection"),
        }
        if not sane_camera(reported):
            # A browser that drew a broken frame can report a camera that cannot
            # draw; taking it would keep the view blank.
            return
        self.apply_camera(reported)
        self.person_moved_camera()

    def person_moved_camera(self) -> None:
        """Take the camera as the person left it in the view. If they moved it, it
        stays there instead of following the data."""
        if self.scene is None:
            return
        moved = self.read_camera()
        if not same_camera(moved, self.scene.get("camera")):
            self.changes.record("person", "camera:drag", "camera turned in the view")
            self.hold_camera(True)
        self.scene["camera"] = moved
        manifest.write(self.root, self.scene)

    def cmd_set_camera(
        self,
        position: list[float] | None = None,
        focal_point: list[float] | None = None,
        view_up: list[float] | None = None,
        azimuth: float = 0.0,
        elevation: float = 0.0,
        zoom: float = 1.0,
        direction: str | None = None,
        parallel_projection: bool | None = None,
    ) -> dict[str, Any]:
        self.require_scene()
        if direction is not None:
            self._look(direction)
        if parallel_projection is not None:
            self.view.CameraParallelProjection = int(bool(parallel_projection))
        current = self.read_camera()
        if position is not None or focal_point is not None or view_up is not None:
            wanted = {
                "position": position or current["position"],
                "focal_point": focal_point or current["focal_point"],
                "view_up": view_up or current["view_up"],
            }
            _check_camera(wanted)
            self.apply_camera(wanted)
        camera = self.view.GetActiveCamera()
        if azimuth:
            camera.Azimuth(float(azimuth))
        if elevation:
            camera.Elevation(float(elevation))
            camera.OrthogonalizeViewUp()
        if zoom and zoom != 1.0:
            camera.Dolly(float(zoom))
        placed = (
            position is not None
            or focal_point is not None
            or view_up is not None
            or bool(azimuth)
            or bool(elevation)
            or bool(zoom and zoom != 1.0)
        )
        if placed:
            self.hold_camera(True)
        elif direction is not None:
            self.hold_camera(False)
        return {"camera": self.read_camera(), "camera_follows": not self.camera_held()}

    def _look(self, direction: str) -> None:
        """Look along an axis at the centre of the scene, then fit it."""
        key = direction.strip().lower()
        if key not in DIRECTIONS:
            raise CommandError(
                "bad_direction",
                f"{direction!r} is not one of: {', '.join(DIRECTIONS)}.",
            )
        look, up = DIRECTIONS[key]
        before = self.read_camera()
        current = before if sane_camera(before) else HOME_CAMERA
        focal = current["focal_point"]
        distance = max(
            sum((p - f) ** 2 for p, f in zip(current["position"], focal)) ** 0.5, 1e-6
        )
        self.apply_camera(
            {
                "position": [f - distance * d for f, d in zip(focal, look)],
                "focal_point": focal,
                "view_up": list(up),
            }
        )
        self.fit_view(before)

    def face(self, normal: Any) -> None:
        """Look at a plane with this normal face on, then fit the scene, so a slice
        does not open edge-on to the camera."""
        n = [float(v) for v in normal]
        length = sum(v * v for v in n) ** 0.5
        if length == 0:
            return
        look = [-v / length for v in n]
        up = (0.0, 1.0, 0.0) if abs(look[2]) > 0.9 else (0.0, 0.0, 1.0)
        before = self.read_camera()
        current = before if sane_camera(before) else HOME_CAMERA
        focal = current["focal_point"]
        self.apply_camera(
            {
                "position": [f - d for f, d in zip(focal, look)],
                "focal_point": focal,
                "view_up": list(up),
            }
        )
        self.fit_view(before)

    def cmd_fit_camera(self, object: str | None = None) -> dict[str, Any]:
        scene = self.require_scene()
        if object is None:
            self.fit_view()
            self.hold_camera(False)
        else:
            item = manifest.find_object(scene, object)
            if item is None:
                raise CommandError("unknown_object", f"No scene object {object!r}.")
            bounds = [
                float(v)
                for v in self.proxies[item["id"]].GetDataInformation().GetBounds()
            ]
            if not sane_bounds(bounds):
                raise CommandError(
                    "bad_bounds",
                    f"{item.get('label') or object} has no real extent to fit "
                    f"(its box is {bounds}); its data may be empty or broken.",
                )
            before = self.read_camera()
            self.view.ResetCamera(*bounds)
            self.keep_camera_sane(before)
            self.hold_camera(True)
        return {"camera": self.read_camera(), "camera_follows": not self.camera_held()}
