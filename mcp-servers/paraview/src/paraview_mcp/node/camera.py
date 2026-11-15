"""The engine's camera: read it, set it, look along an axis, fit it, follow the browser."""

from __future__ import annotations

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


class CameraCommands:
    """Mixed into the engine: the camera commands and the browser's camera reports."""

    view: Any
    root: Path
    scene: dict[str, Any] | None
    proxies: dict[str, Any]
    changes: Any

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

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
        self.apply_camera(
            {
                "position": camera["position"],
                "focal_point": camera.get("focalPoint", camera.get("focal_point")),
                "view_up": camera.get("viewUp", camera.get("view_up")),
                "view_angle": camera.get("viewAngle"),
                "parallel_scale": camera.get("parallelScale"),
                "parallel_projection": camera.get("parallelProjection"),
            }
        )
        moved = self.read_camera()
        if moved != self.scene.get("camera"):
            self.changes.record("person", "camera:drag", "camera turned in the view")
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
            self.apply_camera(
                {
                    "position": position or current["position"],
                    "focal_point": focal_point or current["focal_point"],
                    "view_up": view_up or current["view_up"],
                }
            )
        camera = self.view.GetActiveCamera()
        if azimuth:
            camera.Azimuth(float(azimuth))
        if elevation:
            camera.Elevation(float(elevation))
            camera.OrthogonalizeViewUp()
        if zoom and zoom != 1.0:
            camera.Dolly(float(zoom))
        return {"camera": self.read_camera()}

    def _look(self, direction: str) -> None:
        """Look along an axis at the centre of the scene, then fit it."""
        key = direction.strip().lower()
        if key not in DIRECTIONS:
            raise CommandError(
                "bad_direction",
                f"{direction!r} is not one of: {', '.join(DIRECTIONS)}.",
            )
        look, up = DIRECTIONS[key]
        current = self.read_camera()
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
        self.view.ResetCamera(False)

    def cmd_fit_camera(self, object: str | None = None) -> dict[str, Any]:
        scene = self.require_scene()
        if object is None:
            self.view.ResetCamera(False)
        else:
            item = manifest.find_object(scene, object)
            if item is None:
                raise CommandError("unknown_object", f"No scene object {object!r}.")
            bounds = self.proxies[item["id"]].GetDataInformation().GetBounds()
            self.view.ResetCamera(*bounds)
        return {"camera": self.read_camera()}
