"""The scene engine: one ParaView render view and the scene shown in it.

Each command is applied whole. A command that changes the scene either
commits, and the manifest is written, or fails, and the scene is rebuilt
from the manifest it had before the command started.
"""

from __future__ import annotations

import copy
import difflib
import glob
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import colors, data, manifest
from .camera import CameraCommands
from .changes import ChangeLog, describe
from .data import DataError
from .errors import CommandError, _expand
from .frames import FrameClock, volume_grids
from .reading import SceneReading

MUTATING = {
    "open_view",
    "create_filter",
    "set_filter_property",
    "remove_scene_object",
    "set_representation",
    "set_timestep",
    "set_camera",
    "fit_camera",
}
READING = {"inspect_selection", "measure_field", "capture", "export_artifact"}
READING |= {"export_scene", "scene_state"}


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


def _set_property(proxy: Any, owner: str, key: str, value: Any) -> None:
    """Set ``key`` on a filter. ``"SliceType.Normal"`` (or ``{"SliceType": {...}}``)
    reaches into the proxy a property holds, such as a Slice's plane."""
    head, _, rest = key.partition(".")
    names = proxy.ListProperties()
    if head not in names:
        close = difflib.get_close_matches(head, names, n=5)
        raise CommandError(
            "unknown_property",
            f"{owner} has no property {head!r}."
            + (f" Close matches: {', '.join(close)}." if close else "")
            + f" Its properties are: {', '.join(names)}.",
        )
    if not rest and not isinstance(value, dict):
        setattr(proxy, head, value)
        return
    inner = getattr(proxy, head)
    if not hasattr(inner, "ListProperties"):
        raise CommandError(
            "unknown_property",
            f"{owner}.{head} is a plain value, so {key!r} cannot be set; give {head} itself.",
        )
    settings = {rest: value} if rest else value
    for name, part in settings.items():
        _set_property(inner, f"{owner}.{head}", name, part)


def _close_presets(name: str) -> str:
    from paraview import servermanager

    presets = servermanager.vtkSMTransferFunctionPresets.GetInstance()
    names = [presets.GetPresetName(i) for i in range(presets.GetNumberOfPresets())]
    close = difflib.get_close_matches(name, names, n=5, cutoff=0.3)
    return f" Close matches: {', '.join(close)}." if close else ""


def _is_pattern(path: str) -> bool:
    return any(ch in path for ch in "*?[")


class Engine(CameraCommands, SceneReading):
    """Holds the render view, the live scene and the proxies that show it."""

    def __init__(
        self,
        root: Path,
        threshold_mb: float = 20.0,
        session: str | None = None,
    ) -> None:
        from paraview import simple

        self.simple = simple
        # paraview.simple resets the camera on a process's first Render. The engine owns
        # the camera (a new scene fits it; a restored one applies its saved camera), so
        # that reset would undo a restored camera (FN-O3).
        simple._DisableFirstRenderCameraReset()
        self.root = root
        self.threshold_mb = float(threshold_mb)
        self.session = session
        self.view = simple.CreateRenderView()
        self.view.OrientationAxesVisibility = 1
        simple.SetActiveView(self.view)
        self.frames = FrameClock(self.view)
        self.scene: dict[str, Any] | None = None
        self.proxies: dict[str, Any] = {}
        self._growth: dict[str, Any] = {}
        self.listeners: list[Callable[[str], None]] = []
        self.changes = ChangeLog()

    # -- dispatch --------------------------------------------------------
    def run(self, name: str, args: dict[str, Any], by: str = "agent") -> dict[str, Any]:
        """Apply one command and return its result.

        The agent's tools and the person's actions in the view both come here;
        ``by`` says which, for the change log the view reports.
        """
        if name not in MUTATING and name not in READING:
            raise CommandError("unknown_command", f"No command named {name!r}.")
        handler = getattr(self, f"cmd_{name}")
        if name in READING:
            return self._guard(handler, args)
        before = copy.deepcopy(self.scene)
        try:
            result = self._guard(handler, args)
        except Exception:
            self._rollback(before)
            raise
        if name == "open_view" and (before or {}).get("scene") != (
            self.scene or {}
        ).get("scene"):
            self.changes.clear()
        change = describe(name, args, result, self.scene)
        if change is not None:
            self.changes.record(by, *change)
        self.commit(name)
        return result

    def _guard(self, handler: Callable[..., dict[str, Any]], args: dict[str, Any]):
        try:
            return handler(**args)
        except TypeError as exc:
            raise CommandError("bad_arguments", str(exc)) from exc
        except DataError as exc:
            raise CommandError(exc.code, exc.message) from exc
        except (AttributeError, ValueError, RuntimeError) as exc:
            raise CommandError("paraview_error", f"ParaView refused it: {exc}") from exc

    def commit(self, reason: str) -> None:
        """Write the manifest and tell the view that the scene changed."""
        if self.scene is None:
            return
        self.scene["camera"] = self.read_camera()
        self.scene["colormaps"] = colors.snapshot(self.simple, self.scene)
        manifest.write(self.root, self.scene)
        for listener in list(self.listeners):
            listener(reason)

    def _rollback(self, before: dict[str, Any] | None) -> None:
        self.scene = before
        self._clear()
        if before is not None:
            self._build(before)
        for listener in list(self.listeners):
            listener("rollback")

    def require_scene(self) -> dict[str, Any]:
        if self.scene is None:
            raise CommandError(
                "no_scene",
                "No scene is open. Call open_view with a dataset path first.",
            )
        return self.scene

    # -- building ----------------------------------------------------------
    def _clear(self) -> None:
        for object_id in reversed(list(self.proxies)):
            proxy = self.proxies.pop(object_id)
            item = manifest.find_object(self.scene, object_id) if self.scene else None
            if item is not None and item["kind"] == "session":
                self.simple.Hide(proxy, self.view)
                continue
            try:
                self.simple.Hide(proxy, self.view)
                self.simple.Delete(proxy)
            except Exception as exc:  # noqa: BLE001 - a proxy already gone needs no delete
                print(f"clearing {object_id}: {exc}", flush=True)
        self._growth.clear()
        if not self.session:
            from paraview import servermanager

            colors.forget(servermanager, self.view)

    def _files(self, item: dict[str, Any]) -> list[str] | str:
        path = item["path"]
        if not _is_pattern(path):
            if not os.path.exists(path):
                raise CommandError("not_found", f"There is no file at {path}.")
            return path
        files = sorted(glob.glob(path))
        if not files:
            raise CommandError("not_found", f"No file matches {path}.")
        return files

    def _make(self, item: dict[str, Any]) -> Any:
        simple = self.simple
        if item["kind"] == "reader":
            files = self._files(item)
            proxy = simple.OpenDataFile(files)
            if proxy is None:
                raise CommandError(
                    "no_reader", f"ParaView has no reader for {item['path']}."
                )
            self._growth[item["id"]] = self._growth_key(item)
            return proxy
        if item["kind"] == "session":
            proxy = simple.FindSource(item["label"])
            if proxy is None:
                raise CommandError(
                    "session_object_gone",
                    f"{item['label']} is no longer in the ParaView session.",
                )
            return proxy
        source = self.proxies[item["input"]]
        factory = getattr(simple, item["type"])
        proxy = factory(Input=source)
        for key, value in (item.get("properties") or {}).items():
            _set_property(proxy, item["type"], key, value)
        proxy.UpdatePipeline()
        return proxy

    def _show(self, item: dict[str, Any], proxy: Any, apply: bool) -> None:
        simple = self.simple
        display = item.setdefault("display", {})
        rep = simple.Show(proxy, self.view)
        if not apply:
            display.update(
                visible=True,
                representation=str(
                    rep.SMProxy.GetProperty("Representation").GetElement(0)
                ),
                color_by=[str(v) for v in rep.ColorArrayName] or None,
                opacity=float(rep.Opacity),
                colormap=None,
                scalar_bar=False,
            )
            if display["color_by"] and not display["color_by"][1]:
                display["color_by"] = None
            return
        rep.Representation = display.get("representation") or rep.Representation
        rep.Opacity = float(display.get("opacity", 1.0))
        color_by = display.get("color_by")
        if color_by:
            component = display.get("color_component")
            simple.ColorBy(
                rep,
                (color_by[0], color_by[1])
                if component is None
                else (color_by[0], color_by[1], component),
            )
            rep.RescaleTransferFunctionToDataRange(True, False)
            if display.get("colormap"):
                lut = simple.GetColorTransferFunction(color_by[1])
                try:
                    applied = lut.ApplyPreset(display["colormap"], True)
                except RuntimeError:
                    applied = False
                if not applied:
                    raise CommandError(
                        "unknown_colormap",
                        f"ParaView has no colour preset {display['colormap']!r}."
                        + _close_presets(display["colormap"]),
                    )
        elif rep.ColorArrayName.GetAssociation() not in (None, "NONE"):
            # A reader that colours nothing (a PDB's molecule) reports no association;
            # ParaView's ColorBy(None) cannot clear that, and there is nothing to clear.
            simple.ColorBy(rep, None)
        wanted = bool(display.get("scalar_bar")) or colors.legend_wanted(
            self.scene, color_by
        )
        rep.SetScalarBarVisibility(self.view, bool(color_by) and wanted)
        if not display.get("visible", True):
            simple.Hide(proxy, self.view)

    def _build(self, scene: dict[str, Any]) -> None:
        for item in scene["objects"]:
            proxy = self._make(item)
            self.proxies[item["id"]] = proxy
            self._show(item, proxy, apply="display" in item)
        self._apply_time(scene)
        colors.restore(self.simple, scene.get("colormaps"))
        if scene.get("camera"):
            self.apply_camera(scene["camera"])
        else:
            self.view.ResetCamera(False)
        self.update_view_mode()

    # -- time ------------------------------------------------------------
    def timesteps(self) -> list[float]:
        keeper = self.simple.GetTimeKeeper()
        values = keeper.TimestepValues
        if values is None:
            return []
        if isinstance(values, (int, float)):
            return [float(values)]
        return [float(v) for v in values]

    def _apply_time(self, scene: dict[str, Any]) -> None:
        animation = self.simple.GetAnimationScene()
        animation.UpdateAnimationUsingDataTimeSteps()
        steps = self.timesteps()
        time_state = scene["time"]
        if not steps:
            time_state["index"] = 0
            return
        if time_state.get("follow_latest", True):
            time_state["index"] = len(steps) - 1
        index = max(0, min(int(time_state.get("index", 0)), len(steps) - 1))
        time_state["index"] = index
        animation.AnimationTime = steps[index]
        self.view.ViewTime = steps[index]

    def time_summary(self) -> dict[str, Any]:
        scene = self.scene
        steps = self.timesteps()
        if scene is None or not steps:
            return {"steps": len(steps)}
        index = scene["time"]["index"]
        return {
            "steps": len(steps),
            "index": index,
            "time": steps[index],
            "latest_index": len(steps) - 1,
            "latest_time": steps[-1],
            "follow_latest": scene["time"]["follow_latest"],
            "newer_steps": len(steps) - 1 - index,
        }

    def _growth_key(self, item: dict[str, Any]) -> Any:
        path = item["path"]
        if _is_pattern(path):
            return tuple(sorted(glob.glob(path)))
        try:
            stat = os.stat(path)
        except OSError:
            return None
        return (stat.st_size, stat.st_mtime_ns)

    def check_growth(self) -> dict[str, Any] | None:
        """Re-read the time list of growing sources; follow the latest step."""
        scene = self.scene
        if scene is None or not (scene.get("dataset") or {}).get("growing"):
            return None
        before = len(self.timesteps())
        changed = False
        for item in scene["objects"]:
            if item["kind"] != "reader" or item["id"] not in self.proxies:
                continue
            key = self._growth_key(item)
            if key == self._growth.get(item["id"]):
                continue
            self._growth[item["id"]] = key
            changed = True
            proxy = self.proxies[item["id"]]
            files = self._files(item)
            for name in ("FileNames", "FileName"):
                if name in proxy.ListProperties():
                    # Setting the same value is not a change; clear it first so
                    # the reader re-reads its index (a .series file, say).
                    setattr(proxy, name, [] if name == "FileNames" else "")
                    setattr(proxy, name, files)
                    break
            proxy.UpdatePipelineInformation()
        if not changed:
            return None
        self._apply_time(scene)
        after = len(self.timesteps())
        self.commit("growth")
        return {"new_steps": after - before, **self.time_summary()}

    # -- camera and view mode -----------------------------------------------
    def geometry_mb(self) -> float:
        total_kb = 0
        for item in (self.scene or {}).get("objects", []):
            proxy = self.proxies.get(item["id"])
            if proxy is not None and item.get("display", {}).get("visible", True):
                total_kb += int(proxy.GetDataInformation().GetMemorySize())
        return total_kb / 1024.0

    def update_view_mode(self) -> str:
        """Choose local or remote rendering from the geometry size and what is shown.

        The browser's renderer draws surfaces, lines and points but not a volume, so a
        visible object shown as ``Volume`` is drawn by ParaView; ``reason`` says why the
        view is remote (``volume`` or ``geometry``) so the view and the agent can say so.
        """
        scene = self.scene
        if scene is None:
            return "local"
        view_state = scene["view"]
        size = self.geometry_mb()
        view_state["geometry_mb"] = round(size, 3)
        view_state["threshold_mb"] = self.threshold_mb
        volume = any(
            (item.get("display") or {}).get("visible", True)
            and (item.get("display") or {}).get("representation") == "Volume"
            for item in scene.get("objects", [])
        )
        view_state["reason"] = (
            "volume" if volume else "geometry" if size > self.threshold_mb else None
        )
        view_state["volume_grids"] = volume_grids(scene, self.proxies)
        if not view_state.get("pinned"):
            view_state["mode"] = "remote" if view_state["reason"] else "local"
        return view_state["mode"]

    def set_view_mode(self, mode: str) -> None:
        """The person's own choice of rendering mode; it sticks."""
        scene = self.require_scene()
        scene["view"].update(mode=mode, pinned=True)
        self.commit("view_mode")

    # -- browser reports -----------------------------------------------------
    def pick_display(
        self, x: float, y: float, width: float, height: float
    ) -> list[float] | None:
        """World position under pixel (x, y) of a ``width`` x ``height`` view."""
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        self.view.ViewSize = [max(1, int(width)), max(1, int(height))]
        self.simple.Render(self.view)
        picker = vtkCellPicker()
        picker.SetTolerance(0.005)
        if not picker.Pick(float(x), float(y), 0.0, self.view.GetRenderer()):
            return None
        return [float(v) for v in picker.GetPickPosition()]

    def browser_pick(self, world: list[float]) -> dict[str, Any] | None:
        """Record what the person picked at world position ``world``."""
        if self.scene is None:
            return None
        selection = self._probe_visible(world)
        self.scene["selection"] = selection
        manifest.write(self.root, self.scene)
        for listener in list(self.listeners):
            listener("selection")
        return selection

    def _probe_visible(self, world: list[float]) -> dict[str, Any] | None:
        best = None
        for item in self.require_scene()["objects"]:
            proxy = self.proxies.get(item["id"])
            if proxy is None or not item.get("display", {}).get("visible", True):
                continue
            hit = data.probe(proxy, list(world))
            if hit is not None and (best is None or hit["distance"] < best["distance"]):
                best = {**hit, "object": item["id"], "label": item.get("label")}
        if best is not None:
            best["world_position"] = [float(v) for v in world]
            best["time"] = self.time_summary().get("time")
        return best

    # -- commands that change the scene ----------------------------------------
    def cmd_open_view(
        self,
        path: str | None = None,
        scene: str | None = None,
        growing: bool = False,
        new: bool = False,
    ) -> dict[str, Any]:
        if scene is None and path is None and self.session:
            return self._open_session()
        if scene is None and path is None:
            if self.scene is not None:
                return self.summary()
            raise CommandError(
                "no_dataset",
                "open_view needs a dataset path on this machine, or a scene id.",
            )
        if path is not None:
            path = _expand(path)
        scene_id = scene or manifest.scene_id_for(path or "")
        if self.scene is not None and self.scene["scene"] == scene_id and not new:
            return self.summary()
        saved = None if new else manifest.read(self.root, scene_id)
        if saved is None and path is None:
            raise CommandError(
                "unknown_scene",
                f"There is no scene {scene_id} here. Known scenes: "
                + (", ".join(manifest.known_scenes(self.root)) or "none"),
            )
        self._clear()
        if saved is not None:
            self.scene = saved
            self._build(saved)
            return self.summary(restored=True)
        dataset = {"path": path, "growing": bool(growing)}
        self.scene = manifest.new_scene(scene_id, dataset)
        item = {"id": "o1", "kind": "reader", "path": path, "label": None}
        self.scene["next_id"] = 2
        proxy = self._make(item)
        item["label"] = os.path.basename(path.rstrip("/")) if path else "data"
        item["type"] = proxy.GetXMLName()
        self.scene["objects"].append(item)
        self.proxies["o1"] = proxy
        self._show(item, proxy, apply=False)
        self._apply_time(self.scene)
        self.view.ResetCamera(False)
        self.update_view_mode()
        return self.summary()

    def _open_session(self) -> dict[str, Any]:
        scene_id = "session-" + str(self.session).replace(":", "-")
        self._clear()
        saved = manifest.read(self.root, scene_id)
        if saved is not None:
            self.scene = saved
            self._build(saved)
            return self.summary(restored=True)
        self.scene = manifest.new_scene(scene_id, {"session": self.session})
        for name, proxy in sorted(self.simple.GetSources().items()):
            object_id = f"o{self.scene['next_id']}"
            self.scene["next_id"] += 1
            item = {
                "id": object_id,
                "kind": "session",
                "label": name[0],
                "type": proxy.GetXMLName(),
            }
            self.scene["objects"].append(item)
            self.proxies[object_id] = proxy
            self._show(item, proxy, apply=False)
        self._apply_time(self.scene)
        self.view.ResetCamera(False)
        self.update_view_mode()
        return self.summary()

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
            parent = scene["objects"][-1]
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
    ) -> dict[str, Any]:
        scene = self.require_scene()
        item = manifest.find_object(scene, object)
        if item is None:
            raise CommandError("unknown_object", f"No scene object {object!r}.")
        proxy = self.proxies[item["id"]]
        display = item.setdefault("display", {})
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
            display["color_by"] = (
                None
                if color_by in ("", "none", "solid")
                else [data.find_association(proxy, color_by), color_by]
            )
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
        if rescale and display.get("color_by"):
            rep = self.simple.GetDisplayProperties(proxy, self.view)
            rep.RescaleTransferFunctionToDataRange(False, True)
        self.update_view_mode()
        return {"object": self.object_summary(item)}

    def cmd_set_timestep(
        self,
        index: int | None = None,
        time: float | None = None,
        latest: bool | None = None,
    ) -> dict[str, Any]:
        scene = self.require_scene()
        steps = self.timesteps()
        if not steps:
            raise CommandError("no_time", "This scene's data has no time steps.")
        if latest:
            scene["time"].update(follow_latest=True, index=len(steps) - 1)
        elif time is not None:
            nearest = min(range(len(steps)), key=lambda i: abs(steps[i] - time))
            scene["time"].update(follow_latest=False, index=nearest)
        elif index is not None:
            if not -len(steps) <= index < len(steps):
                raise CommandError(
                    "bad_index", f"There are {len(steps)} steps; index {index} is out."
                )
            index = index % len(steps)
            scene["time"].update(
                follow_latest=index == len(steps) - 1 and latest is not False,
                index=index,
            )
        else:
            raise CommandError("bad_arguments", "Give index, time or latest.")
        self._apply_time(scene)
        self.update_view_mode()
        return {"time": self.time_summary()}
