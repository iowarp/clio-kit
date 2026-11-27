"""The scene engine: one ParaView render view and the scene shown in it.

Each command is applied whole. A command that changes the scene either
commits, and the manifest is written, or fails, and the scene is rebuilt
from the manifest it had before the command started.
"""

from __future__ import annotations

import copy
import difflib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from . import colors, compat, data, manifest, ranges
from .animation import SceneAnimation
from .camera import CameraCommands
from .changes import ChangeLog
from .data import DataError
from .errors import CommandError
from .frames import FrameClock, volume_grids
from .held import LEDGER
from .objects import SceneObjects
from .opening import SceneOpening
from .pool import Readers
from .reading import SceneReading
from .scripts import SceneScripts
from .sources import SourceGrowth
from .sources import set_property as _set_property

EMPTY_SCENE = "empty"

MUTATING = {
    "open_view",
    "create_filter",
    "set_filter_property",
    "remove_scene_object",
    "set_representation",
    "set_timestep",
    "set_camera",
    "fit_camera",
    "run_scene_script",
}
READING = {"inspect_selection", "measure_field", "capture", "export_artifact"}
READING |= {"export_scene", "scene_state", "export_animation", "steps"}
READING |= {"measure_over_time"}
#: Commands the node half runs a step at a time, giving its loop back between
#: steps, each with its generator: they report progress and can be cancelled.
STEPWISE = {"measure_over_time": "measuring_over_time", "export_animation": "animating"}


def _close_presets(name: str) -> str:
    from paraview import servermanager

    names = colors.preset_names(servermanager)
    close = difflib.get_close_matches(name, names, n=5, cutoff=0.3)
    return f" Close matches: {', '.join(close)}." if close else ""


class Engine(
    SceneObjects,
    CameraCommands,
    SceneReading,
    SceneScripts,
    SceneAnimation,
    SourceGrowth,
    SceneOpening,
):
    """Holds the render view, the live scene and the proxies that show it."""

    def __init__(
        self,
        root: Path,
        threshold_mb: float = 20.0,
        session: str | None = None,
        data_on_server: bool = False,
        disk: Any = None,
        readers: Readers | None = None,
    ) -> None:
        from paraview import simple

        from .datafs import LocalDisk, ServerDisk

        self.simple = simple
        # paraview.simple resets the camera on a process's first Render. The engine owns
        # the camera (a new scene fits it; a restored one applies its saved camera), so
        # that reset would undo a restored camera (FN-O3).
        simple._DisableFirstRenderCameraReset()
        compat.install()
        self.root = root
        self.threshold_mb = float(threshold_mb)
        self.session = session
        # A joined pvserver on another machine reads the data there. The engines of
        # one node half share it.
        self.disk = disk or (ServerDisk() if data_on_server else LocalDisk())
        self.data_on_server = data_on_server
        #: Readers of growing sources, shared with the other scenes of the node half.
        self.readers = readers or Readers()
        #: Each shared reader's generation this scene last moved to.
        self._read_at: dict[str, int] = {}
        self.view = simple.CreateRenderView()
        self.view.OrientationAxesVisibility = 1
        self._remote_threshold: float | None = None
        try:
            self._remote_threshold = float(self.view.RemoteRenderThreshold)
        except Exception as exc:  # noqa: BLE001 - only browser drawing on a pvserver needs it
            print(f"remote render threshold: {exc}", flush=True)
        # Each scene keeps its own time: the time keeper would move every view to
        # the step one of them shows.
        keeper = simple.GetTimeKeeper()
        try:
            keeper.Views = [v for v in keeper.Views if v != self.view]
        except Exception as exc:  # noqa: BLE001 - an older keeper only slows a view
            print(f"time keeper: {exc}", flush=True)
        simple.SetActiveView(self.view)
        #: Whether this is the node half's only scene; colour maps are process-wide.
        self.alone: Callable[[], bool] = lambda: True
        self.frames = FrameClock(self.view)
        self.scene: dict[str, Any] | None = None
        self.proxies: dict[str, Any] = {}
        self._growth: dict[str, Any] = {}
        self._settling: dict[str, tuple[Any, float]] = {}
        #: Per growing pattern, when each file still being written was last seen
        #: to change (``SourceGrowth._settled``).
        self._quiet: dict[str, dict[str, tuple[Any, float]]] = {}
        self._held: dict[str, int] = {}
        self._seen: dict[float, float] = {}
        self._seen_scene: str | None = None
        self._replay: int | None = None
        self._replay_armed: str | None = None
        self.last_growth: float | None = None
        #: When growing sources were last checked, and their colours last grown
        #: (monotonic seconds).
        self.checked_at = 0.0
        self._coloured_at = 0.0
        #: Each reader's last look at the disk, and when what it saw last changed.
        self._disk_seen: dict[str, Any] = {}
        self.disk_changed: float | None = None
        self.listeners: list[Callable[[str], None]] = []
        self.changes = ChangeLog()
        self.opening = False
        #: The scene and shown-data box a followed view's camera was last fitted
        #: to, and whether the last followed step fitted it again (camera.py).
        self._framed: tuple[str, list[float]] | None = None
        self.reframed = False

    # -- dispatch --------------------------------------------------------
    def run(self, name: str, args: dict[str, Any], by: str = "agent") -> dict[str, Any]:
        """Apply one command and return its result.

        The agent's tools and the person's actions in the view both come here;
        ``by`` says which, for the change log the view reports.
        """
        if name not in MUTATING and name not in READING:
            raise CommandError("unknown_command", f"No command named {name!r}.")
        self.activate()
        handler = getattr(self, f"cmd_{name}")
        if name in READING:
            return self._guard(handler, args)
        if self.opening:
            raise CommandError(
                "view_opening",
                "This view is still opening; try again once open_view has answered.",
            )
        before = copy.deepcopy(self.scene)
        try:
            result = self._guard(handler, args)
        except Exception:
            self._rollback(before)
            raise
        self.finish(name, args, result, before, by)
        return result

    def _guard(self, handler: Callable[..., Any], args: dict[str, Any]) -> Any:
        try:
            return handler(**args)
        except TypeError as exc:
            raise CommandError("bad_arguments", str(exc)) from exc
        except DataError as exc:
            raise CommandError(exc.code, exc.message) from exc
        except (AttributeError, ValueError, RuntimeError) as exc:
            raise CommandError("paraview_error", f"ParaView refused it: {exc}") from exc

    def series(self, name: str, args: dict[str, Any]) -> Any:
        """A ``STEPWISE`` command as steps for ``advance``; bad arguments are reported
        now, the rest at the first step."""
        self.activate()
        return self._guard(getattr(self, STEPWISE[name]), args)

    def activate(self) -> None:
        """Make this scene's view the active one: ParaView reads data at the active
        view's time, and a node half holds a view per scene."""
        if self.simple.GetActiveView() != self.view:
            self.simple.SetActiveView(self.view)

    def advance(self, steps: Any) -> tuple[tuple[int, int] | None, Any]:
        """One more step of ``series``: ``((done, total), None)`` while it goes on,
        ``(None, result)`` once it is done."""
        self.activate()
        try:
            return self._guard(lambda: next(steps), {}), None
        except StopIteration as done:
            return None, done.value

    def commit(self, reason: str) -> None:
        """Write the manifest and tell the view that the scene changed."""
        if self.scene is None:
            return
        if reason not in ("growth", "replay"):
            # Objects or the camera may have changed: the next followed step
            # measures the data's movement from what is shown then.
            self._framed = None
            self.reframed = False
        with LEDGER.phase("scene record"):
            self.keep_camera_sane(self.scene.get("camera"))
            self.scene["camera"] = self.read_camera()
            self.scene["colormaps"] = colors.snapshot(self.simple, self.scene)
            manifest.write(self.root, self.scene)
        for listener in list(self.listeners):
            listener(reason)

    def _rollback(self, before: dict[str, Any] | None) -> None:
        keep = self._kept_readers(before)
        self.scene = before
        self._clear(keep)
        if before is not None:
            self._build(before, keep)
        for listener in list(self.listeners):
            listener("rollback")

    def _kept_readers(self, before: dict[str, Any] | None) -> dict[str, Any]:
        """BP5 readers a rollback keeps open rather than re-opening. A BP5 folder a
        writer is still filling may not be safe to open again at this moment, and
        the reader already holds the steps it read; what a failed command built on
        it is rebuilt."""
        if before is None or self.scene is None:
            return {}
        fields = ("path", "reader", "reader_properties")
        current = {item["id"]: item for item in self.scene["objects"]}
        live = {id(proxy) for proxy in self.simple.GetSources().values()}
        kept: dict[str, Any] = {}
        for item in before["objects"]:
            proxy = self.proxies.get(item["id"])
            now = current.get(item["id"])
            if (
                proxy is not None
                and id(proxy) in live
                and now is not None
                and item.get("kind") == "reader"
                and now.get("kind") == "reader"
                and self._bp5(item)
                and all(item.get(f) == now.get(f) for f in fields)
            ):
                kept[item["id"]] = proxy
        return kept

    def require_scene(self) -> dict[str, Any]:
        if self.scene is None:
            raise CommandError(
                "no_scene",
                "No scene is open. Call open_view first, with a dataset path, or "
                "with no path for an empty scene.",
            )
        return self.scene

    # -- building ----------------------------------------------------------
    def _clear(self, keep: dict[str, Any] | None = None) -> None:
        keep = keep or {}
        for object_id in reversed(list(self.proxies)):
            proxy = self.proxies.pop(object_id)
            item = manifest.find_object(self.scene, object_id) if self.scene else None
            kept = keep.get(object_id) is proxy
            if kept or (item is not None and item["kind"] == "session"):
                self.simple.Hide(proxy, self.view)
                continue
            if self.readers.leave(proxy, self):
                # Another scene still reads through this shared reader.
                self.simple.Hide(proxy, self.view)
                continue
            try:
                self.simple.Hide(proxy, self.view)
                self.simple.Delete(proxy)
            except Exception as exc:  # noqa: BLE001 - a proxy already gone needs no delete
                print(f"clearing {object_id}: {exc}", flush=True)
        for state in (
            self._growth,
            self._settling,
            self._quiet,
            self._held,
            self._read_at,
        ):
            for object_id in [k for k in state if k not in keep]:
                del state[object_id]
        if not self.session:
            from paraview import servermanager

            colors.forget(servermanager, self.view, maps=self.alone())

    def _make(self, item: dict[str, Any]) -> Any:
        simple = self.simple
        if item["kind"] == "reader":
            return self._make_reader(item)
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
            ranges.fit(self, item, proxy, rep)
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
            ranges.fit(self, item, proxy, rep)
            if display.get("colormap"):
                from paraview import servermanager

                lut = simple.GetColorTransferFunction(color_by[1])
                preset = colors.resolve_preset(servermanager, display["colormap"])
                try:
                    applied = bool(preset) and lut.ApplyPreset(preset, True)
                except RuntimeError:
                    applied = False
                if not applied:
                    raise CommandError(
                        "unknown_colormap",
                        f"ParaView has no colour preset {display['colormap']!r}."
                        + _close_presets(display["colormap"]),
                    )
                display["colormap"] = preset
            if display.get("color_range"):
                # A range set by hand holds through any later change to the display.
                low, high = display["color_range"]
                colors.set_range(simple, self.scene or {}, color_by, (low, high))
        elif rep.ColorArrayName.GetAssociation() not in (None, "NONE"):
            # A reader that colours nothing (a PDB's molecule) reports no association;
            # ParaView's ColorBy(None) cannot clear that, and there is nothing to clear.
            simple.ColorBy(rep, None)
        if not color_by and display.get("solid_color"):
            rep.AmbientColor = rep.DiffuseColor = display["solid_color"]
        wanted = bool(display.get("scalar_bar")) or colors.legend_wanted(
            self.scene, color_by
        )
        rep.SetScalarBarVisibility(self.view, bool(color_by) and wanted)
        if not display.get("visible", True):
            simple.Hide(proxy, self.view)

    # -- time ------------------------------------------------------------
    def timesteps(self) -> list[float]:
        """The time steps of this scene's own objects (other scenes of the node half
        may be at other steps of the same data, or show other data)."""
        found: set[float] = set()
        for proxy in self.proxies.values():
            try:
                values = proxy.TimestepValues
            except AttributeError:
                continue
            if values is None:
                continue
            if isinstance(values, (int, float)):
                found.add(float(values))
            else:
                found.update(float(v) for v in values)
        return sorted(found)

    def _apply_time(self, scene: dict[str, Any]) -> None:
        steps = self.timesteps()
        time_state = scene["time"]
        if not steps:
            time_state["index"] = 0
            return
        if time_state.get("follow_latest", True):
            # A catch-up from the first step shows its own step while following.
            time_state["index"] = (
                len(steps) - 1 if self._replay is None else self._replay
            )
        index = max(0, min(int(time_state.get("index", 0)), len(steps) - 1))
        time_state["index"] = index
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
            "catching_up": self._replay is not None,
            "camera_follows": not bool(scene["view"].get("camera_held")),
        }

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

        The browser's renderer draws surfaces, lines and points, but not a volume nor an
        image slice (the ``Slice`` ParaView picks for a flat image grid: its image-slice
        mapper has no browser counterpart, so the view drew nothing but the colour legend).
        A visible object shown either way is drawn by ParaView. So is a growing source:
        drawn in the browser, each new step resends all of its geometry, and steps that
        land faster than that leave the view blank until they stop. ``reason`` says why the
        view is remote (``volume``, ``slice``, ``growing`` or ``geometry``) so the view and
        the agent can say so.
        """
        scene = self.scene
        if scene is None:
            return "local"
        view_state = scene["view"]
        size = self.geometry_mb()
        view_state["geometry_mb"] = round(size, 3)
        view_state["threshold_mb"] = self.threshold_mb
        shown = {
            (item.get("display") or {}).get("representation")
            for item in scene.get("objects", [])
            if (item.get("display") or {}).get("visible", True)
        }
        view_state["reason"] = (
            "volume"
            if "Volume" in shown
            else "slice"
            if "Slice" in shown
            else "growing"
            if (scene.get("dataset") or {}).get("growing")
            else "geometry"
            if size > self.threshold_mb
            else None
        )
        view_state["volume_grids"] = volume_grids(scene, self.proxies)
        if not view_state.get("pinned"):
            view_state["mode"] = "remote" if view_state["reason"] else "local"
        self._deliver_for(view_state["mode"])
        return view_state["mode"]

    def set_view_mode(self, mode: str) -> None:
        """The person's own choice of rendering mode; it sticks."""
        scene = self.require_scene()
        scene["view"].update(mode=mode, pinned=True)
        self._deliver_for(mode)
        self.commit("view_mode")

    def _deliver_for(self, mode: str) -> None:
        """With data on a joined pvserver, ParaView only sends geometry to this process
        when the view is under its remote render threshold; above it the pvserver
        draws and the actors here stay empty. The browser draws from the actors here,
        so browser mode would show only the colour legend. Browser mode lifts the
        threshold so the geometry comes here.

        ParaView mode puts it back, but a growing source the pvserver draws always:
        under the threshold every new step's geometry crossed the link for each view
        that shows it, to draw a picture here, and on a slow link that held the node
        for seconds a step. Drawn there, only the picture crosses."""
        if not self.data_on_server or self._remote_threshold is None:
            return
        growing = bool(((self.scene or {}).get("dataset") or {}).get("growing"))
        wanted = 1e12 if mode == "local" else 0.0 if growing else self._remote_threshold
        try:
            if float(self.view.RemoteRenderThreshold) != wanted:
                self.view.RemoteRenderThreshold = wanted
        except Exception as exc:  # noqa: BLE001 - the view still draws on the pvserver
            print(f"remote render threshold: {exc}", flush=True)

    # -- browser reports -----------------------------------------------------
    def pick_display(
        self, x: float, y: float, width: float, height: float
    ) -> list[float] | None:
        """World position under pixel (x, y) of a ``width`` x ``height`` view."""
        from vtkmodules.vtkRenderingCore import vtkCellPicker

        self.view.ViewSize = [max(1, int(width)), max(1, int(height))]
        # A scene the pvserver draws has no geometry here: a pick brings it over
        # for this one look, then the pvserver draws again.
        drawn_there = self.data_on_server and self._remote_threshold is not None
        before = float(self.view.RemoteRenderThreshold) if drawn_there else None
        if before is not None and before < 1e12:
            self.view.RemoteRenderThreshold = 1e12
        try:
            self.simple.Render(self.view)
            picker = vtkCellPicker()
            picker.SetTolerance(0.005)
            if not picker.Pick(float(x), float(y), 0.0, self.view.GetRenderer()):
                return None
            return [float(v) for v in picker.GetPickPosition()]
        finally:
            if before is not None and before < 1e12:
                self.view.RemoteRenderThreshold = before

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
    def _open_empty(self, new: bool) -> dict[str, Any]:
        """A scene with no dataset, for sources a scene script makes."""
        self._clear()
        saved = None if new else manifest.read(self.root, EMPTY_SCENE)
        if saved is not None:
            self.scene = saved
            self._build(saved)
            return self.summary(restored=True)
        self.scene = manifest.new_scene(EMPTY_SCENE, {})
        self._apply_time(self.scene)
        self.fit_view()
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
        self.fit_view()
        self.update_view_mode()
        return self.summary()

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
        # A chosen step ends a catch-up from the first step.
        self._replay = self._replay_armed = None
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
