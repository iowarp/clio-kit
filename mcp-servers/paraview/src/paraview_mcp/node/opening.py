"""Opening a view a stage at a time.

Opening a view makes its reader, shows it, then builds what the descriptor's
display asks for: a slice, an outline, a colour map. With the data on a joined
pvserver over a slow link each stage takes seconds, and every view of a node
half runs on one loop, so ``staged`` gives the loop back between stages: the
other views keep following their sources while one opens. ``cmd_open_view``
runs the same stages at once.
"""

from __future__ import annotations

import copy
import os
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any, TypeVar

from . import colors, manifest
from .camera import sane_camera
from .changes import describe
from .data import DataError
from .datafs import LocalDisk
from .errors import CommandError
from .sources import preset_name

T = TypeVar("T")
#: What a stage says it did, and what the whole open returns.
Stages = Generator[str, None, dict[str, Any]]


def drain(steps: Generator[str, None, T]) -> T:
    """Run every stage of ``steps`` at once; what it returns."""
    while True:
        try:
            next(steps)
        except StopIteration as done:
            return done.value  # type: ignore[no-any-return]


class SceneOpening:
    """Mixed into the engine: open_view, and rebuilding a saved scene."""

    simple: Any
    view: Any
    root: Path
    session: str | None
    scene: dict[str, Any] | None
    proxies: dict[str, Any]
    disk: LocalDisk
    changes: Any
    #: Whether a staged open is between its stages: nothing else may change the
    #: scene until it ends.
    opening: bool

    # Provided by the engine and its other parts.
    activate: Callable[[], None]
    commit: Callable[[str], None]
    arm_replay: Callable[[], None]
    summary: Callable[..., dict[str, Any]]
    _guard: Callable[..., Any]
    _rollback: Callable[[dict[str, Any] | None], None]
    _clear: Callable[..., None]
    _make: Callable[[dict[str, Any]], Any]
    _show: Callable[..., None]
    _apply_time: Callable[[dict[str, Any]], None]
    _update_dataset: Callable[..., None]
    _open_empty: Callable[[bool], dict[str, Any]]
    _open_session: Callable[[], dict[str, Any]]
    _files: Callable[[dict[str, Any]], Any]
    replay_script: Callable[[dict[str, Any]], dict[str, Any]]
    apply_camera: Callable[[Any], None]
    fit_view: Callable[[], None]
    update_view_mode: Callable[[], str]
    cmd_create_filter: Callable[..., dict[str, Any]]
    cmd_set_representation: Callable[..., dict[str, Any]]
    face: Callable[[Any], None]

    # -- the command -----------------------------------------------------------
    def cmd_open_view(self, **args: Any) -> dict[str, Any]:
        return drain(self._open_steps(**args))

    def staged(self, args: dict[str, Any], by: str = "agent") -> Stages:
        """open_view a stage at a time, as ``run`` applies it: whole, or rolled back
        to the scene before it when a stage fails or the open is closed early."""
        self.activate()
        before = copy.deepcopy(self.scene)
        steps = self._guard(self._open_steps, args)
        self.opening = True
        try:
            while True:
                self.activate()
                try:
                    label = self._guard(lambda: next(steps), {})
                except StopIteration as done:
                    result: dict[str, Any] = done.value
                    break
                yield label
        except BaseException:
            self.opening = False
            steps.close()
            self._rollback(before)
            raise
        self.opening = False
        self.finish("open_view", args, result, before, by)
        return result

    def finish(
        self,
        name: str,
        args: dict[str, Any],
        result: Any,
        before: dict[str, Any] | None,
        by: str,
    ) -> None:
        """What a command that changed the scene leaves: the change log, and the
        manifest written."""
        if name == "open_view" and (before or {}).get("scene") != (
            self.scene or {}
        ).get("scene"):
            self.changes.clear()
            self.arm_replay()
        change = describe(name, args, result, self.scene)
        if change is not None:
            self.changes.record(by, *change)
        self.commit(name)

    # -- the stages ------------------------------------------------------------
    def _open_steps(
        self,
        path: str | None = None,
        scene: str | None = None,
        growing: bool = False,
        new: bool = False,
        format: str | None = None,
        reader: str | None = None,
        reader_properties: dict[str, Any] | None = None,
        descriptor: dict[str, Any] | None = None,
    ) -> Stages:
        if scene is None and path is None and self.session:
            return self._open_session()
        if scene is None and path is None:
            if self.scene is not None and not new:
                return self.summary()
            return self._open_empty(new)
        if path is not None:
            path = self.disk.expand(path)
        scene_id = scene or manifest.scene_id_for(path or "")
        if self.scene is not None and self.scene["scene"] == scene_id and not new:
            self._update_dataset(self.scene, growing, descriptor)
            return self.summary()
        saved = None if new else manifest.read(self.root, scene_id)
        if saved is not None and (reader or reader_properties):
            # A saved scene read its data another way; the descriptor's reader wins.
            known = saved.get("dataset") or {}
            if (known.get("reader"), known.get("reader_properties")) != (
                reader,
                reader_properties or None,
            ):
                saved = None
        if saved is None and path is None:
            raise CommandError(
                "unknown_scene",
                f"There is no scene {scene_id} here. Known scenes: "
                + (", ".join(manifest.known_scenes(self.root)) or "none"),
            )
        self._clear()
        if saved is not None:
            self._update_dataset(saved, growing, descriptor)
            self.scene = saved
            yield from self._build_steps(saved)
            # A newer descriptor may name geometry the saved scene does not have.
            yield from self._geometry_steps(descriptor)
            return self.summary(restored=True)
        dataset: dict[str, Any] = {"path": path, "growing": bool(growing)}
        reading: dict[str, Any] = {
            "format": format,
            "reader": reader,
            "reader_properties": reader_properties or None,
        }
        dataset.update({k: v for k, v in reading.items() if v})
        if descriptor:
            dataset["descriptor"] = descriptor
        self.scene = manifest.new_scene(scene_id, dataset)
        item: dict[str, Any] = {
            "id": "o1",
            "kind": "reader",
            "path": path,
            "label": None,
        }
        item.update({k: v for k, v in reading.items() if v and k != "format"})
        self.scene["next_id"] = 2
        proxy = self._make(item)
        item["label"] = os.path.basename(path.rstrip("/")) if path else "data"
        item["type"] = proxy.GetXMLName()
        self.scene["objects"].append(item)
        self.proxies["o1"] = proxy
        yield "reader"
        self._show(item, proxy, apply=False)
        self._apply_time(self.scene)
        self.fit_view()
        yield "shown"
        yield from self._first_view_steps((descriptor or {}).get("display"))
        yield from self._geometry_steps(descriptor)
        self.update_view_mode()
        return self.summary()

    def _build(self, scene: dict[str, Any]) -> None:
        drain(self._build_steps(scene))

    def _build_steps(self, scene: dict[str, Any]) -> Generator[str, None, None]:
        """Rebuild a saved scene, an object at a time."""
        made: dict[str, Any] = {}
        for item in list(scene["objects"]):
            if item.get("kind") == "script":
                made.update(self.replay_script(item))
                yield f"script {item['id']}"
                continue
            if item.get("kind") == "geometry":
                proxy = self._geometry_or_note(item)
                if proxy is None:
                    scene["objects"].remove(item)
                    continue
            elif item.get("kind") == "made":
                proxy = made.pop(item["id"])
            else:
                proxy = self._make(item)
            self.proxies[item["id"]] = proxy
            self._show(item, proxy, apply="display" in item)
            yield f"object {item['id']}"
        self._apply_time(scene)
        colors.restore(self.simple, scene.get("colormaps"))
        if sane_camera(scene.get("camera")):
            self.apply_camera(scene["camera"])
        else:
            self.fit_view()
        self.update_view_mode()

    def _first_view(self, display: dict[str, Any] | None) -> None:
        drain(self._first_view_steps(display))

    def _first_view_steps(
        self, display: dict[str, Any] | None
    ) -> Generator[str, None, None]:
        """Show what the descriptor's display asks for: a slice or contour of the data,
        coloured by an array over a fixed range. A viewer may ignore it, so each part
        that cannot be shown is left out and noted with the dataset; the open stands."""
        if not display or self.scene is None:
            return
        array = display.get("array")
        kind = display.get("representation")
        notes: list[str] = []

        def attempt(step: Callable[[], Any]) -> Any:
            try:
                return step()
            except (CommandError, DataError) as exc:
                notes.append(str(exc))
                return None

        target = "o1"
        made = None
        if kind == "Slice":
            normal = display.get("slice_normal")
            made = attempt(
                lambda: self.cmd_create_filter(
                    "Slice",
                    "o1",
                    {"SliceType.Normal": normal} if normal else None,
                    label="slice",
                )
            )
            yield "slice"
        elif kind == "Contour" and array and display.get("contour_values"):
            contour = {
                "ContourBy": [display.get("association") or "POINTS", array],
                "Isosurfaces": display["contour_values"],
                "ComputeScalars": True,
            }
            made = attempt(
                lambda: self.cmd_create_filter(
                    "Contour", "o1", contour, label="contour"
                )
            )
            yield "contour"
        elif kind in ("Outline", "Surface", "Volume"):
            attempt(lambda: self.cmd_set_representation("o1", representation=kind))
            yield "representation"
        if made:
            target = made["object"]["id"]
            attempt(lambda: self.cmd_set_representation("o1", representation="Outline"))
            if kind == "Slice":
                attempt(lambda: self.face(self.proxies[target].SliceType.Normal))
            yield "outline"
        if array:
            colour: dict[str, Any] = {"color_by": array, "scalar_bar": True}
            if display.get("range"):
                colour["color_range"] = display["range"]
            coloured = attempt(lambda: self.cmd_set_representation(target, **colour))
            yield "colour"
            if coloured:
                preset = display.get("color_map")
                name = preset_name(preset) if preset else None
                if preset and name is None:
                    notes.append(f"ParaView has no colour preset {preset!r}.")
                elif name:
                    attempt(lambda: self.cmd_set_representation(target, colormap=name))
                    yield "colour map"
        if notes:
            self.scene["dataset"]["display_note"] = (
                "The descriptor's display was shown in part: " + " ".join(notes)
            )

    def _geometry_steps(
        self, descriptor: dict[str, Any] | None
    ) -> Generator[str, None, None]:
        """Open the descriptor's geometry: surfaces in the data's coordinates, such as
        the body a solver voxelises, each a scene object a script can show, hide or
        colour beside the data. One that cannot be opened is left out and noted."""
        scene = self.scene
        wanted = (descriptor or {}).get("geometry") or []
        if scene is None or not wanted:
            return
        # A geometry the person removed stays removed when the scene is reopened.
        offered = scene.setdefault("dataset", {}).setdefault("geometry_offered", [])
        have = {o.get("path") for o in scene["objects"] if o.get("kind") == "geometry"}
        have.update(offered)
        for part in wanted:
            path = self.disk.expand(part["path"])
            if path in have:
                continue
            item: dict[str, Any] = {
                "id": f"o{scene['next_id']}",
                "kind": "geometry",
                "path": path,
                "label": part.get("label") or os.path.basename(path),
            }
            proxy = self._geometry_or_note(item)
            if proxy is None:
                continue
            scene["next_id"] += 1
            item["type"] = proxy.GetXMLName()
            scene["objects"].append(item)
            self.proxies[item["id"]] = proxy
            self._show(item, proxy, apply=False)
            if not part.get("visible", True):
                item["display"]["visible"] = False
                self.simple.Hide(proxy, self.view)
            have.add(path)
            offered.append(path)
            yield f"geometry {item['id']}"

    def _geometry_or_note(self, item: dict[str, Any]) -> Any:
        """A reader of one geometry file, or None with the reason kept in the
        dataset's geometry_note: missing geometry leaves the data's view whole."""
        try:
            files = self._files(item)
            proxy = self.simple.OpenDataFile(files)
            if proxy is None:
                raise CommandError(
                    "no_reader", f"ParaView has no reader for {item['path']}."
                )
            proxy.UpdatePipeline()
            return proxy
        except (CommandError, DataError) as exc:
            if self.scene is not None:
                dataset = self.scene.setdefault("dataset", {})
                kept = dataset.get("geometry_note")
                dataset["geometry_note"] = (
                    kept or "Geometry left out of the scene:"
                ) + f" {item['label']}: {exc}"
            return None
