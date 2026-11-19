"""Readers, and sources that are still being written.

A reader comes from a view descriptor: the path or path pattern, the format,
the ParaView reader and that reader's properties, and whether the source is
still being written. Without a named reader, ParaView picks one from the
file name.

For a growing source the engine watches what is on disk: the files a pattern
matches, a file's size and time, or the entries of a folder (an ADIOS2 BP
output is a folder). When that changes it re-reads the reader's time list. A
view following the latest step moves to it; a view parked on a step stays
there and reports how many newer steps exist. ``steps`` lists the steps with
when each was first seen, so a caller can report each new one.
"""

from __future__ import annotations

import difflib
import glob
import os
import time
from collections.abc import Callable
from typing import Any

from . import bp5
from .errors import CommandError

#: Seconds a grown BP5 folder must stay unchanged before it is read, when its data
#: may still be short of the last step.
SETTLE = 1.0


def set_property(proxy: Any, owner: str, key: str, value: Any) -> None:
    """Set ``key`` on a proxy. ``"SliceType.Normal"`` (or ``{"SliceType": {...}}``)
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
        set_property(inner, f"{owner}.{head}", name, part)


def is_pattern(path: str) -> bool:
    return any(ch in path for ch in "*?[")


def preset_name(name: str) -> str | None:
    """ParaView's name for colour preset ``name``: the name itself, or the same
    name without a qualifier such as " (matplotlib)", which ParaView 6 drops
    ("Viridis (matplotlib)" is "Viridis"), in any letter case."""
    from paraview import servermanager

    presets = servermanager.vtkSMTransferFunctionPresets.GetInstance()
    names = [presets.GetPresetName(i) for i in range(presets.GetNumberOfPresets())]
    bare = name.split(" (", 1)[0].strip()
    for wanted in (name, bare):
        for known in names:
            if known.lower() == wanted.lower():
                return known
    return None


def _stat(path: str) -> tuple[int, int] | None:
    try:
        stat = os.stat(path)
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def disk_key(path: str) -> Any:
    """What is on disk for ``path``, to tell when a source has grown: each matched
    file (or each entry of a folder) with its size and modification time."""
    if is_pattern(path):
        return tuple((name, _stat(name)) for name in sorted(glob.glob(path)))
    if os.path.isdir(path):
        try:
            names = sorted(os.listdir(path))
        except OSError:
            return None
        return tuple((name, _stat(os.path.join(path, name))) for name in names)
    return _stat(path)


class SourceGrowth:
    """Mixed into the engine: making readers and following a growing source."""

    simple: Any
    view: Any
    scene: dict[str, Any] | None
    proxies: dict[str, Any]
    _growth: dict[str, Any]
    _settling: dict[str, tuple[Any, float]]
    _seen: dict[float, float]
    _seen_scene: str | None
    last_growth: float | None

    # Provided by the engine.
    def require_scene(self) -> dict[str, Any]:
        raise NotImplementedError

    def timesteps(self) -> list[float]:
        raise NotImplementedError

    def time_summary(self) -> dict[str, Any]:
        raise NotImplementedError

    def _apply_time(self, scene: dict[str, Any]) -> None:
        raise NotImplementedError

    def commit(self, reason: str) -> None:
        raise NotImplementedError

    cmd_create_filter: Callable[..., dict[str, Any]]
    cmd_set_representation: Callable[..., dict[str, Any]]

    @staticmethod
    def _update_dataset(
        scene: dict[str, Any], growing: bool, descriptor: dict[str, Any] | None
    ) -> None:
        """A reopened scene takes what a newer descriptor says about following it."""
        dataset = scene.get("dataset")
        if not isinstance(dataset, dict):
            return
        if growing:
            dataset["growing"] = True
        if descriptor:
            dataset["descriptor"] = {**(dataset.get("descriptor") or {}), **descriptor}

    def _first_view(self, display: dict[str, Any] | None) -> None:
        """Show what the descriptor's display asks for: a slice or contour of the data,
        coloured by an array over a fixed range. A viewer may ignore it, so each part
        that cannot be shown is left out and noted with the dataset; the open stands."""
        if not display or self.scene is None:
            return
        from .data import DataError

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
        elif kind in ("Outline", "Surface", "Volume"):
            attempt(lambda: self.cmd_set_representation("o1", representation=kind))
        if made:
            target = made["object"]["id"]
            attempt(lambda: self.cmd_set_representation("o1", representation="Outline"))
        if array:
            colour: dict[str, Any] = {"color_by": array, "scalar_bar": True}
            if display.get("range"):
                colour["color_range"] = display["range"]
            if attempt(lambda: self.cmd_set_representation(target, **colour)):
                preset = display.get("color_map")
                name = preset_name(preset) if preset else None
                if preset and name is None:
                    notes.append(f"ParaView has no colour preset {preset!r}.")
                elif name:
                    attempt(lambda: self.cmd_set_representation(target, colormap=name))
        if notes:
            self.scene["dataset"]["display_note"] = (
                "The descriptor's display was shown in part: " + " ".join(notes)
            )

    def _files(self, item: dict[str, Any]) -> list[str] | str:
        path = item["path"]
        if not is_pattern(path):
            if not os.path.exists(path):
                raise CommandError("not_found", f"There is no file at {path}.")
            return path
        files = sorted(glob.glob(path))
        if not files:
            raise CommandError("not_found", f"No file matches {path}.")
        return files

    def _bp5(self, item: dict[str, Any]) -> bool:
        path = item["path"]
        return not is_pattern(path) and bp5.is_bp(path, item.get("reader"))

    def _make_reader(self, item: dict[str, Any]) -> Any:
        files = self._files(item)
        if self._bp5(item):
            reason = bp5.check(item["path"]).reason
            if reason:
                raise CommandError(
                    "not_ready",
                    f"{item['path']} cannot be read yet: {reason}. Open it once its "
                    "writer has written a whole step; open_view with growing set "
                    "waits for that while the folder is still changing.",
                )
        name = item.get("reader")
        if not name:
            proxy = self.simple.OpenDataFile(files)
            if proxy is None:
                raise CommandError(
                    "no_reader",
                    f"ParaView has no reader for {item['path']}. Name one in the "
                    "view descriptor's reader.",
                )
        else:
            from paraview import servermanager

            known = [n for n in dir(servermanager.sources) if not n.startswith("_")]
            if name not in known:
                readers = [n for n in known if "Reader" in n] or known
                close = difflib.get_close_matches(name, readers, n=5, cutoff=0.4)
                raise CommandError(
                    "unknown_reader",
                    f"ParaView has no reader {name!r}."
                    + (f" Close matches: {', '.join(close)}." if close else ""),
                )
            proxy = getattr(self.simple, name)()
            self._point_reader(proxy, item, files)
        for key, value in (item.get("reader_properties") or {}).items():
            set_property(proxy, name or proxy.GetXMLName(), key, value)
        proxy.UpdatePipelineInformation()
        self._growth[item["id"]] = disk_key(item["path"])
        return proxy

    @staticmethod
    def _point_reader(
        proxy: Any, item: dict[str, Any], files: list[str] | str, reset: bool = False
    ) -> None:
        """Give the reader its files. ``reset`` clears them first: setting the same
        value is not a change, and the reader must re-read its index."""
        names = proxy.ListProperties()
        if "FileNames" in names:
            if reset:
                proxy.FileNames = []
            proxy.FileNames = files if isinstance(files, list) else [files]
            return
        if "FileName" not in names:
            raise CommandError(
                "bad_reader",
                f"{proxy.GetXMLName()} takes no file name; it is not a reader.",
            )
        if reset:
            proxy.FileName = [] if isinstance(files, list) else ""
        if isinstance(files, list):
            # A repeatable FileName (the XML readers') takes a file series.
            if proxy.GetProperty("FileName").SMProperty.GetRepeatable():
                proxy.FileName = files
                return
            if len(files) != 1:
                raise CommandError(
                    "bad_path",
                    f"{proxy.GetXMLName()} reads one file or folder, and "
                    f"{item['path']} matches {len(files)}.",
                )
            files = files[0]
        proxy.FileName = files

    def _note_steps(self) -> None:
        """Remember when each time value was first seen, per scene."""
        scene_id = (self.scene or {}).get("scene")
        if scene_id != self._seen_scene:
            self._seen_scene = scene_id
            self._seen = {}
        now = time.time()
        for value in self.timesteps():
            self._seen.setdefault(value, now)

    def _grow_colours(self, scene: dict[str, Any]) -> None:
        """A colour map nobody fixed grows to take in each new step's values, as
        ParaView's own maps grow on Apply; a range set by hand stays where it is."""
        self.view.Update()
        for item in scene["objects"]:
            shown = item.get("display") or {}
            proxy = self.proxies.get(item["id"])
            if proxy is None or not shown.get("color_by") or shown.get("color_range"):
                continue
            if shown.get("visible") is False:
                continue
            rep = self.simple.GetDisplayProperties(proxy, self.view)
            rep.RescaleTransferFunctionToDataRange(True, False)

    def _bp5_settled(self, item: dict[str, Any], key: Any) -> bool:
        """Whether a grown BP5 folder can be re-read: its files agree, and its data
        clearly holds the last step or nothing has changed for ``SETTLE`` seconds.
        Until then the reader keeps the steps it has."""
        state = bp5.check(item["path"])
        if state.reason:
            self._settling.pop(item["id"], None)
            return False
        if state.clear:
            self._settling.pop(item["id"], None)
            return True
        now = time.monotonic()
        seen = self._settling.get(item["id"])
        if seen is None or seen[0] != key:
            self._settling[item["id"]] = (key, now)
            return False
        if now - seen[1] < SETTLE:
            return False
        self._settling.pop(item["id"], None)
        return True

    def check_growth(self) -> dict[str, Any] | None:
        """Re-read the time list of growing sources; follow the latest step."""
        scene = self.scene
        if scene is None or not (scene.get("dataset") or {}).get("growing"):
            return None
        self._note_steps()
        before = len(self.timesteps())
        changed = False
        for item in scene["objects"]:
            if item["kind"] != "reader" or item["id"] not in self.proxies:
                continue
            key = disk_key(item["path"])
            if key == self._growth.get(item["id"]):
                continue
            if self._bp5(item) and not self._bp5_settled(item, key):
                continue
            self._growth[item["id"]] = key
            changed = True
            proxy = self.proxies[item["id"]]
            if is_pattern(item["path"]):
                self._point_reader(proxy, item, self._files(item), reset=True)
            elif not self.simple.ReloadFiles(proxy):
                # A folder or file written in place, such as an ADIOS2 BP output:
                # the reader re-opens it. Resetting the name is the fallback.
                self._point_reader(proxy, item, self._files(item), reset=True)
            proxy.UpdatePipelineInformation()
        if not changed:
            return None
        self._apply_time(scene)
        if scene["time"].get("follow_latest", True):
            self._grow_colours(scene)
        self._note_steps()
        self.last_growth = time.time()
        after = len(self.timesteps())
        self.commit("growth")
        return {"new_steps": after - before, **self.time_summary()}

    def cmd_steps(self, after: int = 0) -> dict[str, Any]:
        """The scene's time steps from index ``after`` on, each with when it was first
        seen, after checking a growing source for new ones."""
        scene = self.require_scene()
        self._note_steps()
        self.check_growth()
        start = max(0, int(after))
        steps = self.timesteps()
        dataset = scene.get("dataset") or {}
        return {
            "scene": scene["scene"],
            "growing": bool(dataset.get("growing")),
            "follow": (dataset.get("descriptor") or {}).get("follow"),
            "steps": [
                {"index": i, "time": value, "seen": self._seen.get(value)}
                for i, value in enumerate(steps)
                if i >= start
            ],
            "time": self.time_summary(),
            "last_growth": self.last_growth,
            "checked": time.time(),
        }
