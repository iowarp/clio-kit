"""Colour maps as scene state: saved with the scene, restored exactly, never shared.

ParaView keeps one colour map (lookup table) per field name for the whole
process. Two things followed from that: a reopened scene got ParaView's
default colour map, or a range ParaView recomputed, rather than the one it had;
and a scene opened after another one inherited that scene's colour maps. So
each commit saves every colour map the scene uses (its colour points, which
carry both the preset and the range, and how it maps vectors), a rebuild
applies them back, and clearing a scene removes the process's colour maps so
the next scene starts as a fresh ParaView would.
"""

from __future__ import annotations

from typing import Any

SAVED = (
    "RGBPoints",
    "ColorSpace",
    "NanColor",
    "UseLogScale",
    "VectorMode",
    "VectorComponent",
    "AutomaticRescaleRangeMode",
    "NameOfLastPresetApplied",
    "Discretize",
    "NumberOfTableValues",
)
GROUPS = ("lookup_tables", "piecewise_functions")


def _plain(value: Any) -> Any:
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    try:
        return [_plain(v) for v in value]
    except TypeError:
        return str(value)


def fields(scene: dict[str, Any]) -> list[str]:
    """Every field the scene colours an object by."""
    names: list[str] = []
    for item in scene.get("objects") or []:
        color_by = (item.get("display") or {}).get("color_by")
        if color_by and color_by[1] and color_by[1] not in names:
            names.append(color_by[1])
    return names


def snapshot(simple: Any, scene: dict[str, Any]) -> dict[str, Any]:
    """The colour map of each field the scene colours by, as plain data."""
    saved: dict[str, Any] = {}
    for name in fields(scene):
        lut = simple.GetColorTransferFunction(name)
        entry = {
            key: _plain(getattr(lut, key))
            for key in SAVED
            if key in lut.ListProperties()
        }
        entry["OpacityPoints"] = _plain(simple.GetOpacityTransferFunction(name).Points)
        saved[name] = entry
    return saved


def restore(simple: Any, saved: dict[str, Any] | None) -> None:
    """Apply saved colour maps exactly (after the objects are shown)."""
    for name, entry in (saved or {}).items():
        lut = simple.GetColorTransferFunction(name)
        known = lut.ListProperties()
        for key in SAVED:
            if key in entry and key in known and key != "NameOfLastPresetApplied":
                setattr(lut, key, entry[key])
        if "NameOfLastPresetApplied" in entry and "NameOfLastPresetApplied" in known:
            lut.NameOfLastPresetApplied = entry["NameOfLastPresetApplied"]
        if entry.get("OpacityPoints"):
            simple.GetOpacityTransferFunction(name).Points = entry["OpacityPoints"]


def forget(servermanager: Any, view: Any) -> None:
    """Remove the process's colour maps and legends, so the next scene starts fresh."""
    pxm = servermanager.ProxyManager()
    for (name, _id), bar in list(pxm.GetProxiesInGroup("scalar_bars").items()):
        if bar in view.Representations:
            view.Representations.remove(bar)
        pxm.UnRegisterProxy("scalar_bars", name, bar)
    for group in GROUPS:
        for (name, _id), proxy in list(pxm.GetProxiesInGroup(group).items()):
            pxm.UnRegisterProxy(group, name, proxy)


def preset_name(simple: Any, field: str | None) -> str | None:
    """The preset a field's colour map last took, for the Display panel."""
    if not field:
        return None
    lut = simple.GetColorTransferFunction(field)
    if "NameOfLastPresetApplied" not in lut.ListProperties():
        return None
    return str(lut.NameOfLastPresetApplied) or None
