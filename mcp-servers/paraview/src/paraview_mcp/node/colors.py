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


def _same_field(
    scene: dict[str, Any] | None, field: str | None
) -> list[dict[str, Any]]:
    out = []
    for item in (scene or {}).get("objects") or []:
        shown = item.get("display") or {}
        if field and (shown.get("color_by") or [None, None])[1] == field:
            out.append(shown)
    return out


def share(scene: dict[str, Any], color_by: Any, key: str, value: Any) -> None:
    """ParaView has one colour map and one legend per field: record a change of
    either on every object coloured by that field."""
    for shown in _same_field(scene, (color_by or [None, None])[1]):
        shown[key] = value


def legend_wanted(scene: dict[str, Any] | None, color_by: Any) -> bool:
    """Whether the field's one legend is on: any object coloured by it asks for it."""
    field = (color_by or [None, None])[1]
    return any(shown.get("scalar_bar") for shown in _same_field(scene, field))


def checked_range(
    color_range: list[float] | None, rescale: bool | None
) -> tuple[float, float] | None:
    """A hand-set colour range, checked before anything changes."""
    from .errors import CommandError

    if color_range is None:
        return None
    if rescale:
        raise CommandError(
            "range_and_rescale",
            "Give either rescale (the data's range) or color_range (a range by hand).",
        )
    low, high = (float(v) for v in color_range)
    if not low < high:
        raise CommandError(
            "bad_color_range",
            f"The colour range must rise: {low:g} is not below {high:g}.",
        )
    return low, high


def set_range(
    simple: Any, scene: dict[str, Any], color_by: Any, span: tuple[float, float]
) -> None:
    """Set a field's colour map, and its opacity ramp for volumes, to ``span``.

    The range then stays put as time steps change; ``rescale`` returns it to the data.
    """
    from .errors import CommandError

    field = (color_by or [None, None])[1]
    if not field:
        raise CommandError(
            "no_color_field", "A colour range needs the object coloured by a field."
        )
    lut = simple.GetColorTransferFunction(field)
    if "AutomaticRescaleRangeMode" in lut.ListProperties():
        lut.AutomaticRescaleRangeMode = "Never"
    lut.RescaleTransferFunction(*span)
    simple.GetOpacityTransferFunction(field).RescaleTransferFunction(*span)
    share(scene, color_by, "color_range", list(span))


#: Plain colours a person names, as ParaView's 0-1 RGB.
NAMED = {
    "white": (1.0, 1.0, 1.0),
    "black": (0.0, 0.0, 0.0),
    "grey": (0.5, 0.5, 0.5),
    "gray": (0.5, 0.5, 0.5),
    "red": (0.89, 0.1, 0.11),
    "orange": (1.0, 0.5, 0.0),
    "yellow": (1.0, 0.85, 0.0),
    "green": (0.2, 0.75, 0.2),
    "cyan": (0.0, 0.85, 0.9),
    "blue": (0.15, 0.4, 1.0),
    "purple": (0.6, 0.25, 0.85),
    "magenta": (0.95, 0.15, 0.75),
    "pink": (1.0, 0.55, 0.7),
    "brown": (0.6, 0.35, 0.15),
}


def parse_color(text: str) -> list[float]:
    """A colour by name or ``#rrggbb``, as ParaView's 0-1 RGB."""
    from .errors import CommandError

    key = text.strip().casefold()
    if key in NAMED:
        return list(NAMED[key])
    hexa = key.removeprefix("#")
    if len(hexa) == 6 and all(c in "0123456789abcdef" for c in hexa):
        return [round(int(hexa[i : i + 2], 16) / 255, 4) for i in (0, 2, 4)]
    raise CommandError(
        "unknown_color",
        f"{text!r} is not a colour: give #rrggbb or one of {', '.join(sorted(NAMED))}.",
    )
