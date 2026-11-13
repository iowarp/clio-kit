"""The filters the view's Add panel offers, and their short property forms.

Each filter has a few fields, named by their ParaView property, with defaults
taken from the input's data (its bounds and arrays). The form's values become
the ``properties`` of the same ``create_filter`` command the agent's tool sends.
Everything here is plain Python, so the front's tests can check it.
"""

from __future__ import annotations

import math
from typing import Any

# name, icon, short tooltip; what the input must carry ("scalar" or "vector").
CATALOG: tuple[dict[str, Any], ...] = (
    {"name": "Contour", "icon": "mdi-circle-double", "needs": "point_scalar"},
    {"name": "Slice", "icon": "mdi-slash-forward-box", "needs": None},
    {"name": "Clip", "icon": "mdi-content-cut", "needs": None},
    {"name": "Threshold", "icon": "mdi-tune-variant", "needs": "scalar"},
    {"name": "Glyph", "icon": "mdi-arrow-top-right-bold-outline", "needs": "vector"},
    {"name": "StreamTracer", "icon": "mdi-weather-windy", "needs": "vector"},
    {"name": "Calculator", "icon": "mdi-calculator-variant-outline", "needs": None},
)
TITLES = {"StreamTracer": "Stream Tracer"}
NEEDS_TEXT = {
    "point_scalar": "needs a point field",
    "scalar": "needs a field",
    "vector": "needs a 3-component point field",
}
GLYPH_TYPES = ("Arrow", "Cone", "Sphere", "Line", "Cylinder", "Box")
SEED_TYPES = ("Point Cloud", "Line")
ATTRIBUTE_TYPES = ("Point Data", "Cell Data")


def title(name: str) -> str:
    return TITLES.get(name, name)


def _center(bounds: list[float] | None) -> list[float]:
    if not bounds:
        return [0.0, 0.0, 0.0]
    return [round((bounds[i] + bounds[i + 1]) / 2, 6) for i in (0, 2, 4)]


def _diagonal(bounds: list[float] | None) -> float:
    if not bounds:
        return 1.0
    return math.sqrt(sum((bounds[i + 1] - bounds[i]) ** 2 for i in (0, 2, 4))) or 1.0


def _arrays(info: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    arrays = info.get("arrays") or []
    if kind == "point_scalar":
        return [a for a in arrays if a["association"] == "POINTS"]
    if kind == "vector":
        return [
            a for a in arrays if a["association"] == "POINTS" and a["components"] == 3
        ]
    return list(arrays)


def array_key(array: dict[str, Any]) -> str:
    return f"{array['association']}:{array['name']}"


def _array_items(arrays: list[dict[str, Any]]) -> list[dict[str, str]]:
    return [
        {
            "value": array_key(a),
            "title": f"{a['name']} ({'point' if a['association'] == 'POINTS' else 'cell'})",
        }
        for a in arrays
    ]


def offered(info: dict[str, Any] | None) -> list[dict[str, Any]]:
    """The catalog for an input: each filter, and why it is off when it is."""
    out = []
    for entry in CATALOG:
        needs = entry["needs"]
        usable = info is not None and (not needs or bool(_arrays(info, needs)))
        reason = "" if usable else NEEDS_TEXT.get(needs or "", "select an object first")
        out.append(
            {
                "name": entry["name"],
                "title": title(entry["name"]),
                "icon": entry["icon"],
                "disabled": not usable,
                "tip": title(entry["name"]) + (f" ({reason})" if reason else ""),
            }
        )
    return out


def _field(key: str, label: str, kind: str, value: Any, **extra: Any) -> dict[str, Any]:
    return {"key": key, "label": label, "kind": kind, "value": value, **extra}


def form(name: str, info: dict[str, Any]) -> list[dict[str, Any]]:
    """The form fields for ``name`` on an input described by ``info``."""
    bounds = info.get("bounds")
    center = _center(bounds)
    if name == "Contour":
        arrays = _arrays(info, "point_scalar")
        first = arrays[0] if arrays else None
        mid = (first["range"][0] + first["range"][1]) / 2 if first else 0.0
        return [
            _field(
                "ContourBy",
                "Contour by",
                "array",
                array_key(first) if first else "",
                items=_array_items(arrays),
                ranges=_ranges(arrays),
            ),
            _field("Isosurfaces", "Values", "numbers", f"{mid:.6g}"),
        ]
    if name in ("Slice", "Clip"):
        part = "SliceType" if name == "Slice" else "ClipType"
        fields = [
            _field(f"{part}.Origin", "Origin", "vector", center),
            _field(f"{part}.Normal", "Normal", "vector", [1.0, 0.0, 0.0], axes=True),
        ]
        if name == "Clip":
            fields.append(_field("Invert", "Keep inside", "bool", True))
        return fields
    if name == "Threshold":
        arrays = _arrays(info, "scalar")
        first = arrays[0] if arrays else None
        low, high = first["range"] if first else (0.0, 1.0)
        return [
            _field(
                "Scalars",
                "Field",
                "array",
                array_key(first) if first else "",
                items=_array_items(arrays),
                ranges=_ranges(arrays),
            ),
            _field("LowerThreshold", "From", "number", low),
            _field("UpperThreshold", "To", "number", high),
        ]
    if name == "Glyph":
        arrays = _arrays(info, "vector")
        return [
            _field(
                "OrientationArray",
                "Orient by",
                "array",
                array_key(arrays[0]) if arrays else "",
                items=_array_items(arrays),
            ),
            _field("GlyphType", "Shape", "select", "Arrow", items=list(GLYPH_TYPES)),
            _field(
                "ScaleFactor", "Scale", "number", round(_diagonal(bounds) * 0.05, 6)
            ),
        ]
    if name == "StreamTracer":
        arrays = _arrays(info, "vector")
        return [
            _field(
                "Vectors",
                "Vectors",
                "array",
                array_key(arrays[0]) if arrays else "",
                items=_array_items(arrays),
            ),
            _field(
                "SeedType", "Seeds", "select", "Point Cloud", items=list(SEED_TYPES)
            ),
            _field(
                "MaximumStreamlineLength",
                "Max length",
                "number",
                round(_diagonal(bounds), 6),
            ),
        ]
    if name == "Calculator":
        return [
            _field("ResultArrayName", "Result", "text", "Result"),
            _field("Function", "Expression", "text", ""),
            _field(
                "AttributeType",
                "On",
                "select",
                "Point Data",
                items=list(ATTRIBUTE_TYPES),
            ),
        ]
    raise KeyError(name)


def _ranges(arrays: list[dict[str, Any]]) -> dict[str, list[float]]:
    return {array_key(a): a["range"] for a in arrays}


class FormError(ValueError):
    """A form value that is not a number, vector or field where one is needed."""


def _float(label: str, value: Any) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        raise FormError(f"{label}: {value!r} is not a number.") from None
    if not math.isfinite(number):
        raise FormError(f"{label}: give a finite number.")
    return number


def properties(fields: list[dict[str, Any]], values: dict[str, Any]) -> dict[str, Any]:
    """Turn the form's values into ``create_filter`` properties."""
    out: dict[str, Any] = {}
    for spec in fields:
        key, label, kind = spec["key"], spec["label"], spec["kind"]
        value = values.get(key, spec.get("value"))
        if kind == "array":
            if not value:
                raise FormError(f"{label}: choose a field.")
            association, _, array = str(value).partition(":")
            out[key] = [association, array]
        elif kind == "number":
            out[key] = _float(label, value)
        elif kind == "numbers":
            parts = [p for p in str(value).replace(",", " ").split() if p]
            if not parts:
                raise FormError(f"{label}: give at least one value.")
            out[key] = [_float(label, p) for p in parts]
        elif kind == "vector":
            items = list(value) if isinstance(value, (list, tuple)) else []
            if len(items) != 3:
                raise FormError(f"{label}: give three numbers.")
            vector = [_float(label, v) for v in items]
            if spec.get("axes") and not any(vector):
                raise FormError(f"{label}: the normal cannot be zero.")
            out[key] = vector
        elif kind == "bool":
            out[key] = 1 if value else 0
        elif kind == "text":
            text = str(value or "").strip()
            if not text:
                raise FormError(f"{label}: this is required.")
            out[key] = text
        else:
            out[key] = value
    return out
