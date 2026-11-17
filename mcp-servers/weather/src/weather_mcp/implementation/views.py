"""The view each output is shown in: one CLIO workspace component, ready to place.

A person checks profiles, sections, fields and tracks by looking at them, so a tool's
result names, per output, the component that shows it with the output file as its data
reference (``Output.view``). The agent puts it on a surface as it is, or adapts it
(title, linked selection, which variable first); nothing is shown without it. Outputs
whose shape has no fitting component (a summary table, a series) get no view.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

#: Columns identifying one profile, in the order they are looked for.
PROFILE_ID = ("profile", "profile_id", "cast", "station")
#: Vertical columns and their units; ``True`` when the axis is drawn increasing downward.
VERTICAL = {
    "pressure": ("dbar", True),
    "depth": ("m", True),
    "pressure_hpa": ("hPa", True),
    "height_m": ("m", False),
}
#: Columns that place or identify a row rather than measure something.
_COORDINATES = {
    "time",
    "latitude",
    "longitude",
    "lat",
    "lon",
    "requested",
    "name",
    *PROFILE_ID,
    *VERTICAL,
}
#: Most variables a profile view offers to switch between.
MAX_VARIABLES = 8
#: Point and event tables: columns identifying a feature, in the order looked for.
FEATURE_ID = ("id", "station", "event", "track", "profile", "name")


def _first(columns: list[str], names: tuple[str, ...] | list[str]) -> str:
    return next((name for name in names if name in columns), "")


def _numeric(table: Any) -> list[str]:
    import pandas as pd

    return [
        str(column)
        for column in table.columns
        if str(column) not in _COORDINATES
        and pd.api.types.is_numeric_dtype(table[column])
    ]


#: Columns naming the track a row belongs to (eddies, storms, floats, particles).
TRACK_ID = ("track", "particle", "storm", "sid", "float", "platform")


def table_view(
    role: str, path: Path, table: Any, title: str = ""
) -> dict[str, Any] | None:
    """The component showing a saved table, or ``None`` when no component fits it."""

    columns = [str(column) for column in table.columns]
    label = title or path.stem
    if role == "profiles":
        profile = _first(columns, PROFILE_ID)
        vertical = _first(columns, tuple(VERTICAL))
        values = _numeric(table)
        if not (profile and vertical and values):
            return None
        unit, downward = VERTICAL[vertical]
        return {
            "component": "clio.profile-stack.v1",
            "title": label,
            "dataUri": str(path),
            "profileField": profile,
            "verticalField": vertical,
            "verticalUnit": unit,
            "invertVertical": downward,
            "valueField": values[0],
            "variables": values[:MAX_VARIABLES],
        }
    if "latitude" not in columns or "longitude" not in columns:
        return None
    track = _first(columns, TRACK_ID) if role == "tracks" else None
    if track and "time" in columns:
        layer = {
            "id": "tracks",
            "kind": "tracks",
            "label": label,
            "dataUri": str(path),
            "latitudeField": "latitude",
            "longitudeField": "longitude",
            "idField": track,
            "trackField": track,
            "timeField": "time",
        }
    else:
        feature = _first(columns, FEATURE_ID)
        if not feature:
            return None
        layer = {
            "id": "points",
            "kind": "points",
            "label": label,
            "dataUri": str(path),
            "latitudeField": "latitude",
            "longitudeField": "longitude",
            "idField": feature,
        }
    return {"component": "clio.layered-map.v1", "title": label, "layers": [layer]}


def _zero_span(values: Any) -> float:
    """The largest distance from zero when the values lie on both sides of it, else 0."""
    import numpy as np

    data = np.asarray(values)
    if data.size == 0 or not np.issubdtype(data.dtype, np.number):
        return 0.0
    low, high = float(np.nanmin(data)), float(np.nanmax(data))
    if not (np.isfinite(low) and np.isfinite(high) and low < 0 < high):
        return 0.0
    return max(-low, high)


def grid_view(
    role: str, path: Path, dataset: Any, title: str = ""
) -> dict[str, Any] | None:
    """The component showing a saved grid or section, or ``None`` when none fits it."""

    label = title or path.stem
    variables = [str(name) for name in dataset.data_vars]
    if not variables:
        return None
    if role == "section":
        return {
            "component": "clio.section.v1",
            "title": label,
            "sectionUri": str(path),
            "variable": variables[0],
            "variables": variables[:MAX_VARIABLES],
        }
    dims = {str(dim) for dim in dataset.dims}
    horizontal = ({"latitude", "longitude"} <= dims) or ({"lat", "lon"} <= dims)
    mapped = [
        name
        for name in variables
        if {"latitude", "longitude"} <= set(map(str, dataset[name].dims))
        or {"lat", "lon"} <= set(map(str, dataset[name].dims))
    ]
    if not (horizontal and mapped):
        return None
    layer = {
        "id": "field",
        "kind": "grid",
        "label": label,
        "gridUri": str(path),
        "variable": mapped[0],
    }
    span = _zero_span(dataset[mapped[0]])
    if span:
        # Values either side of zero (a bias, an anomaly): a diverging scale centred on 0.
        layer["colormap"] = "balance"
        layer["valueRange"] = [-span, span]
    return {"component": "clio.layered-map.v1", "title": label, "layers": [layer]}


def grid_file_view(role: str, path: Path, title: str = "") -> dict[str, Any] | None:
    """:func:`grid_view` of a grid another process wrote, read from its metadata."""

    import xarray as xr

    with xr.open_dataset(path) as dataset:
        return grid_view(role, path, dataset, title)
