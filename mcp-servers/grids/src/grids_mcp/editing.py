"""Hand edits to a gridded field, as an ordered list that can be replayed.

A model grid (bathymetry, a land mask, a topography) often needs edits by
hand: close a channel the model cannot resolve, open a strait, set a minimum
depth, remove lakes cut off from the open ocean. Every edit here is a plain
record in an ordered list, located by longitude and latitude, so the same
list can be applied again to a regenerated grid. Applying a list to a base
grid is deterministic: the result depends only on the base and the list.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr
from scipy import ndimage

LOG_SCHEMA = "clio.grid-edits/1"
OPS = ("set", "clip", "fill_enclosed")
WET_RULES = ("positive", "negative", "nonzero")
MAX_WINDOW_CELLS = 512
LAT_NAMES = ("lat", "latitude", "nav_lat", "y", "yh", "lat_rho", "geolat")
LON_NAMES = ("lon", "longitude", "nav_lon", "x", "xh", "lon_rho", "geolon")


class GridError(Exception):
    """A typed failure: ``reason`` is one of the server's reason codes."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason
        self.message = message


@dataclass
class Grid:
    """One 2-D field with its cell-centre longitudes and latitudes (2-D each)."""

    path: Path
    variable: str
    values: np.ndarray
    lon: np.ndarray
    lat: np.ndarray
    dims: tuple[str, str]
    dataset: xr.Dataset

    @property
    def shape(self) -> tuple[int, int]:
        return (int(self.values.shape[0]), int(self.values.shape[1]))


def _coordinate(
    dataset: xr.Dataset, data: xr.DataArray, names: tuple[str, ...]
) -> np.ndarray:
    for name in names:
        if name in data.coords or name in dataset:
            return np.asarray(dataset[name].values, dtype=float)
    raise GridError(
        "bad_input",
        f"no {names[0]} coordinate found (looked for {', '.join(names)})",
    )


def open_grid(path: str | Path, variable: str | None = None) -> Grid:
    """Open one 2-D variable of a NetCDF file, with 2-D lon/lat of the cell centres."""

    file = Path(path).expanduser()
    if not file.is_file():
        raise GridError("bad_input", f"{file} does not exist")
    try:
        dataset = xr.open_dataset(file).load()
    except Exception as exc:
        raise GridError(
            "bad_input", f"{file} is not a readable NetCDF file: {exc}"
        ) from exc
    candidates = [name for name, var in dataset.data_vars.items() if var.ndim == 2]
    if variable is None:
        if len(candidates) != 1:
            raise GridError(
                "bad_input",
                f"name the variable: 2-D variables in {file.name} are {candidates or 'none'}",
            )
        variable = str(candidates[0])
    if variable not in dataset.data_vars:
        raise GridError(
            "bad_input", f"{variable!r} is not in {file.name}; 2-D: {candidates}"
        )
    data = dataset[variable]
    if data.ndim != 2:
        raise GridError(
            "bad_input", f"{variable!r} has {data.ndim} dimensions; a grid has 2"
        )
    lat = _coordinate(dataset, data, LAT_NAMES)
    lon = _coordinate(dataset, data, LON_NAMES)
    if lat.ndim == 1 and lon.ndim == 1:
        lon, lat = np.meshgrid(lon, lat)
    if lat.shape != data.shape or lon.shape != data.shape:
        raise GridError(
            "bad_input",
            f"coordinates {lat.shape} do not match {variable!r} {data.shape}",
        )
    return Grid(
        path=file,
        variable=variable,
        values=np.asarray(data.values, dtype=float),
        lon=lon,
        lat=lat,
        dims=(str(data.dims[0]), str(data.dims[1])),
        dataset=dataset,
    )


def grid_hash(values: np.ndarray, lon: np.ndarray, lat: np.ndarray) -> str:
    """A content hash of the field and its cell centres (NaN-stable)."""

    digest = hashlib.sha256()
    digest.update(repr(values.shape).encode())
    for array in (values, lon, lat):
        digest.update(
            np.ascontiguousarray(np.nan_to_num(array, nan=-9.87e33), dtype="<f8")
        )
    return digest.hexdigest()[:16]


def _wrap(lon: np.ndarray | float) -> Any:
    return (np.asarray(lon) + 180.0) % 360.0 - 180.0


def region_mask(grid: Grid, region: dict[str, Any] | None) -> np.ndarray:
    """The cells an edit's region covers: a lon/lat box, a list of points, or all."""

    if not region:
        return np.ones(grid.shape, dtype=bool)
    if "box" in region:
        box = region["box"]
        if not (isinstance(box, list) and len(box) == 4):
            raise GridError("bad_input", "region.box is [west, south, east, north]")
        west, south, east, north = (float(v) for v in box)
        if south > north:
            raise GridError("bad_input", "region.box south is above north")
        lon = _wrap(grid.lon)
        west, east = float(_wrap(west)), float(_wrap(east))
        inside_lon = (
            (lon >= west) & (lon <= east)
            if west <= east
            else (lon >= west) | (lon <= east)
        )
        return inside_lon & (grid.lat >= south) & (grid.lat <= north)
    if "points" in region:
        mask = np.zeros(grid.shape, dtype=bool)
        lon = _wrap(grid.lon)
        for point in region["points"]:
            plon, plat = float(_wrap(point[0])), float(point[1])
            dlon = np.abs(_wrap(lon - plon)) * np.cos(np.radians(plat))
            index = np.unravel_index(
                np.argmin(dlon**2 + (grid.lat - plat) ** 2), grid.shape
            )
            mask[index] = True
        return mask
    raise GridError(
        "bad_input", "region is {box: [w, s, e, n]} or {points: [[lon, lat], ...]}"
    )


def _wet(values: np.ndarray, rule: str) -> np.ndarray:
    if rule == "positive":
        return values > 0
    if rule == "negative":
        return values < 0
    return (values != 0) & np.isfinite(values)


def _check(edit: Any, index: int) -> dict[str, Any]:
    where = f"edits[{index}]"
    if not isinstance(edit, dict):
        raise GridError("bad_input", f"{where} is not an object")
    op = edit.get("op")
    if op not in OPS:
        raise GridError(
            "bad_input", f"{where}.op is one of {', '.join(OPS)}, not {op!r}"
        )
    if op == "set" and not isinstance(edit.get("value"), (int, float)):
        raise GridError("bad_input", f"{where}: set needs a numeric value")
    if op == "clip" and edit.get("min") is None and edit.get("max") is None:
        raise GridError("bad_input", f"{where}: clip needs min, max or both")
    if op == "clip" and edit.get("wet") not in (None, *WET_RULES):
        raise GridError("bad_input", f"{where}: clip's wet is one of {WET_RULES}")
    if op == "fill_enclosed":
        if edit.get("wet") not in WET_RULES:
            raise GridError(
                "bad_input", f"{where}: fill_enclosed needs wet, one of {WET_RULES}"
            )
        if not isinstance(edit.get("land_value"), (int, float)):
            raise GridError(
                "bad_input", f"{where}: fill_enclosed needs a numeric land_value"
            )
    return edit


def apply_edit(
    values: np.ndarray, grid: Grid, edit: dict[str, Any]
) -> tuple[np.ndarray, int]:
    """One edit applied to ``values``: the new values and the number of cells changed."""

    mask = region_mask(grid, edit.get("region"))
    out = values.copy()
    op = edit["op"]
    if op == "set":
        out[mask] = float(edit["value"])
    elif op == "clip":
        low = edit.get("min")
        high = edit.get("max")
        target = mask & _wet(values, edit["wet"]) if edit.get("wet") else mask
        clipped = np.clip(
            values,
            -np.inf if low is None else float(low),
            np.inf if high is None else float(high),
        )
        out[target] = clipped[target]
    else:
        wet = _wet(values, edit["wet"])
        labels, count = ndimage.label(wet)
        if count > 1:
            sizes = np.bincount(labels.ravel())
            sizes[0] = 0
            keep = int(np.argmax(sizes))
            enclosed = wet & (labels != keep) & mask
            out[enclosed] = float(edit["land_value"])
    changed = int(
        np.count_nonzero(~((out == values) | (np.isnan(out) & np.isnan(values))))
    )
    return out, changed


def apply_edits(
    grid: Grid, edits: list[Any]
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    """The base grid with every edit applied in order, and each edit with its cell count."""

    values = grid.values
    applied = []
    for index, raw in enumerate(edits):
        edit = _check(raw, index)
        values, changed = apply_edit(values, grid, edit)
        applied.append({**edit, "cells_changed": changed})
    return values, applied


def write_result(
    grid: Grid, values: np.ndarray, output: Path, log: dict[str, Any]
) -> None:
    """The edited grid as NetCDF (the base's other variables kept) and its edit log."""

    dataset = grid.dataset.copy()
    dataset[grid.variable] = (
        grid.dims,
        values.astype(grid.dataset[grid.variable].dtype),
    )
    dataset[grid.variable].attrs = dict(grid.dataset[grid.variable].attrs)
    history = dataset.attrs.get("history", "")
    dataset.attrs["history"] = (
        f"{history}\n" if history else ""
    ) + f"grids-mcp: {len(log['edits'])} edits from {output.name}.edits.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    log_path(output).write_text(json.dumps(log, indent=2), encoding="utf-8")


def log_path(output: Path) -> Path:
    return output.with_name(output.name + ".edits.json")


def window(grid: Grid, values: np.ndarray, max_cells: int) -> dict[str, Any]:
    """The field strided to at most ``max_cells`` per axis, for a view."""

    cells = max(8, min(int(max_cells), MAX_WINDOW_CELLS))
    rows, cols = grid.shape
    step_r = max(1, int(np.ceil(rows / cells)))
    step_c = max(1, int(np.ceil(cols / cells)))
    part = values[::step_r, ::step_c]
    finite = part[np.isfinite(part)]
    return {
        "shape": list(part.shape),
        "stride": [step_r, step_c],
        "values": [
            [None if not np.isfinite(v) else round(float(v), 4) for v in row]
            for row in part
        ],
        "lon": [round(float(v), 5) for v in grid.lon[::step_r, ::step_c][0]],
        "lat": [round(float(v), 5) for v in grid.lat[::step_r, ::step_c][:, 0]],
        "range": [float(finite.min()), float(finite.max())] if finite.size else None,
    }
