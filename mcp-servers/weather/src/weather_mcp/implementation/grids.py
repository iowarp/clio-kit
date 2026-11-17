"""Gridded-field helpers shared by the weather tools.

Coordinates are found by name, not assumed; longitudes are compared in the
dataset's own convention (0..360 or -180..180); area means are weighted by
cos(latitude).
"""

from __future__ import annotations

import glob
from pathlib import Path
from typing import Any

import numpy as np
import xarray as xr

from .common import WeatherToolError

LON = ("longitude", "lon", "x")
LAT = ("latitude", "lat", "y")
TIME = ("time", "valid_time", "t")
LEVEL = ("pressure_level", "isobaricInhPa", "level", "plev", "lev")
MEMBER = ("number", "member", "realization", "ensemble")


def pick(
    ds: xr.Dataset | xr.DataArray, names: tuple[str, ...], required: bool = True
) -> str | None:
    for name in names:
        if name in ds.coords or name in ds.dims:
            return name
    if required:
        raise WeatherToolError(
            "bad_input", f"no coordinate named any of {', '.join(names)}"
        )
    return None


def open_grid(path: str, variables: list[str] | None = None) -> xr.Dataset:
    """Open NetCDF, GRIB (through cfgrib), Zarr, or a glob of NetCDF files."""
    expanded = str(Path(path).expanduser())
    matches = (
        sorted(glob.glob(expanded))
        if not expanded.startswith(("gs://", "s3://", "http"))
        else [expanded]
    )
    if not matches:
        raise WeatherToolError("bad_input", f"no file matches {path}")
    try:
        first = matches[0]
        if len(matches) > 1:
            ds = xr.open_mfdataset(matches, combine="by_coords")
        elif first.endswith((".grib", ".grib2", ".grb", ".grb2")):
            ds = xr.open_dataset(
                first, engine="cfgrib", backend_kwargs={"indexpath": ""}
            )
        elif (
            first.endswith(".zarr") or first.startswith("gs://") or Path(first).is_dir()
        ):
            ds = xr.open_zarr(
                first,
                storage_options={"token": "anon"}
                if first.startswith("gs://")
                else None,
            )
        else:
            ds = xr.open_dataset(first)
    except WeatherToolError:
        raise
    except Exception as exc:
        raise WeatherToolError("bad_input", f"could not open {path}: {exc}") from exc
    if variables:
        missing = [v for v in variables if v and v not in ds]
        if missing:
            raise WeatherToolError(
                "bad_input",
                f"{Path(path).name} lacks {', '.join(missing)}; it has {', '.join(map(str, ds.data_vars))}",
            )
    return ds


def to_dataset_lon(ds: xr.Dataset | xr.DataArray, lon: float) -> float:
    """A longitude in the dataset's convention."""
    name = pick(ds, LON)
    values = ds[name].to_numpy()
    if values.max() > 180 and lon < 0:
        return lon + 360
    if values.min() < 0 and lon > 180:
        return lon - 360
    return lon


def select_box(da: xr.DataArray, box: list[float]) -> xr.DataArray:
    """``[lon_min, lon_max, lat_min, lat_max]`` subset, either latitude order."""
    lon, lat = pick(da, LON), pick(da, LAT)
    x0, x1 = to_dataset_lon(da, box[0]), to_dataset_lon(da, box[1])
    if x0 <= x1:
        out = da.sel({lon: slice(x0, x1)})
    else:  # box crosses the dataset's longitude seam
        out = xr.concat(
            [da.sel({lon: slice(x0, None)}), da.sel({lon: slice(None, x1)})], dim=lon
        )
    ascending = bool(out[lat][0] < out[lat][-1]) if out.sizes[lat] > 1 else True
    out = out.sel({lat: slice(box[2], box[3]) if ascending else slice(box[3], box[2])})
    if out.size == 0:
        raise WeatherToolError("no_data", f"box {box} selects no grid cells")
    return out


def area_mean(da: xr.DataArray) -> xr.DataArray:
    lon, lat = pick(da, LON), pick(da, LAT)
    weights = xr.DataArray(
        np.cos(np.deg2rad(da[lat].to_numpy())), coords={lat: da[lat]}, dims=[lat]
    )
    return da.weighted(weights).mean([lon, lat])


def at_point(da: xr.DataArray, point: list[float]) -> xr.DataArray:
    lon, lat = pick(da, LON), pick(da, LAT)
    return da.sel({lon: to_dataset_lon(da, point[0]), lat: point[1]}, method="nearest")


def describe(ds: xr.Dataset) -> dict[str, Any]:
    """Short description of a dataset for a result."""
    out: dict[str, Any] = {
        "variables": [str(v) for v in ds.data_vars],
        "dims": {str(k): int(v) for k, v in ds.sizes.items()},
    }
    time = pick(ds, TIME, required=False)
    if time:
        values = ds[time].to_numpy()
        out["time_range"] = [str(values.min())[:19], str(values.max())[:19]]
    return out
