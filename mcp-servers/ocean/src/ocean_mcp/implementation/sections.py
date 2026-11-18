"""Sections: a field along a path against a vertical coordinate.

Two sources. From profiles (a cruise or a glider transect): order the casts,
interpolate each onto pressure levels, and lay them along the along-track
distance. From a gridded field: sample the field along a path the caller
gives (a list of lon/lat vertices) at a stated spacing.

Every section is a NetCDF file with dimensions ``(distance, <vertical>)``
plus per-column ``latitude``, ``longitude`` and, for profile sections,
``profile``; the path is stored as a GeoJSON LineString in the ``path``
attribute so a map can draw it and a redrawn path can be compared with it.
"""

from __future__ import annotations

import itertools
import json
from typing import Any, Literal

import gsw
import numpy as np
import pandas as pd
import xarray as xr

from .common import OceanToolError

Order = Literal["time", "latitude", "longitude", "given"]


def along_track_km(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Cumulative great-circle distance along vertices, km."""
    if len(lon) < 2:
        return np.zeros(len(lon))
    steps = gsw.distance(np.asarray(lon, float), np.asarray(lat, float), 0) / 1000.0
    return np.concatenate([[0.0], np.cumsum(np.asarray(steps).ravel())])


def path_geojson(lon: np.ndarray, lat: np.ndarray) -> dict[str, Any]:
    return {
        "type": "LineString",
        "coordinates": [[float(x), float(y)] for x, y in zip(lon, lat, strict=True)],
    }


def section_from_profiles(
    table: pd.DataFrame,
    variables: list[str],
    *,
    order: Order = "time",
    profile_order: list[str] | None = None,
    pressure_step: float = 10.0,
    max_pressure: float | None = None,
    horizontal_step_km: float | None = None,
) -> xr.Dataset:
    """Grid profiles into a distance x pressure section."""
    missing = [v for v in variables if v not in table]
    if missing:
        raise OceanToolError("bad_input", f"profiles lack {', '.join(missing)}")
    heads = table.groupby("profile", sort=False).first().reset_index()
    if order == "given":
        if not profile_order:
            raise OceanToolError("bad_input", "order 'given' needs profile_order")
        heads = heads.set_index("profile").loc[profile_order].reset_index()
    else:
        heads = heads.sort_values(order)
    if len(heads) < 2:
        raise OceanToolError("no_data", "a section needs at least two profiles")
    distance = along_track_km(
        heads["longitude"].to_numpy(), heads["latitude"].to_numpy()
    )
    top = max_pressure or float(table["pressure"].max())
    levels = np.arange(0.0, top + pressure_step / 2, pressure_step)
    grids = {v: np.full((len(heads), len(levels)), np.nan) for v in variables}
    for i, profile in enumerate(heads["profile"]):
        group = (
            table[table["profile"] == profile]
            .sort_values("pressure")
            .drop_duplicates("pressure")
        )
        p = group["pressure"].to_numpy()
        for v in variables:
            valid = group[v].notna().to_numpy()
            if valid.sum() >= 2:
                grids[v][i] = np.interp(
                    levels,
                    p[valid],
                    group[v].to_numpy()[valid],
                    left=np.nan,
                    right=np.nan,
                )
    ds = xr.Dataset(
        {v: (("distance", "pressure"), grids[v]) for v in variables},
        coords={
            "distance": (
                "distance",
                distance,
                {"units": "km", "long_name": "along-track distance"},
            ),
            "pressure": ("pressure", levels, {"units": "dbar", "positive": "down"}),
            "latitude": ("distance", heads["latitude"].to_numpy(float)),
            "longitude": ("distance", heads["longitude"].to_numpy(float)),
            "profile": ("distance", heads["profile"].astype(str).to_numpy()),
        },
    )
    if horizontal_step_km:
        ds = regrid_distance(ds, horizontal_step_km)
    ds.attrs["path"] = json.dumps(
        path_geojson(heads["longitude"].to_numpy(), heads["latitude"].to_numpy())
    )
    ds.attrs["source"] = "profiles"
    ds.attrs["vertical"] = "pressure"
    return ds


def regrid_distance(ds: xr.Dataset, step_km: float) -> xr.Dataset:
    """Linear interpolation between stations onto a regular distance axis.

    Station positions stay as ``station_distance`` so the view can mark them.
    """
    stations = ds["distance"].to_numpy()
    target = np.arange(stations[0], stations[-1] + step_km / 2, step_km)
    out = ds.drop_vars(["latitude", "longitude", "profile"]).interp(distance=target)
    out = out.assign_coords(
        latitude=("distance", np.interp(target, stations, ds["latitude"].to_numpy())),
        longitude=("distance", np.interp(target, stations, ds["longitude"].to_numpy())),
    )
    out["station_distance"] = ("station", stations)
    out["station_profile"] = ("station", ds["profile"].to_numpy())
    out.attrs.update(ds.attrs)
    return out


def _coord_names(ds: xr.Dataset) -> tuple[str, str, str | None, str | None]:
    """(lon, lat, vertical, time) coordinate names in a gridded dataset."""

    def pick(options: tuple[str, ...]) -> str | None:
        for name in options:
            if name in ds.coords or name in ds.dims:
                return name
        return None

    lon = pick(("longitude", "lon", "xh", "xq", "nav_lon", "x"))
    lat = pick(("latitude", "lat", "yh", "yq", "nav_lat", "y"))
    vertical = pick(
        (
            "depth",
            "deptht",
            "z_l",
            "zl",
            "lev",
            "level",
            "pressure",
            "isobaricInhPa",
            "z",
        )
    )
    time = pick(("time", "valid_time", "Time"))
    if lon is None or lat is None:
        raise OceanToolError(
            "bad_input", "gridded field needs longitude and latitude coordinates"
        )
    return lon, lat, vertical, time


def densify_path(
    vertices: list[list[float]], step_km: float
) -> tuple[np.ndarray, np.ndarray]:
    """Points every ``step_km`` along a polyline (linear in lon/lat per leg)."""
    if len(vertices) < 2:
        raise OceanToolError(
            "bad_input", "a path needs at least two [lon, lat] vertices"
        )
    lons: list[float] = []
    lats: list[float] = []
    for (x0, y0), (x1, y1) in itertools.pairwise(vertices):
        leg_km = float(along_track_km(np.array([x0, x1]), np.array([y0, y1]))[-1])
        n = max(int(np.ceil(leg_km / step_km)), 1)
        frac = np.linspace(0, 1, n, endpoint=False)
        lons.extend(x0 + frac * (x1 - x0))
        lats.extend(y0 + frac * (y1 - y0))
    lons.append(vertices[-1][0])
    lats.append(vertices[-1][1])
    return np.asarray(lons, float), np.asarray(lats, float)


def section_from_grid(
    ds: xr.Dataset,
    variables: list[str],
    path: list[list[float]],
    *,
    step_km: float,
    time: str | None = None,
) -> xr.Dataset:
    """Sample gridded variables along a path (bilinear in lon/lat)."""
    missing = [v for v in variables if v not in ds]
    if missing:
        raise OceanToolError(
            "bad_input",
            f"dataset lacks {', '.join(missing)}; has {', '.join(map(str, ds.data_vars))}",
        )
    lon_name, lat_name, vertical, time_name = _coord_names(ds)
    data = ds[variables]
    if time and time_name:
        data = data.sel({time_name: np.datetime64(time)}, method="nearest")
    lon, lat = densify_path(path, step_km)
    lon_values = ds[lon_name].to_numpy()
    if lon_values.max() > 180 and lon.min() < 0:
        lon = np.where(lon < 0, lon + 360, lon)
    points = xr.Dataset({lon_name: ("distance", lon), lat_name: ("distance", lat)})
    sampled = data.interp({lon_name: points[lon_name], lat_name: points[lat_name]})
    distance = along_track_km(lon, lat)
    sampled = sampled.assign_coords(
        distance=(
            "distance",
            distance,
            {"units": "km", "long_name": "along-path distance"},
        ),
        latitude=("distance", lat),
        longitude=("distance", lon),
    )
    drop = [
        c
        for c in (lon_name, lat_name)
        if c in sampled.coords and c not in ("latitude", "longitude")
    ]
    sampled = sampled.drop_vars(drop, errors="ignore")
    sampled.attrs["path"] = json.dumps({"type": "LineString", "coordinates": path})
    sampled.attrs["source"] = "grid"
    sampled.attrs["vertical"] = vertical or ""
    if vertical:
        order = ["distance", vertical] + [
            d for d in sampled.dims if d not in ("distance", vertical)
        ]
        sampled = sampled.transpose(*order)
    return sampled


def section_summary(ds: xr.Dataset) -> dict[str, Any]:
    vertical = ds.attrs.get("vertical") or ""
    out: dict[str, Any] = {
        "columns": int(ds.sizes["distance"]),
        "length_km": float(ds["distance"].max() - ds["distance"].min()),
        "variables": {},
    }
    if vertical and vertical in ds.dims:
        out["levels"] = int(ds.sizes[vertical])
        out["vertical"] = vertical
    for name, da in ds.data_vars.items():
        if "distance" not in da.dims:
            continue
        values = da.to_numpy()
        finite = values[np.isfinite(values)]
        out["variables"][str(name)] = (
            {
                "min": float(finite.min()),
                "max": float(finite.max()),
                "valid_fraction": float(finite.size / values.size),
            }
            if finite.size
            else {"valid_fraction": 0.0}
        )
    return out
