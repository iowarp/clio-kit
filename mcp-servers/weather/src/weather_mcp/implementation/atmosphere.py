"""Cross-sections and soundings on pressure-level fields, with MetPy.

Sections follow a path of [lon, lat] vertices (each leg a geodesic, MetPy's
``cross_section``) and add potential and equivalent potential temperature
and the wind across and along the path. Soundings are profiles at points
with surface-based, mixed-layer (lowest 100 hPa) and most-unstable CAPE and
CIN, the lifted index and the LCL; the parcel path is returned so a skew-T
can draw it.
"""

from __future__ import annotations

import itertools
import json
from typing import Any

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import xarray as xr
from metpy.interpolate import cross_section as metpy_cross_section
from metpy.units import units

from .common import WeatherToolError
from .grids import LAT, LEVEL, LON, TIME, pick, to_dataset_lon


def _prepare(ds: xr.Dataset, time: str | None) -> xr.Dataset:
    t = pick(ds, TIME, required=False)
    if t and t in ds.dims:
        if time is None:
            raise WeatherToolError(
                "bad_input", "the field has several times; give the time to use"
            )
        ds = ds.sel({t: np.datetime64(time)}, method="nearest")
    lon, lat = pick(ds, LON), pick(ds, LAT)
    if lon != "longitude" or lat != "latitude":
        ds = ds.rename({lon: "longitude", lat: "latitude"})
    if ds.latitude[0] > ds.latitude[-1]:
        ds = ds.sortby("latitude")
    return ds.metpy.assign_crs(
        grid_mapping_name="latitude_longitude", earth_radius=6371229.0
    )


def cross_section(
    ds: xr.Dataset,
    path: list[list[float]],
    *,
    temperature: str,
    humidity: str | None,
    u: str | None,
    v: str | None,
    others: list[str],
    steps_per_leg: int,
    time: str | None,
) -> xr.Dataset:
    """Section along a multi-leg path, with theta, theta-e and normal/tangential wind."""
    ds = _prepare(ds, time)
    level = pick(ds, LEVEL)
    names = [n for n in (temperature, humidity, u, v, *others) if n]
    data = ds[names]
    legs: list[xr.Dataset] = []
    for (x0, y0), (x1, y1) in itertools.pairwise(path):
        start = (y0, to_dataset_lon(ds, x0))
        end = (y1, to_dataset_lon(ds, x1))
        leg = metpy_cross_section(
            data, start, end, steps=steps_per_leg, interp_type="linear"
        )
        legs.append(leg if not legs else leg.isel(index=slice(1, None)))
    section = xr.concat(legs, dim="index")
    lat = section["latitude"].to_numpy()
    lon = np.where(
        section["longitude"].to_numpy() > 180,
        section["longitude"].to_numpy() - 360,
        section["longitude"].to_numpy(),
    )
    step = (
        np.radians(
            np.hypot(
                np.diff(lat),
                np.diff(lon) * np.cos(np.radians((lat[:-1] + lat[1:]) / 2)),
            )
        )
        * 6371.229
    )
    distance = np.concatenate([[0.0], np.cumsum(step)])
    p = section[level] * units.hPa
    temp = section[temperature].metpy.quantify()
    out = xr.Dataset(
        {name: section[name].drop_vars("metpy_crs", errors="ignore") for name in names}
    )
    out["potential_temperature"] = (
        mpcalc.potential_temperature(p, temp)
        .metpy.dequantify()
        .drop_vars("metpy_crs", errors="ignore")
    )
    if humidity:
        q = section[humidity].metpy.quantify()
        td = mpcalc.dewpoint_from_specific_humidity(p, q)
        out["equivalent_potential_temperature"] = (
            mpcalc.equivalent_potential_temperature(p, temp, td)
            .metpy.dequantify()
            .drop_vars("metpy_crs", errors="ignore")
        )
    if u and v:
        heading = np.degrees(
            np.arctan2(np.gradient(lat), np.gradient(lon) * np.cos(np.radians(lat)))
        )
        tx, ty = np.cos(np.radians(heading)), np.sin(np.radians(heading))
        uu, vv = section[u], section[v]
        out["tangential_wind"] = uu * xr.DataArray(
            tx, dims="index"
        ) + vv * xr.DataArray(ty, dims="index")
        out["normal_wind"] = -uu * xr.DataArray(ty, dims="index") + vv * xr.DataArray(
            tx, dims="index"
        )
    out = out.assign_coords(
        distance=("index", distance, {"units": "km"}),
        latitude=("index", lat),
        longitude=("index", lon),
    )
    out = out.swap_dims({"index": "distance"}).drop_vars(
        ["index", "metpy_crs"], errors="ignore"
    )
    out = out.transpose("distance", level, ...)
    out.attrs["path"] = json.dumps({"type": "LineString", "coordinates": path})
    out.attrs["vertical"] = level
    return out


def soundings(
    ds: xr.Dataset,
    points: list[dict[str, Any]],
    *,
    temperature: str,
    humidity: str,
    u: str | None,
    v: str | None,
    time: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Profiles at points and their instability indices."""
    ds = _prepare(ds, time)
    level = pick(ds, LEVEL)
    profile_rows, index_rows = [], []
    for number, point in enumerate(points, start=1):
        name = str(point.get("name") or f"P{number}")
        col = ds.sel(
            longitude=to_dataset_lon(ds, float(point["longitude"])),
            latitude=float(point["latitude"]),
            method="nearest",
        )
        col = col.sortby(level, ascending=False)  # surface first
        p = col[level].to_numpy() * units.hPa
        t = col[temperature].metpy.quantify().metpy.unit_array.to("degC")
        q = col[humidity].metpy.quantify().metpy.unit_array
        td = mpcalc.dewpoint_from_specific_humidity(p, q).to("degC")
        valid = np.isfinite(t.m) & np.isfinite(td.m)
        if valid.sum() < 5:
            raise WeatherToolError("no_data", f"{name}: fewer than five valid levels")
        p, t, td = p[valid], t[valid], td[valid]
        parcel = mpcalc.parcel_profile(p, t[0], td[0]).to("degC")
        sb_cape, sb_cin = mpcalc.surface_based_cape_cin(p, t, td)
        ml_cape, ml_cin = mpcalc.mixed_layer_cape_cin(p, t, td, depth=100 * units.hPa)
        mu_cape, mu_cin = mpcalc.most_unstable_cape_cin(p, t, td)
        li = mpcalc.lifted_index(p, t, parcel)
        lcl_p, lcl_t = mpcalc.lcl(p[0], t[0], td[0])
        index_rows.append(
            {
                "point": name,
                "longitude": float(point["longitude"]),
                "latitude": float(point["latitude"]),
                "surface_pressure_hpa": float(p[0].m),
                "sbcape": float(sb_cape.m),
                "sbcin": float(sb_cin.m),
                "mlcape": float(ml_cape.m),
                "mlcin": float(ml_cin.m),
                "mucape": float(mu_cape.m),
                "mucin": float(mu_cin.m),
                "lifted_index": float(np.atleast_1d(li.m)[0]),
                "lcl_pressure_hpa": float(lcl_p.m),
                "lcl_temperature_c": float(lcl_t.m),
            }
        )
        uu = col[u].to_numpy()[valid] if u else np.full(len(p), np.nan)
        vv = col[v].to_numpy()[valid] if v else np.full(len(p), np.nan)
        profile_rows.append(
            pd.DataFrame(
                {
                    "profile": name,
                    "latitude": float(point["latitude"]),
                    "longitude": float(point["longitude"]),
                    "pressure": p.m,
                    "temperature_c": t.m,
                    "dewpoint_c": td.m,
                    "parcel_temperature_c": parcel.m,
                    "u_ms": uu,
                    "v_ms": vv,
                }
            )
        )
    return pd.concat(profile_rows, ignore_index=True), pd.DataFrame(index_rows)
