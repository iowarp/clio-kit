"""Lagrangian particles with Parcels 3 (RK4 advection, forward or backward).

The velocity field is any NetCDF/Zarr with surface (or chosen-depth) u and v
on a regular lon-lat grid, such as a GLORYS or altimetry-derived subset.
Particles are seeded on a disc around a station; trajectories come back as
a long table and the end positions as a density grid (where water came from
when run backward, where it went when run forward).
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .common import OceanToolError
from .sections import _coord_names


def seed_disc(
    lon: float, lat: float, radius_km: float, count: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """``count`` points spread uniformly over a disc (reproducible with ``seed``)."""
    rng = np.random.default_rng(seed)
    r = radius_km * np.sqrt(rng.random(count))
    theta = 2 * np.pi * rng.random(count)
    dlat = r * np.cos(theta) / 111.2
    dlon = r * np.sin(theta) / (111.2 * np.cos(np.deg2rad(lat)))
    return lon + dlon, lat + dlat


def run_particles(
    ds: xr.Dataset,
    *,
    u: str,
    v: str,
    depth: float | None,
    station: tuple[float, float],
    radius_km: float,
    count: int,
    days: float,
    direction: Literal["forward", "backward"],
    start: str | None,
    dt_minutes: float,
    output_hours: float,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Advect particles with Parcels; returns trajectories and numbers."""
    import parcels

    lon_name, lat_name, vertical, time_name = _coord_names(ds)
    if time_name is None:
        raise OceanToolError(
            "bad_input", "particles need a time-varying velocity field"
        )
    field = ds[[u, v]]
    if vertical and vertical in field.dims:
        field = field.sel(
            {vertical: depth if depth is not None else float(ds[vertical].min())},
            method="nearest",
        )
    field = field.fillna(0.0)
    lon_values = field[lon_name].to_numpy()
    station_lon = (
        station[0] + 360 if lon_values.max() > 180 and station[0] < 0 else station[0]
    )
    times = pd.to_datetime(field[time_name].to_numpy())
    begin = (
        pd.Timestamp(start)
        if start
        else (times[-1] if direction == "backward" else times[0])
    )
    span = pd.Timedelta(days=days)
    if direction == "backward" and begin - span < times[0]:
        raise OceanToolError(
            "out_of_range",
            f"the field starts {times[0]}; {days} days back from {begin} needs earlier data",
        )
    if direction == "forward" and begin + span > times[-1]:
        raise OceanToolError(
            "out_of_range",
            f"the field ends {times[-1]}; {days} days on from {begin} needs later data",
        )
    fieldset = parcels.FieldSet.from_xarray_dataset(
        field,
        variables={"U": u, "V": v},
        dimensions={"lon": lon_name, "lat": lat_name, "time": time_name},
        mesh="spherical",
        allow_time_extrapolation=False,
    )
    lons, lats = seed_disc(station_lon, station[1], radius_km, count, seed)
    release = np.full(
        count, np.datetime64(begin.tz_localize(None) if begin.tzinfo else begin)
    )
    pset = parcels.ParticleSet.from_list(
        fieldset=fieldset,
        pclass=parcels.ScipyParticle,
        lon=lons,
        lat=lats,
        time=release,
    )
    sign = -1 if direction == "backward" else 1
    with tempfile.TemporaryDirectory() as tmp:
        store = Path(tmp) / "trajectories.zarr"
        output = pset.ParticleFile(
            name=str(store), outputdt=pd.Timedelta(hours=output_hours).to_pytimedelta()
        )

        def delete_out_of_bounds(particle, fieldset, time):
            if particle.state >= 50:  # out of bounds / error states
                particle.delete()

        pset.execute(
            [parcels.AdvectionRK4, delete_out_of_bounds],
            runtime=span.to_pytimedelta(),
            dt=sign * pd.Timedelta(minutes=dt_minutes).to_pytimedelta(),
            output_file=output,
            verbose_progress=False,
        )
        traj = _read_trajectories(store)
    table = (
        traj[["lon", "lat", "time"]]
        .to_dataframe()
        .reset_index()
        .dropna(subset=["lon", "lat"])
    )
    table = table.rename(
        columns={"trajectory": "particle", "lon": "longitude", "lat": "latitude"}
    )
    table = table[["particle", "time", "longitude", "latitude"]]
    # The output file holds positions at each output interval; add every surviving
    # particle's final position so the end points are where the run actually ended.
    final_time = begin.tz_localize(None) if begin.tzinfo else begin
    final_time = final_time + sign * span
    final = pd.DataFrame(
        {
            "particle": [int(p.id) for p in pset],
            "time": final_time,
            "longitude": [float(p.lon) for p in pset],
            "latitude": [float(p.lat) for p in pset],
        }
    )
    table = pd.concat([table, final], ignore_index=True).drop_duplicates(
        ["particle", "time"], keep="last"
    )
    ends = (
        table.sort_values("time").groupby("particle").last()
        if direction == "forward"
        else table.sort_values("time").groupby("particle").first()
    )
    displacement = np.hypot(
        (ends["longitude"] - station_lon) * np.cos(np.deg2rad(station[1])) * 111.2,
        (ends["latitude"] - station[1]) * 111.2,
    )
    numbers = {
        "particles": count,
        "kept_to_end": len(ends),
        "direction": direction,
        "days": days,
        "release": begin.isoformat(),
        "median_displacement_km": float(np.median(displacement))
        if len(displacement)
        else float("nan"),
        "parcels_version": parcels.__version__,
        "integrator": "AdvectionRK4, ScipyParticle",
    }
    return table, numbers


def _read_trajectories(store: Path) -> xr.Dataset:
    """Read Parcels 3's zarr (v2) trajectory store with zarr directly.

    Parcels 3 writes zarr v2 stores; reading them through zarr keeps the
    server independent of which zarr format xarray's own backend expects.
    """
    import zarr

    group = zarr.open_group(str(store), mode="r")
    variables = {}
    for name in ("lon", "lat", "time"):
        array = group[name]
        dims = tuple(array.attrs.get("_ARRAY_DIMENSIONS", ("trajectory", "obs")))
        attrs = {k: v for k, v in array.attrs.items() if k != "_ARRAY_DIMENSIONS"}
        variables[name] = xr.Variable(dims, array[:], attrs)
    return xr.decode_cf(xr.Dataset(variables))


def end_density(table: pd.DataFrame, direction: str, step_degrees: float) -> xr.Dataset:
    """Fraction of particles ending (forward) or starting (backward) in each cell."""
    order = table.sort_values("time").groupby("particle")
    ends = order.last() if direction == "forward" else order.first()
    lon_edges = np.arange(
        np.floor(ends["longitude"].min() / step_degrees) * step_degrees,
        ends["longitude"].max() + step_degrees * 1.01,
        step_degrees,
    )
    lat_edges = np.arange(
        np.floor(ends["latitude"].min() / step_degrees) * step_degrees,
        ends["latitude"].max() + step_degrees * 1.01,
        step_degrees,
    )
    hist, _, _ = np.histogram2d(
        ends["latitude"], ends["longitude"], bins=[lat_edges, lon_edges]
    )
    return xr.Dataset(
        {"fraction": (("latitude", "longitude"), hist / max(len(ends), 1))},
        coords={
            "latitude": (lat_edges[:-1] + lat_edges[1:]) / 2,
            "longitude": (lon_edges[:-1] + lon_edges[1:]) / 2,
        },
        attrs={"meaning": "origins" if direction == "backward" else "destinations"},
    )
