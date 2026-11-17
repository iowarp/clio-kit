"""Transport through a section, and the meridional overturning streamfunction.

Section transport samples u and v along a path on a regular lon-lat product
and integrates the velocity normal to the path. Because the path does not
follow the model's own cell faces, the result is an approximation; every
result says so (plan O4).

The overturning streamfunction uses the model's own grid: either a meridional
velocity with its cell widths, or a meridional transport the model already
wrote per cell (MOM6 ``vmo`` in kg s-1 or ``vh`` in m3 s-1). Depth space sums
from the surface down; density space bins the transport into density classes
and sums from light to dense.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
import xarray as xr

from .common import OceanToolError
from .sections import _coord_names, along_track_km, densify_path

EARTH_RADIUS = 6_371_000.0
RHO0 = 1025.0
CP = 3992.0  # J kg-1 K-1, near-surface seawater (TEOS-10 cp0 is 3991.87)
SV = 1.0e6


def _layer_thickness(depth: np.ndarray) -> np.ndarray:
    """Thickness of cells centred on ``depth`` (m), from mid-points."""
    depth = np.asarray(depth, float)
    edges = np.concatenate(
        [[0.0], (depth[:-1] + depth[1:]) / 2, [depth[-1] + (depth[-1] - depth[-2]) / 2]]
    )
    return np.diff(edges)


def section_transport(
    ds: xr.Dataset,
    path: list[list[float]],
    *,
    u: str,
    v: str,
    temperature: str | None,
    step_km: float,
    reference_temperature: float,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Volume (and heat) transport normal to a path, per time step.

    Positive transport is to the right of the direction of travel along the
    path (path drawn west to east: positive is southward; south to north:
    positive is eastward).
    """
    lon_name, lat_name, vertical, time_name = _coord_names(ds)
    if vertical is None:
        raise OceanToolError("bad_input", "transport needs a depth coordinate")
    lon, lat = densify_path(path, step_km)
    if ds[lon_name].max() > 180 and lon.min() < 0:
        lon = np.where(lon < 0, lon + 360, lon)
    where = {
        lon_name: xr.DataArray(lon, dims="distance"),
        lat_name: xr.DataArray(lat, dims="distance"),
    }
    names = [u, v] + ([temperature] if temperature else [])
    missing = [n for n in names if n not in ds]
    if missing:
        raise OceanToolError("bad_input", f"dataset lacks {', '.join(missing)}")
    sampled = ds[names].interp(where)
    # Unit tangent of each point's leg in local east/north metres.
    dlon = np.gradient(lon) * np.cos(np.deg2rad(lat))
    dlat = np.gradient(lat)
    norm = np.hypot(dlon, dlat)
    norm[norm == 0] = 1.0
    tx, ty = dlon / norm, dlat / norm
    normal = sampled[u] * xr.DataArray(ty, dims="distance") - sampled[v] * xr.DataArray(
        tx, dims="distance"
    )
    distance_km = along_track_km(lon, lat)
    ds_m = xr.DataArray(np.gradient(distance_km) * 1000.0, dims="distance")
    dz = xr.DataArray(
        _layer_thickness(ds[vertical].to_numpy()),
        dims=vertical,
        coords={vertical: ds[vertical]},
    )
    flux = (normal * ds_m * dz).fillna(0.0)
    volume = flux.sum(("distance", vertical)) / SV
    out = xr.Dataset({"volume_transport": volume.assign_attrs(units="Sv")})
    if temperature:
        heat = (
            RHO0 * CP * flux * (sampled[temperature] - reference_temperature)
        ).fillna(0.0)
        out["heat_transport"] = (heat.sum(("distance", vertical)) / 1e15).assign_attrs(
            units="PW", reference_temperature_degC=reference_temperature
        )
    normal_section = normal.assign_coords(
        distance=("distance", distance_km),
        latitude=("distance", lat),
        longitude=("distance", lon),
    )
    out["normal_velocity"] = normal_section.assign_attrs(
        units="m s-1", positive="right of path direction"
    )
    numbers: dict[str, Any] = {}
    series = out["volume_transport"]
    numbers["mean_volume_transport_sv"] = float(series.mean())
    if time_name and time_name in series.dims and series.sizes[time_name] > 1:
        numbers["std_volume_transport_sv"] = float(series.std())
        numbers["time_steps"] = int(series.sizes[time_name])
    if temperature:
        numbers["mean_heat_transport_pw"] = float(out["heat_transport"].mean())
    numbers["approximation"] = (
        "Velocities were interpolated onto the drawn path on a regular lon-lat product; the path "
        "does not follow the model's cell faces, so transport is approximate (no exact flux closure)."
    )
    return out, numbers


Space = Literal["depth", "density"]


def overturning(
    ds: xr.Dataset,
    *,
    transport: str | None,
    velocity: str | None,
    space: Space,
    density: str | None,
    density_bins: np.ndarray | None,
) -> xr.Dataset:
    """Overturning streamfunction psi(latitude, depth or density) in Sv.

    ``transport`` names a per-cell meridional transport (``vmo`` kg s-1 or
    ``vh`` m3 s-1); otherwise ``velocity`` names v (m s-1) and cell widths
    come from the longitude spacing and the layer thickness.
    """
    lon_name, lat_name, vertical, _time_name = _coord_names(ds)
    if vertical is None:
        raise OceanToolError("bad_input", "overturning needs a vertical coordinate")
    if transport:
        da = ds[transport]
        units = str(da.attrs.get("units", "")).replace(" ", "")
        if units in ("kgs-1", "kg/s", "kgs^-1") or transport == "vmo":
            cell = da / RHO0
        else:
            cell = da
    elif velocity:
        da = ds[velocity]
        lat = ds[lat_name]
        dlon = np.deg2rad(abs(float(ds[lon_name].diff(lon_name).median())))
        dx = EARTH_RADIUS * np.cos(np.deg2rad(lat)) * dlon
        dz = xr.DataArray(
            _layer_thickness(ds[vertical].to_numpy()),
            dims=vertical,
            coords={vertical: ds[vertical]},
        )
        cell = da * dx * dz
    else:
        raise OceanToolError(
            "bad_input", "give a transport variable (vmo/vh) or a velocity (v)"
        )
    cell = cell.fillna(0.0)
    zonal = cell.sum(lon_name)
    if space == "depth":
        psi = zonal.cumsum(vertical) / SV
        psi = psi.rename("psi").assign_attrs(
            units="Sv", long_name="overturning streamfunction (depth)"
        )
        return psi.to_dataset()
    if density is None or density_bins is None:
        raise OceanToolError(
            "bad_input", "density space needs a density variable and density_bins"
        )
    rho = ds[density]
    classes = []
    for lo, hi in itertools.pairwise(density_bins):
        mask = (rho >= lo) & (rho < hi)
        classes.append(cell.where(mask, 0.0).sum((lon_name, vertical)))
    binned = xr.concat(classes, dim="density").assign_coords(
        density=(density_bins[:-1] + density_bins[1:]) / 2
    )
    psi = binned.cumsum("density") / SV
    psi = psi.rename("psi").assign_attrs(
        units="Sv", long_name="overturning streamfunction (density)"
    )
    return psi.to_dataset()


def overturning_at(
    psi: xr.Dataset, latitude: float
) -> tuple[xr.DataArray, dict[str, Any]]:
    """Maximum of psi over the vertical at the nearest latitude, per time."""
    lat_name = next(
        (n for n in ("latitude", "lat", "yq", "yh", "y") if n in psi.dims), None
    )
    if lat_name is None:
        raise OceanToolError("bad_input", "streamfunction has no latitude dimension")
    at = psi["psi"].sel({lat_name: latitude}, method="nearest")
    vertical = next(d for d in at.dims if d not in ("time", "Time"))
    series = at.max(str(vertical))
    numbers = {
        "latitude_used": float(at[lat_name]),
        "mean_max_sv": float(series.mean()),
        "level_of_max": float(
            at.mean([d for d in at.dims if d != vertical]).idxmax(vertical)
        )
        if at.ndim > 1
        else float(at.idxmax(vertical)),
    }
    if series.ndim:
        numbers["std_max_sv"] = float(series.std())
    return series.rename("psi_max"), numbers
