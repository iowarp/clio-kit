"""Checks on model runs: spin-up drift, and the model against observations."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .common import OceanToolError
from .sections import _coord_names


def drift(
    ds: xr.Dataset,
    *,
    temperature: str,
    salinity: str | None,
    u: str | None,
    v: str | None,
    window: int,
    tolerance: dict[str, float],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Domain means per level and time, and an equilibration verdict per level.

    A level is judged equilibrated when the linear drift over the last
    ``window`` time steps, extrapolated across that window, is smaller than
    the stated tolerance for that quantity (for example 0.01 °C for
    temperature). The verdict names the criterion it used.
    """
    lon, lat, vertical, time = _coord_names(ds)
    if time is None:
        raise OceanToolError("bad_input", "drift needs a time dimension")
    weights = (
        xr.DataArray(np.cos(np.deg2rad(ds[lat].to_numpy())), coords=ds[lat].coords)
        if lat in ds.coords
        else None
    )
    series: dict[str, xr.DataArray] = {}
    for name in [n for n in (temperature, salinity) if n]:
        da = ds[name]
        series[name] = (
            da.weighted(weights.fillna(0)).mean((lon, lat))
            if weights is not None
            else da.mean((lon, lat))
        )
    if u and v:
        ke = 0.5 * (ds[u] ** 2 + ds[v] ** 2)
        # u and v may sit on different (staggered) grids; average each over its own horizontal dims.
        horizontal = [d for d in ke.dims if d not in (vertical, time)]
        series["kinetic_energy"] = ke.mean(horizontal)
    frame = xr.Dataset(series).to_dataframe().reset_index()
    verdicts = []
    groups = (
        frame.groupby(vertical) if vertical and vertical in frame else [(None, frame)]
    )
    for level, group in groups:
        tail = group.sort_values(time).tail(window)
        steps = np.arange(len(tail), dtype=float)
        for quantity in series:
            values = tail[quantity].to_numpy(float)
            if len(values) < 3 or not np.isfinite(values).all():
                continue
            slope = np.polyfit(steps, values, 1)[0]
            change = abs(slope) * (len(values) - 1)
            limit = tolerance.get(quantity, tolerance.get("default", np.inf))
            verdicts.append(
                {
                    "level": level,
                    "quantity": quantity,
                    "change_over_window": float(change),
                    "tolerance": float(limit),
                    "equilibrated": bool(change < limit),
                }
            )
    return frame, pd.DataFrame(verdicts)


def colocate(
    ds: xr.Dataset,
    observations: pd.DataFrame,
    *,
    model_variable: str,
    observed_variable: str,
    depth_bands: list[float],
    region_step_degrees: float,
    max_time_offset_days: float,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Model at each observation's position, depth and time; bias and RMSE.

    Observations need ``time``, ``latitude``, ``longitude``, ``depth`` (m) or
    ``pressure`` (dbar, used as depth to within 1 %), and the observed
    variable. Observations further than ``max_time_offset_days`` from any model
    time are dropped and counted.
    """
    lon, lat, vertical, time = _coord_names(ds)
    if vertical is None or time is None:
        raise OceanToolError(
            "bad_input", "colocation needs depth and time in the model output"
        )
    obs = observations.dropna(subset=[observed_variable]).copy()
    if "depth" not in obs:
        obs["depth"] = obs["pressure"]
    obs["time"] = pd.to_datetime(obs["time"], utc=True).dt.tz_localize(None)
    model_times = pd.to_datetime(ds[time].to_numpy())
    nearest = np.abs(
        model_times.to_numpy()[None, :] - obs["time"].to_numpy()[:, None]
    ).min(axis=1)
    keep = nearest <= np.timedelta64(int(max_time_offset_days * 86400), "s")
    dropped = int((~keep).sum())
    obs = obs[keep]
    if obs.empty:
        raise OceanToolError(
            "no_data", "no observation falls within the model's time span"
        )
    query_lon = obs["longitude"].to_numpy()
    if float(ds[lon].max()) > 180:
        query_lon = np.where(query_lon < 0, query_lon + 360, query_lon)
    sampled = ds[model_variable].interp(
        {
            lon: xr.DataArray(query_lon, dims="obs"),
            lat: xr.DataArray(obs["latitude"].to_numpy(), dims="obs"),
            vertical: xr.DataArray(obs["depth"].to_numpy(), dims="obs"),
            time: xr.DataArray(obs["time"].to_numpy(), dims="obs"),
        }
    )
    obs["model"] = sampled.to_numpy()
    obs["difference"] = obs["model"] - obs[observed_variable]
    obs = obs.dropna(subset=["model"])
    bands = np.asarray(depth_bands, float)
    obs["depth_band"] = pd.cut(obs["depth"], bands, right=False).astype(str)
    obs["region"] = (
        (np.floor(obs["latitude"] / region_step_degrees) * region_step_degrees).astype(
            str
        )
        + "N_"
        + (
            np.floor(obs["longitude"] / region_step_degrees) * region_step_degrees
        ).astype(str)
        + "E"
    )
    stats = (
        obs.groupby(["depth_band", "region"])["difference"]
        .agg(bias="mean", rmse=lambda d: float(np.sqrt(np.mean(d**2))), count="size")
        .reset_index()
    )
    by_band = obs.groupby("depth_band")["difference"].agg(
        bias="mean", rmse=lambda d: float(np.sqrt(np.mean(d**2))), count="size"
    )
    numbers = {
        "matched": len(obs),
        "dropped_outside_time": dropped,
        "overall_bias": float(obs["difference"].mean()),
        "overall_rmse": float(np.sqrt(np.mean(obs["difference"] ** 2))),
        "by_depth_band": {
            str(k): {kk: float(vv) for kk, vv in row.items()}
            for k, row in by_band.iterrows()
        },
        "worst_regions": stats.sort_values("rmse", ascending=False)
        .head(5)
        .to_dict("records"),
    }
    return obs, stats, numbers
