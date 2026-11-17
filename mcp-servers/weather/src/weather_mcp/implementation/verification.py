"""Verification: gridded scores by lead time, station verification, track errors.

Gridded scores interpolate the reference (for example ERA5) bilinearly onto
the forecast grid at each valid time; RMSE and bias are cos(latitude)
weighted over the region; ACC uses anomalies from a climatology the caller
gives; CRPS for ensembles is the fair estimator (Ferro 2014):
mean|X - y| - 1/(2 m (m - 1)) sum|X_i - X_j|.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .common import WeatherToolError
from .grids import LAT, LON, MEMBER, pick, select_box

LEAD = ("lead_time", "step", "prediction_timedelta", "lead")
INIT = ("time", "init_time", "forecast_reference_time")


def _lead_and_init(da: xr.DataArray) -> tuple[str, str | None]:
    lead = pick(da, LEAD)
    init = next((n for n in INIT if n in da.coords and n != lead), None)
    return str(lead), init


def _weights(da: xr.DataArray) -> xr.DataArray:
    lat = pick(da, LAT)
    return xr.DataArray(
        np.cos(np.deg2rad(da[lat].to_numpy())), coords={lat: da[lat]}, dims=[lat]
    )


def fair_crps(ensemble: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Fair CRPS over axis 0 (members)."""
    m = ensemble.shape[0]
    term1 = np.nanmean(np.abs(ensemble - obs[None]), axis=0)
    diffs = np.abs(ensemble[:, None] - ensemble[None, :]).sum(axis=(0, 1))
    return term1 - diffs / (2 * m * (m - 1))


def score(
    forecast: xr.Dataset,
    reference: xr.Dataset,
    *,
    variable: str,
    reference_variable: str,
    box: list[float] | None,
    climatology: xr.DataArray | None,
    map_lead_hours: float | None,
) -> tuple[pd.DataFrame, xr.Dataset | None, dict[str, Any]]:
    """Scores per lead time (and per initialisation, averaged)."""
    fc = forecast[variable]
    if box:
        fc = select_box(fc, box)
    lead, init = _lead_and_init(fc)
    member = pick(fc, MEMBER, required=False)
    lon, lat = pick(fc, LON), pick(fc, LAT)
    ref = reference[reference_variable]
    rlon, rlat = pick(ref, LON), pick(ref, LAT)
    rtime = pick(ref, ("time", "valid_time"))
    rows = []
    error_map = None
    inits: list[Any]
    if init and init in fc.dims:
        inits = list(fc[init].to_numpy())
    elif init:
        inits = [fc[init].to_numpy()[()]]
    else:
        inits = [None]
    for t0 in inits:
        f0 = fc.sel({init: t0}) if init and init in fc.dims else fc
        for step in f0[lead].to_numpy():
            f = f0.sel({lead: step})
            valid = (
                pd.Timestamp(t0) + pd.Timedelta(step)
                if t0 is not None
                else pd.Timestamp(f["valid_time"].to_numpy())
            )
            try:
                r = ref.sel(
                    {
                        rtime: np.datetime64(
                            valid.tz_localize(None) if valid.tzinfo else valid
                        )
                    }
                )
            except KeyError:
                continue
            r = r.interp({rlon: f[lon], rlat: f[lat]})
            w = _weights(f)
            mean_f = f.mean(member) if member else f
            err = mean_f - r
            record: dict[str, Any] = {
                "init": pd.Timestamp(t0).isoformat() if t0 is not None else None,
                "lead_hours": pd.Timedelta(step).total_seconds() / 3600,
                "valid": valid.isoformat(),
                "rmse": float(np.sqrt((err**2).weighted(w).mean())),
                "bias": float(err.weighted(w).mean()),
            }
            if climatology is not None:
                clim = climatology.interp(
                    {pick(climatology, LON): f[lon], pick(climatology, LAT): f[lat]}
                )
                if "dayofyear" in clim.dims:
                    clim = clim.sel(dayofyear=valid.dayofyear)
                fa, ra = mean_f - clim, r - clim
                num = (fa * ra).weighted(w).mean()
                den = np.sqrt((fa**2).weighted(w).mean() * (ra**2).weighted(w).mean())
                record["acc"] = float(num / den)
            if member:
                crps = fair_crps(
                    f.transpose(member, ...).to_numpy(),
                    r.transpose(*[d for d in f.dims if d != member]).to_numpy(),
                )
                record["crps"] = float(
                    xr.DataArray(crps, coords=r.coords, dims=r.dims).weighted(w).mean()
                )
                record["spread"] = float(np.sqrt(f.var(member).weighted(w).mean()))
            rows.append(record)
            if (
                map_lead_hours is not None
                and abs(record["lead_hours"] - map_lead_hours) < 1e-6
                and error_map is None
            ):
                error_map = err.rename("error").to_dataset()
                error_map.attrs.update(
                    {"lead_hours": map_lead_hours, "valid": valid.isoformat()}
                )
    if not rows:
        raise WeatherToolError(
            "no_data", "no forecast valid time is present in the reference"
        )
    table = pd.DataFrame(rows)
    by_lead = (
        table.groupby("lead_hours")[
            [c for c in ("rmse", "bias", "acc", "crps", "spread") if c in table]
        ]
        .mean()
        .reset_index()
    )
    numbers = {
        "initialisations": int(table["init"].nunique()),
        "leads": len(by_lead),
        "by_lead": by_lead.to_dict("records"),
        "interpolation": "reference bilinear onto the forecast grid",
    }
    return table, error_map, numbers


def station_verify(
    forecast: xr.Dataset,
    observations: pd.DataFrame,
    *,
    variable: str,
    observed: str,
    offset: float,
    tolerance_minutes: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Forecast at each station (bilinear) against the nearest observation in time."""
    fc = forecast[variable] + offset
    lead, init = _lead_and_init(fc)
    lon, lat = pick(fc, LON), pick(fc, LAT)
    obs = observations.dropna(subset=[observed]).copy()
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    rows = []
    for station, group in obs.groupby("station"):
        x, y = float(group["longitude"].iloc[0]), float(group["latitude"].iloc[0])
        if float(fc[lon].max()) > 180 and x < 0:
            x += 360
        at = fc.interp({lon: x, lat: y})
        series = group.set_index("time")[observed].sort_index()
        inits = (
            at[init].to_numpy() if init and init in at.dims else [at[init].to_numpy()]
        )
        for t0 in np.atleast_1d(inits):
            a = at.sel({init: t0}) if init and init in at.dims else at
            for step in a[lead].to_numpy():
                valid = pd.Timestamp(t0, tz="UTC") + pd.Timedelta(step)
                pos = series.index.get_indexer([valid], method="nearest")[0]
                if pos < 0 or abs(series.index[pos] - valid) > pd.Timedelta(
                    minutes=tolerance_minutes
                ):
                    continue
                f = float(a.sel({lead: step}))
                rows.append(
                    {
                        "station": station,
                        "init": pd.Timestamp(t0).isoformat(),
                        "lead_hours": pd.Timedelta(step).total_seconds() / 3600,
                        "valid": valid.isoformat(),
                        "forecast": f,
                        "observed": float(series.iloc[pos]),
                        "error": f - float(series.iloc[pos]),
                    }
                )
    if not rows:
        raise WeatherToolError(
            "no_data",
            "no observation lies within the tolerance of any forecast valid time",
        )
    pairs = pd.DataFrame(rows)
    stats = (
        pairs.groupby(["station", "lead_hours"])["error"]
        .agg(
            bias="mean",
            rmse=lambda e: float(np.sqrt(np.mean(e**2))),
            mae=lambda e: float(np.mean(np.abs(e))),
            count="size",
        )
        .reset_index()
    )
    by_lead = (
        pairs.groupby("lead_hours")["error"]
        .agg(bias="mean", rmse=lambda e: float(np.sqrt(np.mean(e**2))), count="size")
        .reset_index()
    )
    numbers = {
        "pairs": len(pairs),
        "stations": int(pairs["station"].nunique()),
        "by_lead": by_lead.to_dict("records"),
        "interpolation": "bilinear to the station; nearest observation in time",
    }
    return pairs, stats, numbers


def great_circle_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dl = np.radians(lon2 - lon1)
    c = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * 6371.0 * np.arcsin(np.sqrt(np.clip(c, 0, 1)))


def track_score(
    forecasts: pd.DataFrame, observed: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Position, pressure and wind errors of forecast tracks against the best track.

    ``forecasts`` has model, time (valid), lead_hours, latitude, longitude and
    optionally mslp_hpa and wind_kt; the best track is interpolated linearly
    (longitude unwrapped) to each forecast time inside its life.
    """
    obs = observed.sort_values("time").copy()
    obs["time"] = pd.to_datetime(obs["time"], utc=True)
    epoch = pd.Timestamp("1970-01-01", tz="UTC")
    ot = (obs["time"] - epoch).dt.total_seconds().to_numpy()
    olon = np.degrees(np.unwrap(np.radians(obs["longitude"].to_numpy())))
    rows = []
    fc = forecasts.copy()
    fc["time"] = pd.to_datetime(fc["time"], utc=True)
    for _, row in fc.iterrows():
        t = (row["time"] - epoch).total_seconds()
        if t < ot[0] or t > ot[-1]:
            continue
        lat = float(np.interp(t, ot, obs["latitude"]))
        lon = float(np.interp(t, ot, olon))
        lon = ((lon + 180) % 360) - 180
        record = {
            "model": row["model"],
            "lead_hours": float(row["lead_hours"]),
            "time": row["time"].isoformat(),
            "position_error_km": float(
                great_circle_km(
                    np.array([lat]),
                    np.array([lon]),
                    np.array([row["latitude"]]),
                    np.array([((row["longitude"] + 180) % 360) - 180]),
                )[0]
            ),
        }
        if "mslp_hpa" in fc and obs["mslp_hpa"].notna().any():
            record["mslp_error_hpa"] = float(
                row["mslp_hpa"]
                - np.interp(t, ot[obs["mslp_hpa"].notna()], obs["mslp_hpa"].dropna())
            )
        if "wind_kt" in fc and obs["wind_kt"].notna().any():
            record["wind_error_kt"] = float(
                row["wind_kt"]
                - np.interp(t, ot[obs["wind_kt"].notna()], obs["wind_kt"].dropna())
            )
        rows.append(record)
    if not rows:
        raise WeatherToolError(
            "no_data", "no forecast point falls inside the best track's life"
        )
    table = pd.DataFrame(rows)
    cols = [
        c
        for c in ("position_error_km", "mslp_error_hpa", "wind_error_kt")
        if c in table
    ]
    by = table.groupby(["model", "lead_hours"])[cols].mean().reset_index()
    numbers = {
        "models": sorted(table["model"].unique().tolist()),
        "by_model_lead": by.to_dict("records"),
        "mean_position_error_km": table.groupby("model")["position_error_km"]
        .mean()
        .to_dict(),
    }
    return table, numbers
