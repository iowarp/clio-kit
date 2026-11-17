"""Ensemble exceedance probabilities and threshold events on series.

Exceedance: accumulated fields (precipitation in ECMWF ENS is accumulated
since the run start, in metres) are differenced into daily or windowed
amounts per member before counting the members above a threshold; the
probability is that count over the member count.

Threshold events: a calendar-day percentile threshold from a stated baseline
(window around each day of year), events of at least ``min_duration`` days
above it; the definition is the caller's and is returned with the result.
"""

from __future__ import annotations

import itertools
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from .common import WeatherToolError
from .grids import MEMBER, pick


def _valid_times(da: xr.DataArray) -> pd.DatetimeIndex:
    if "valid_time" in da.coords:
        return pd.DatetimeIndex(np.atleast_1d(da["valid_time"].to_numpy()))
    if "time" in da.coords and "step" in da.coords:
        return pd.DatetimeIndex(
            np.atleast_1d(da["time"].to_numpy() + da["step"].to_numpy())
        )
    raise WeatherToolError(
        "bad_input", "ensemble field needs valid_time, or time and step"
    )


def exceedance(
    ds: xr.Dataset,
    *,
    variable: str,
    thresholds: list[float],
    accumulated: bool,
    scale: float,
    window_hours: int,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Per window: each member's amount and the fraction above each threshold.

    ``scale`` converts units (1000 for metres to mm). Windows are consecutive
    ``window_hours`` blocks from the first valid time; an accumulated field's
    amount in a window is its value at the window end minus at the start.
    """
    da = ds[variable] * scale
    member = pick(da, MEMBER)
    if "step" not in da.dims:
        raise WeatherToolError("bad_input", "ensemble field needs a step dimension")
    extra = [d for d in da.dims if d not in (member, "step")]
    if extra:
        raise WeatherToolError(
            "bad_input", f"cut the field to a point first (extra dims {extra})"
        )
    da = da.transpose(member, "step")
    times = _valid_times(da)
    values = da.to_numpy()
    start = times[0]
    edges = pd.date_range(start, times[-1], freq=f"{window_hours}h")
    rows: list[dict[str, Any]] = []
    plume: list[dict[str, Any]] = []
    for t0, t1 in itertools.pairwise(edges):
        if t0 not in times or t1 not in times:
            continue
        i0, i1 = times.get_loc(t0), times.get_loc(t1)
        amount = (
            values[:, i1] - values[:, i0]
            if accumulated
            else values[:, i0 : i1 + 1].mean(axis=1)
        )
        record: dict[str, Any] = {
            "window_start": t0,
            "window_end": t1,
            "members": int(np.isfinite(amount).sum()),
            "mean": float(np.nanmean(amount)),
            "p10": float(np.nanpercentile(amount, 10)),
            "median": float(np.nanmedian(amount)),
            "p90": float(np.nanpercentile(amount, 90)),
            "max": float(np.nanmax(amount)),
        }
        for thr in thresholds:
            record[f"p_gt_{thr:g}"] = float(np.mean(amount > thr))
        rows.append(record)
        plume.extend(
            {"member": int(m), "window_end": t1, "amount": float(a)}
            for m, a in zip(da[member].to_numpy(), amount, strict=True)
        )
    if not rows:
        raise WeatherToolError(
            "no_data", f"no {window_hours}-hour window fits the forecast steps"
        )
    table = pd.DataFrame(rows)
    numbers: dict[str, Any] = {
        "windows": len(table),
        "members": int(table["members"].max()),
        "window_hours": window_hours,
        "accumulated": accumulated,
    }
    total = None
    if accumulated:
        # Over the whole period covered by windows: each member's total amount.
        last = times.get_loc(table["window_end"].iloc[-1])
        total = (
            values[:, last] - values[:, times.get_loc(table["window_start"].iloc[0])]
        )
        numbers["period"] = [
            str(table["window_start"].iloc[0]),
            str(table["window_end"].iloc[-1]),
        ]
    for thr in thresholds:
        col = f"p_gt_{thr:g}"
        numbers[f"max_probability_gt_{thr:g}"] = float(table[col].max())
        if total is not None:
            numbers[f"probability_total_gt_{thr:g}"] = float(np.mean(total > thr))
    return table, pd.DataFrame(plume), numbers


def daily_series(
    series: pd.Series, statistic: Literal["mean", "max", "min", "sum"]
) -> pd.Series:
    index = (
        series.index.tz_convert(None) if series.index.tz is not None else series.index
    )
    s = pd.Series(series.to_numpy(), index=index)
    return getattr(s.resample("D"), statistic)()


def threshold_climatology(
    daily: pd.Series, *, baseline: tuple[int, int], percentile: float, window_days: int
) -> pd.DataFrame:
    """Calendar-day mean and percentile over the baseline (window around each day)."""
    base = daily[(daily.index.year >= baseline[0]) & (daily.index.year <= baseline[1])]
    if base.notna().sum() < 365 * 3:
        raise WeatherToolError(
            "no_data",
            f"baseline {baseline[0]}-{baseline[1]} has fewer than three years of data",
        )
    doy = (
        base.index.dayofyear
        + ((~base.index.is_leap_year) & (base.index.month > 2)).astype(int)
    ).to_numpy()
    values = base.to_numpy()
    half = window_days // 2
    mean, thresh = np.full(366, np.nan), np.full(366, np.nan)
    for d in range(1, 367):
        near = np.abs(((doy - d + 183) % 366) - 183) <= half
        sample = values[near]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            mean[d - 1] = sample.mean()
            thresh[d - 1] = np.percentile(sample, percentile)
    all_doy = (
        daily.index.dayofyear
        + ((~daily.index.is_leap_year) & (daily.index.month > 2)).astype(int)
    ).to_numpy()
    return pd.DataFrame(
        {
            "value": daily.to_numpy(),
            "climatology": mean[all_doy - 1],
            "threshold": thresh[all_doy - 1],
        },
        index=daily.index.rename("time"),
    )


def threshold_events(
    frame: pd.DataFrame, *, min_duration: int, max_gap: int
) -> pd.DataFrame:
    above = (frame["value"] > frame["threshold"]).fillna(False).to_numpy()
    runs: list[list[int]] = []
    start = None
    for i, flag in enumerate(np.append(above, False)):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            runs.append([start, i - 1])
            start = None
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_duration]
    joined: list[list[int]] = []
    for run in runs:
        if joined and run[0] - joined[-1][1] - 1 <= max_gap:
            joined[-1][1] = run[1]
        else:
            joined.append(run)
    rows = []
    for number, (a, b) in enumerate(joined, start=1):
        seg = frame.iloc[a : b + 1]
        excess = seg["value"] - seg["threshold"]
        anomaly = seg["value"] - seg["climatology"]
        rows.append(
            {
                "event": number,
                "start": seg.index[0],
                "end": seg.index[-1],
                "duration_days": len(seg),
                "peak_day": seg["value"].idxmax(),
                "peak_value": float(seg["value"].max()),
                "mean_anomaly": float(anomaly.mean()),
                "max_excess_over_threshold": float(excess.max()),
                "cumulative_anomaly": float(anomaly.sum()),
            }
        )
    return pd.DataFrame(rows)
