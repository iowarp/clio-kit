"""Events on series: marine heatwaves (Hobday), bloom starts, heat budgets.

Marine heatwaves follow Hobday et al. (2016): a day-of-year climatology and
percentile threshold from a stated baseline (11-day window, 31-day smoothing
by default), events of at least ``min_duration`` days above the threshold,
gaps of up to ``max_gap`` days joined. Categories follow Hobday et al. (2018):
the peak intensity as a multiple of the threshold's distance from the
climatology (1-2 moderate, 2-3 strong, 3-4 severe, 4+ extreme).
"""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from .common import OceanToolError

CATEGORIES = ("Moderate", "Strong", "Severe", "Extreme")


def daily(series: pd.Series) -> pd.Series:
    """Daily means on a complete calendar (gaps stay NaN)."""
    index = (
        series.index.tz_convert(None) if series.index.tz is not None else series.index
    )
    s = pd.Series(series.to_numpy(), index=index)
    return s.resample("D").mean()


def climatology(
    series: pd.Series,
    *,
    baseline: tuple[int, int],
    percentile: float,
    window_days: int,
    smooth_days: int,
) -> pd.DataFrame:
    """Day-of-year mean and percentile over the baseline years (Hobday 2016)."""
    s = daily(series)
    base = s[(s.index.year >= baseline[0]) & (s.index.year <= baseline[1])]
    if base.notna().sum() < 365 * 3:
        raise OceanToolError(
            "no_data",
            f"baseline {baseline[0]}-{baseline[1]} holds fewer than three years of data; "
            "Hobday et al. recommend at least 30",
        )
    # Day of year on a 366-day calendar so 29 Feb has its own slot.
    doy = base.index.day_of_year + (
        (~base.index.is_leap_year) & (base.index.month > 2)
    ).astype(int)
    values = base.to_numpy()
    half = window_days // 2
    seas, thresh = np.full(366, np.nan), np.full(366, np.nan)
    for d in range(1, 367):
        offsets = ((doy.to_numpy() - d + 183) % 366) - 183
        sample = values[np.abs(offsets) <= half]
        sample = sample[np.isfinite(sample)]
        if sample.size:
            seas[d - 1] = sample.mean()
            thresh[d - 1] = np.percentile(sample, percentile)

    def smooth(x: np.ndarray) -> np.ndarray:
        if smooth_days <= 1:
            return x
        padded = np.concatenate([x[-smooth_days:], x, x[:smooth_days]])
        kernel = np.ones(smooth_days) / smooth_days
        out = np.convolve(
            pd.Series(padded).interpolate().to_numpy(), kernel, mode="same"
        )
        return out[smooth_days:-smooth_days]

    seas, thresh = smooth(seas), smooth(thresh)
    all_doy = s.index.day_of_year + (
        (~s.index.is_leap_year) & (s.index.month > 2)
    ).astype(int)
    return pd.DataFrame(
        {
            "value": s.to_numpy(),
            "climatology": seas[all_doy.to_numpy() - 1],
            "threshold": thresh[all_doy.to_numpy() - 1],
        },
        index=s.index.rename("time"),
    )


def detect_events(
    frame: pd.DataFrame,
    *,
    min_duration: int,
    max_gap: int,
    tail: Literal["upper", "lower"] = "upper",
) -> pd.DataFrame:
    """Events from a value/climatology/threshold frame, with Hobday metrics."""
    above = (
        frame["value"] > frame["threshold"]
        if tail == "upper"
        else frame["value"] < frame["threshold"]
    )
    above = above.fillna(False).to_numpy()
    runs = []
    start = None
    for i, flag in enumerate(above):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            runs.append([start, i - 1])
            start = None
    if start is not None:
        runs.append([start, len(above) - 1])
    runs = [r for r in runs if r[1] - r[0] + 1 >= min_duration]
    joined: list[list[int]] = []
    for run in runs:
        if joined and run[0] - joined[-1][1] - 1 <= max_gap:
            joined[-1][1] = run[1]
        else:
            joined.append(run)
    sign = 1.0 if tail == "upper" else -1.0
    rows = []
    for number, (a, b) in enumerate(joined, start=1):
        seg = frame.iloc[a : b + 1]
        intensity = sign * (seg["value"] - seg["climatology"])
        spread = sign * (seg["threshold"] - seg["climatology"])
        peak = int(np.nanargmax(intensity.to_numpy()))
        ratio = (
            float(intensity.iloc[peak] / spread.iloc[peak])
            if spread.iloc[peak]
            else float("nan")
        )
        category = (
            CATEGORIES[min(max(int(np.floor(ratio)) - 1, 0), 3)]
            if np.isfinite(ratio)
            else "unknown"
        )
        onset_days = max(peak, 1)
        decline_days = max(len(seg) - 1 - peak, 1)
        rows.append(
            {
                "event": number,
                "start": seg.index[0],
                "end": seg.index[-1],
                "peak": seg.index[peak],
                "duration_days": len(seg),
                "intensity_max": float(intensity.iloc[peak]),
                "intensity_mean": float(intensity.mean()),
                "intensity_cumulative": float(intensity.sum()),
                "intensity_ratio": ratio,
                "category": category,
                "rate_onset_per_day": float(intensity.iloc[peak] / onset_days),
                "rate_decline_per_day": float(intensity.iloc[peak] / decline_days),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- blooms

BloomMethod = Literal["median_plus_5pct", "max_rate"]


def bloom_start(
    series: pd.Series, *, method: BloomMethod, season: tuple[int, int], smooth_days: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bloom initiation per year.

    Both methods look at months ``season[0]..season[1]`` and tie the start to
    that window's chlorophyll peak. ``median_plus_5pct`` (Siegel et al. 2002):
    the first day of the run above the year's median plus 5 % that leads into
    the peak; when that run is already under way on the window's first day the
    onset was not observed and no date is given. ``max_rate``: the day of the
    steepest increase of the smoothed log-chlorophyll before the peak.
    """
    s = daily(series).interpolate(limit=15)
    if smooth_days > 1:
        s = s.rolling(
            smooth_days, center=True, min_periods=max(smooth_days // 3, 1)
        ).mean()
    rows = []
    for year, group in s.groupby(s.index.year):
        months = group.index.month
        window = group[(months >= season[0]) & (months <= season[1])].dropna()
        if len(window) < 30:
            rows.append(
                {
                    "year": year,
                    "bloom_start": None,
                    "reason": "fewer than 30 valid days",
                }
            )
            continue
        peak = window.idxmax()
        rising = window[window.index <= peak]
        if method == "median_plus_5pct":
            level = float(group.median()) * 1.05
            below = rising[rising <= level]
            if rising.iloc[-1] <= level:
                start, reason = None, "never exceeded"
            elif below.empty:
                start = None
                reason = "above the threshold when the window opens; onset not observed"
            else:
                start = rising.index[rising.index > below.index[-1]][0]
                reason = ""
            rows.append(
                {
                    "year": year,
                    "bloom_start": start,
                    "peak": peak,
                    "threshold": level,
                    "reason": reason,
                }
            )
        else:
            rate = np.log(rising.clip(lower=1e-3)).diff().dropna()
            start = rate.idxmax() if len(rate) and rate.max() > 0 else None
            rows.append(
                {
                    "year": year,
                    "bloom_start": start,
                    "peak": peak,
                    "max_rate_per_day": float(rate.max()) if len(rate) else None,
                    "reason": "" if start is not None else "no rise before the peak",
                }
            )
    table = pd.DataFrame(rows)
    found = table.dropna(subset=["bloom_start"])
    numbers: dict[str, Any] = {"years": len(table), "years_with_start": len(found)}
    if len(found):
        doy = pd.to_datetime(found["bloom_start"]).dt.day_of_year
        numbers["mean_start_day_of_year"] = float(doy.mean())
        numbers["start_day_range"] = [int(doy.min()), int(doy.max())]
        if len(found) >= 5:
            slope = np.polyfit(found["year"].astype(float), doy.astype(float), 1)[0]
            numbers["start_trend_days_per_decade"] = float(slope * 10)
    return table, numbers


# ------------------------------------------------------------- heat budget

RHO0 = 1025.0
CP = 3992.0


def penetrating_fraction(
    depth: np.ndarray, r: float = 0.58, z1: float = 0.35, z2: float = 23.0
) -> np.ndarray:
    """Shortwave fraction below depth (Paulson & Simpson 1977, water type I)."""
    return r * np.exp(-depth / z1) + (1 - r) * np.exp(-depth / z2)


def mixed_layer_budget(
    frame: pd.DataFrame,
    *,
    temperature: str,
    net_flux: str,
    shortwave: str | None,
    mld: str,
    start: str,
    end: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """dT/dt = (Q_net - Q_pen(h)) / (rho cp h) + residual.

    ``net_flux`` is the net downward surface heat flux (W m-2); ``shortwave``
    (W m-2) if given lets the penetrating part leave the mixed layer. The
    residual holds advection, entrainment and diffusion together.
    """
    d = frame.resample("D").mean().loc[start:end]
    if len(d) < 3:
        raise OceanToolError("no_data", "the event window holds fewer than three days")
    h = d[mld].clip(lower=1.0)
    q = d[net_flux].copy()
    if shortwave:
        q = q - d[shortwave] * penetrating_fraction(h.to_numpy())
    seconds = 86400.0
    # Forward difference: the flux on day i drives the change from day i to i + 1.
    tendency = (d[temperature].shift(-1) - d[temperature]) / seconds
    forcing = (q / (RHO0 * CP * h)).where(tendency.notna())
    out = pd.DataFrame(
        {
            "temperature": d[temperature],
            "mld": h,
            "tendency_K_per_day": tendency * seconds,
            "surface_flux_K_per_day": forcing * seconds,
        }
    )
    out["residual_K_per_day"] = (
        out["tendency_K_per_day"] - out["surface_flux_K_per_day"]
    )
    for column in (
        "tendency_K_per_day",
        "surface_flux_K_per_day",
        "residual_K_per_day",
    ):
        out[column.replace("_per_day", "_cumulative")] = out[column].fillna(0).cumsum()
    total = float(out["tendency_K_cumulative"].iloc[-1])
    flux = float(out["surface_flux_K_cumulative"].iloc[-1])
    numbers = {
        "window": [str(d.index[0].date()), str(d.index[-1].date())],
        "temperature_change_K": total,
        "surface_flux_contribution_K": flux,
        "residual_contribution_K": total - flux,
        "surface_flux_fraction": flux / total if total else float("nan"),
        "shortwave_penetration": "Paulson & Simpson (1977) type I"
        if shortwave
        else "not removed",
    }
    return out.reset_index(names="time"), numbers
