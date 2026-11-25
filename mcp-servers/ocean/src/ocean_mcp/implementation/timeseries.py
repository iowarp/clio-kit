"""Tides, spectra and trends on time series (moorings, gauges, altimetry).

Series tables are long or wide CSV/Parquet/NetCDF with a ``time`` column;
the caller names the value columns. Results are tables written as Parquet.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from scipy import signal, stats

from .common import OceanToolError


def read_series(
    path: str | Path, columns: list[str], time_column: str = "time"
) -> pd.DataFrame:
    """Read a time series file into a time-indexed frame with ``columns``."""
    path = Path(path).expanduser()
    if not path.exists():
        raise OceanToolError("bad_input", f"{path} does not exist")
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        frame = pd.read_parquet(path)
    elif suffix in (".csv", ".txt"):
        frame = pd.read_csv(path)
    elif suffix in (".nc", ".nc4"):
        import xarray as xr

        with xr.open_dataset(path) as ds:
            missing = [c for c in columns if c not in ds]
            if missing:
                raise OceanToolError(
                    "bad_input", f"{path.name} lacks {', '.join(missing)}"
                )
            frame = ds[columns].to_dataframe().reset_index()
    else:
        raise OceanToolError(
            "bad_input", f"{path.name}: series are .parquet, .csv or .nc"
        )
    if time_column not in frame:
        raise OceanToolError("bad_input", f"{path.name} has no '{time_column}' column")
    missing = [c for c in columns if c not in frame]
    if missing:
        raise OceanToolError("bad_input", f"{path.name} lacks {', '.join(missing)}")
    frame[time_column] = pd.to_datetime(frame[time_column], utc=True, errors="coerce")
    frame = frame.dropna(subset=[time_column]).set_index(time_column).sort_index()
    return frame[columns].astype(float)


# ----------------------------------------------------------------- harmonics


def harmonic_analysis(
    series: pd.DataFrame,
    *,
    height: str | None,
    u: str | None,
    v: str | None,
    latitude: float,
    constituents: list[str] | str,
    method: Literal["ols", "robust"],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """UTide solve + reconstruct.

    Returns the constituent table (amplitude, phase and, for currents, the
    ellipse parameters), the observed/predicted/residual series, and numbers.
    """
    import utide

    time = series.index.tz_convert(None).to_pydatetime()
    values: list[np.ndarray]
    if height:
        values = [series[height].to_numpy()]
    elif u and v:
        values = [series[u].to_numpy(), series[v].to_numpy()]
    else:
        raise OceanToolError(
            "bad_input", "give a height column, or u and v columns for currents"
        )
    coef = utide.solve(
        time,
        *values,
        lat=latitude,
        constit=constituents,
        method=method,
        conf_int="linear",
        trend=False,
        verbose=False,
    )
    rec = utide.reconstruct(time, coef, verbose=False)
    names = [str(n) for n in coef["name"]]
    if height:
        table = pd.DataFrame(
            {
                "constituent": names,
                "amplitude": coef["A"],
                "amplitude_ci": coef["A_ci"],
                "phase_deg": coef["g"],
                "phase_ci": coef["g_ci"],
                "snr": coef["diagn"]["SNR"],
            }
        )
        predicted = rec["h"]
        observed = values[0]
        out = pd.DataFrame(
            {"time": series.index, "observed": observed, "predicted": predicted}
        )
        out["residual"] = out["observed"] - out["predicted"]
        variance = float(np.nanvar(observed))
        explained = (
            1 - float(np.nanvar(out["residual"])) / variance
            if variance
            else float("nan")
        )
    else:
        table = pd.DataFrame(
            {
                "constituent": names,
                "semi_major": coef["Lsmaj"],
                "semi_minor": coef["Lsmin"],
                "inclination_deg": coef["theta"],
                "phase_deg": coef["g"],
                "snr": coef["diagn"]["SNR"],
            }
        )
        out = pd.DataFrame(
            {
                "time": series.index,
                "u_observed": values[0],
                "v_observed": values[1],
                "u_predicted": rec["u"],
                "v_predicted": rec["v"],
            }
        )
        out["u_residual"] = out["u_observed"] - out["u_predicted"]
        out["v_residual"] = out["v_observed"] - out["v_predicted"]
        total = float(np.nanvar(values[0]) + np.nanvar(values[1]))
        resid = float(np.nanvar(out["u_residual"]) + np.nanvar(out["v_residual"]))
        explained = 1 - resid / total if total else float("nan")
    table = table.sort_values(table.columns[1], ascending=False).reset_index(drop=True)
    numbers = {
        "constituents_solved": len(table),
        "significant": int((table["snr"] >= 2).sum()),
        "variance_explained": explained,
        "record_days": float(
            (series.index[-1] - series.index[0]).total_seconds() / 86400
        ),
        "dominant": table["constituent"].head(5).tolist(),
    }
    return table, out, numbers


def compare_constituents(
    observed: pd.DataFrame, model: pd.DataFrame, constituents: list[str]
) -> pd.DataFrame:
    """Amplitude and phase differences, and the complex difference |Zo - Zm|."""
    left = observed.set_index("constituent")
    right = model.set_index("constituent")
    rows = []
    for name in constituents:
        if name not in left.index or name not in right.index:
            continue
        ao, go = float(left.loc[name, "amplitude"]), float(left.loc[name, "phase_deg"])
        am, gm = (
            float(right.loc[name, "amplitude"]),
            float(right.loc[name, "phase_deg"]),
        )
        dg = (gm - go + 180) % 360 - 180
        zo = ao * np.exp(1j * np.deg2rad(go))
        zm = am * np.exp(1j * np.deg2rad(gm))
        rows.append(
            {
                "constituent": name,
                "observed_amplitude": ao,
                "model_amplitude": am,
                "amplitude_difference": am - ao,
                "observed_phase_deg": go,
                "model_phase_deg": gm,
                "phase_difference_deg": dg,
                "complex_difference": float(abs(zo - zm)),
            }
        )
    return pd.DataFrame(rows)


# ------------------------------------------------------------------- spectra


def spectra(
    series: pd.DataFrame,
    *,
    u: str,
    v: str | None,
    segment_days: float,
    latitude: float | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Welch power spectrum, or rotary (clockwise/anticlockwise) spectra for u+iv.

    Frequencies are cycles per day. With ``latitude`` the inertial frequency
    is reported so the clockwise peak can be read against it.
    """
    dt = (
        float(series.index.to_series().diff().dt.total_seconds().median()) / 86400.0
    )  # days
    if not np.isfinite(dt) or dt <= 0:
        raise OceanToolError("bad_input", "series needs a regular time step")
    fs = 1.0 / dt
    nperseg = max(int(segment_days / dt), 16)
    if v is None:
        x = series[u].interpolate(limit_direction="both").to_numpy()
        freq, power = signal.welch(
            signal.detrend(x), fs=fs, nperseg=min(nperseg, len(x))
        )
        table = pd.DataFrame({"frequency_cpd": freq, "power": power})
    else:
        w = (series[u] + 1j * series[v]).interpolate(limit_direction="both").to_numpy()
        freq, power = signal.welch(
            signal.detrend(w),
            fs=fs,
            nperseg=min(nperseg, len(w)),
            return_onesided=False,
        )
        order = np.argsort(freq)
        freq, power = freq[order], power[order]
        positive = freq > 0
        negative = freq < 0
        cw_freq = -freq[negative][::-1]
        cw_power = power[negative][::-1]
        table = pd.DataFrame(
            {
                "frequency_cpd": freq[positive],
                # Positive frequencies of u+iv rotate anticlockwise, negative clockwise.
                "anticlockwise": power[positive],
                "clockwise": np.interp(freq[positive], cw_freq, cw_power),
            }
        )
    numbers: dict[str, Any] = {
        "sample_interval_hours": dt * 24,
        "segment_days": nperseg * dt,
    }
    if latitude is not None:
        omega = 7.2921e-5
        f = 2 * omega * np.sin(np.deg2rad(latitude))
        numbers["inertial_frequency_cpd"] = float(abs(f) * 86400 / (2 * np.pi))
    peaks = {}
    for column in [c for c in table.columns if c != "frequency_cpd"]:
        i = int(np.argmax(table[column].to_numpy()[1:])) + 1
        peaks[column] = float(table["frequency_cpd"].iloc[i])
    numbers["peak_frequency_cpd"] = peaks
    return table, numbers


# --------------------------------------------------------------------- trends


def trend_per_year(series: pd.Series) -> dict[str, float]:
    """OLS trend with its standard error, in units per year (annual cycle removed)."""
    series = series.dropna()
    if len(series) < 24:
        raise OceanToolError("no_data", "a trend needs at least 24 samples")
    years = (series.index - series.index[0]).total_seconds() / (365.25 * 86400)
    w = 2 * np.pi * np.asarray(years)
    design = np.column_stack([np.ones(len(w)), np.asarray(years), np.cos(w), np.sin(w)])
    coef, *_ = np.linalg.lstsq(design, series.to_numpy(), rcond=None)
    resid = series.to_numpy() - design @ coef
    dof = max(len(series) - design.shape[1], 1)
    cov = np.linalg.inv(design.T @ design) * (resid @ resid / dof)
    # Lag-1 autocorrelation inflates the error (effective sample size).
    r1 = float(pd.Series(resid).autocorr(1)) if len(resid) > 2 else 0.0
    inflation = np.sqrt((1 + r1) / (1 - r1)) if -1 < r1 < 1 else 1.0
    return {
        "trend_per_year": float(coef[1]),
        "stderr_per_year": float(np.sqrt(cov[1, 1]) * inflation),
        "lag1_autocorrelation": r1,
        "samples": len(series),
    }


def sea_level_decompose(
    gauge: pd.Series, altimetry: pd.Series, *, resample: str = "MS"
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trends of gauge and offshore altimetry over their common period.

    The difference (gauge minus altimetry) is the local part, which includes
    vertical land motion (a sinking gauge shows a larger relative rise) and
    any coastal signal altimetry does not see.
    """
    g = gauge.resample(resample).mean()
    a = altimetry.resample(resample).mean()
    joined = pd.concat({"gauge": g, "altimetry": a}, axis=1).dropna()
    if len(joined) < 24:
        raise OceanToolError(
            "no_data", "gauge and altimetry overlap by fewer than 24 samples"
        )
    joined["gauge"] -= joined["gauge"].mean()
    joined["altimetry"] -= joined["altimetry"].mean()
    joined["difference"] = joined["gauge"] - joined["altimetry"]
    tg = trend_per_year(joined["gauge"])
    ta = trend_per_year(joined["altimetry"])
    td = trend_per_year(joined["difference"])
    numbers = {
        "common_period": [joined.index[0].isoformat(), joined.index[-1].isoformat()],
        "gauge_trend": tg,
        "altimetry_trend": ta,
        "local_trend": td,
        "correlation": float(stats.pearsonr(joined["gauge"], joined["altimetry"])[0]),
        "interpretation": (
            "local_trend = gauge minus altimetry; with the gauge measuring relative sea level, a "
            "positive local trend is consistent with land subsidence of that rate (VLM = -local_trend)."
        ),
    }
    return joined.reset_index(names="time"), numbers
