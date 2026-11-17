"""Synthetic fields with known answers for the weather server tests."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def _outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    monkeypatch.setenv("WEATHER_MCP_OUTPUT_DIR", str(out))
    return out


@pytest.fixture
def ensemble_point(tmp_path: Path) -> Path:
    """51 members at a point; member k rains k mm per day, accumulated in metres."""
    steps = np.arange(0, 73, 6)
    members = np.arange(51)
    run = np.datetime64("2026-11-16T00:00")
    daily_mm = members.astype(float)  # 0..50 mm/day
    hours = steps.astype(float)
    tp = (daily_mm[:, None] * hours[None, :] / 24.0) / 1000.0
    ds = xr.Dataset(
        {"tp": (("number", "step"), tp, {"units": "m"})},
        coords={
            "number": members,
            "step": steps.astype("timedelta64[h]").astype("timedelta64[ns]"),
            "time": run,
            "valid_time": ("step", run + steps.astype("timedelta64[h]")),
        },
    )
    path = tmp_path / "ens.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def hourly_t2m(tmp_path: Path) -> Path:
    """1991-2022 hourly 2 m temperature (K) with a planted 6-day heat wave in July 2022."""
    t = pd.date_range("1991-01-01", "2022-12-31 23:00", freq="3h")
    rng = np.random.default_rng(3)
    doy = t.day_of_year.to_numpy()
    hour = t.hour.to_numpy()
    seasonal = 290 + 8 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    diurnal = 4 * np.sin(2 * np.pi * (hour - 9) / 24)
    daily_noise = rng.normal(0, 1.2, len(pd.date_range(t[0], t[-1], freq="D")))
    value = seasonal + diurnal + np.repeat(daily_noise, 8)[: len(t)]
    wave = (t >= "2022-07-10") & (t < "2022-07-16")
    value[wave] += 7.0
    frame = pd.DataFrame({"time": t, "t2m": value})
    path = tmp_path / "t2m.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def pressure_levels(tmp_path: Path) -> Path:
    """A conditionally unstable atmosphere at one time on a small grid."""
    levels = np.array([1000, 925, 850, 700, 600, 500, 400, 300, 250, 200], float)
    lat = np.arange(30, 41, 1.0)
    lon = np.arange(-100, -89, 1.0)
    z = 44330 * (1 - (levels / 1013.25) ** 0.1903)
    temp = 303.0 - 0.0065 * z  # K, warm surface, standard lapse rate
    q = np.clip(0.016 * np.exp(-z / 2500.0), 1e-5, None)
    shape = (len(levels), len(lat), len(lon))
    t3 = np.broadcast_to(temp[:, None, None], shape).copy()
    t3 += (lon[None, None, :] + 95) * 0.3  # a temperature gradient across longitude
    ds = xr.Dataset(
        {
            "temperature": (("level", "latitude", "longitude"), t3, {"units": "K"}),
            "specific_humidity": (
                ("level", "latitude", "longitude"),
                np.broadcast_to(q[:, None, None], shape).copy(),
                {"units": "kg/kg"},
            ),
            "u_component_of_wind": (
                ("level", "latitude", "longitude"),
                np.full(shape, 10.0),
                {"units": "m/s"},
            ),
            "v_component_of_wind": (
                ("level", "latitude", "longitude"),
                np.zeros(shape),
                {"units": "m/s"},
            ),
        },
        coords={
            "level": ("level", levels, {"units": "hPa"}),
            "latitude": lat,
            "longitude": lon,
        },
    )
    path = tmp_path / "plev.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def forecast_and_truth(tmp_path: Path) -> tuple[Path, Path]:
    """Forecast = truth + 1 K everywhere; truth hourly."""
    lat = np.arange(-10, 10.1, 2.5)
    lon = np.arange(0, 20.1, 2.5)
    valid = pd.date_range("2026-01-01", periods=11, freq="6h")
    truth = xr.Dataset(
        {
            "t2m": (
                ("time", "latitude", "longitude"),
                280 + np.random.default_rng(0).normal(0, 2, (11, len(lat), len(lon))),
            )
        },
        coords={"time": valid, "latitude": lat, "longitude": lon},
    )
    leads = pd.to_timedelta(np.arange(0, 61, 6), unit="h")
    fc = xr.Dataset(
        {
            "t2m": (
                ("lead_time", "latitude", "longitude"),
                truth["t2m"].to_numpy() + 1.0,
            )
        },
        coords={
            "lead_time": leads,
            "time": np.datetime64("2026-01-01"),
            "latitude": lat,
            "longitude": lon,
        },
    )
    tp, fp = tmp_path / "truth.nc", tmp_path / "fc.nc"
    truth.to_netcdf(tp)
    fc.to_netcdf(fp)
    return fp, tp


@pytest.fixture
def te_bin() -> str:
    folder = os.environ.get("TEMPESTEXTREMES_BIN")
    if not folder or not (Path(folder) / "DetectBlobs").is_file():
        pytest.skip("TempestExtremes binaries not available (set TEMPESTEXTREMES_BIN)")
    return folder
