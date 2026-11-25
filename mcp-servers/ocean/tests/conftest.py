"""Synthetic fixtures with known answers for the ocean server tests."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr


@pytest.fixture(autouse=True)
def _outputs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    out = tmp_path / "outputs"
    monkeypatch.setenv("OCEAN_MCP_OUTPUT_DIR", str(out))
    return out


def write_cnv(
    path: Path,
    lat: str,
    lon: str,
    pressure: np.ndarray,
    temp: np.ndarray,
    sal: np.ndarray,
) -> Path:
    header = [
        "* Sea-Bird SBE 9 Data File:",
        f"* NMEA Latitude = {lat}",
        f"* NMEA Longitude = {lon}",
        "* NMEA UTC (Time) = Aug 05 2013 12:00:00",
        "# nquan = 3",
        "# name 0 = prDM: Pressure, Digiquartz [db]",
        "# name 1 = t090C: Temperature [ITS-90, deg C]",
        "# name 2 = sal00: Salinity, Practical [PSU]",
        "# bad_flag = -9.990e-29",
        "*END*",
    ]
    rows = [
        f"{p:11.3f} {t:10.4f} {s:10.4f}"
        for p, t, s in zip(pressure, temp, sal, strict=True)
    ]
    path.write_text("\n".join(header + rows) + "\n", encoding="latin-1")
    return path


def stratified(pressure: np.ndarray, mld: float, surface: float = 20.0) -> np.ndarray:
    """Temperature uniform to ``mld`` then falling 0.05 °C per dbar, floored at 2 °C."""
    return np.where(
        pressure <= mld, surface, np.maximum(surface - 0.05 * (pressure - mld), 2.0)
    )


@pytest.fixture
def cast_files(tmp_path: Path) -> list[Path]:
    p = np.arange(2.0, 1502.0, 2.0)
    files = []
    for i, (lat, mld) in enumerate(
        [("30 00.00 N", 50.0), ("31 00.00 N", 80.0), ("32 00.00 N", 120.0)]
    ):
        temp = stratified(p, mld)
        sal = 35.0 + 0.0 * p
        files.append(
            write_cnv(tmp_path / f"cast{i:02d}.cnv", lat, "40 00.00 W", p, temp, sal)
        )
    return files


@pytest.fixture
def whp_file(tmp_path: Path) -> Path:
    lines = [
        "CTD,20131001SIOODFSIO",
        "# synthetic",
        "NUMBER_HEADERS = 9",
        "EXPOCODE = 33RO20130803",
        "SECT_ID = A16N",
        "STNNBR = 7",
        "CASTNO = 1",
        "DATE = 20130805",
        "TIME = 1530",
        "LATITUDE = 40.5000",
        "LONGITUDE = -20.0000",
        "CTDPRS,CTDPRS_FLAG_W,CTDTMP,CTDTMP_FLAG_W,CTDSAL,CTDSAL_FLAG_W",
        "DBAR,,ITS-90,,PSS-78,",
        "    2.0,2,  18.0000,2,  36.1000,2",
        "  500.0,2,  11.0000,2,  35.6000,2",
        " 1000.0,2,   9.5000,2,  35.9000,2",
        " 1100.0,2,   9.0000,4,  35.8000,2",
        " 2500.0,2,   3.0000,2,  34.9500,2",
        "END_DATA",
    ]
    path = tmp_path / "a16n_00007_00001_ct1.csv"
    path.write_text("\n".join(lines) + "\n")
    return path


@pytest.fixture
def velocity_grid(tmp_path: Path) -> Path:
    """Uniform 0.1 m/s northward flow, 1000 m deep, three time steps."""
    lon = np.arange(-10.0, 10.01, 0.25)
    lat = np.arange(-5.0, 5.01, 0.25)
    depth = np.arange(5.0, 1000.0, 10.0)
    time = pd.date_range("2020-01-01", periods=3, freq="D")
    shape = (len(time), len(depth), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "uo": (("time", "depth", "latitude", "longitude"), np.zeros(shape)),
            "vo": (("time", "depth", "latitude", "longitude"), np.full(shape, 0.1)),
            "thetao": (
                ("time", "depth", "latitude", "longitude"),
                np.full(shape, 10.0),
            ),
        },
        coords={"time": time, "depth": depth, "latitude": lat, "longitude": lon},
    )
    path = tmp_path / "velocity.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def surface_flow(tmp_path: Path) -> Path:
    """Uniform 0.1 m/s eastward surface flow over 12 days."""
    lon = np.arange(0.0, 10.01, 0.1)
    lat = np.arange(-3.0, 3.01, 0.1)
    time = pd.date_range("2021-01-01", periods=13, freq="D")
    shape = (len(time), len(lat), len(lon))
    ds = xr.Dataset(
        {
            "uo": (("time", "latitude", "longitude"), np.full(shape, 0.1)),
            "vo": (("time", "latitude", "longitude"), np.zeros(shape)),
        },
        coords={"time": time, "latitude": lat, "longitude": lon},
    )
    path = tmp_path / "surface.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def tide_series(tmp_path: Path) -> Path:
    """60 days hourly: M2 amplitude 1.0 m phase 30°, K1 0.3 m phase 120°, plus noise."""
    t = pd.date_range("2020-01-01", periods=24 * 60, freq="h", tz="UTC")
    hours = (t - pd.Timestamp("2020-01-01", tz="UTC")).total_seconds() / 3600
    m2 = 1.0 * np.cos(2 * np.pi * hours / 12.4206012 - np.deg2rad(30))
    k1 = 0.3 * np.cos(2 * np.pi * hours / 23.9344697 - np.deg2rad(120))
    rng = np.random.default_rng(1)
    frame = pd.DataFrame({"time": t, "height": m2 + k1 + rng.normal(0, 0.02, len(t))})
    path = tmp_path / "gauge.csv"
    frame.to_csv(path, index=False)
    return path


@pytest.fixture
def sst_series(tmp_path: Path) -> Path:
    """1982-2022 daily SST: seasonal cycle, noise, and a planted 2022 heatwave."""
    t = pd.date_range("1982-01-01", "2022-12-31", freq="D")
    rng = np.random.default_rng(2)
    doy = t.day_of_year.to_numpy()
    sst = 20 + 4 * np.sin(2 * np.pi * (doy - 120) / 365.25) + rng.normal(0, 0.3, len(t))
    event = (t >= "2022-06-01") & (t <= "2022-07-10")
    sst[event] += 2.0 * np.sin(np.linspace(0, np.pi, event.sum()))
    frame = pd.DataFrame({"time": t, "sst": sst})
    path = tmp_path / "sst.csv"
    frame.to_csv(path, index=False)
    return path
