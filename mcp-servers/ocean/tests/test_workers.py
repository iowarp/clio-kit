"""The copyleft workers end to end: they build their own environment and run.

Marked ``worker``: the first run builds each environment with uv (minutes),
so CI may deselect them with ``-m "not worker"``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from .test_tools import call

pytestmark = (
    [pytest.mark.worker, pytest.mark.timeout(3600)]
    if hasattr(pytest.mark, "timeout")
    else [pytest.mark.worker]
)


@pytest.fixture
def sla_grid(tmp_path: Path) -> Path:
    """Twelve days: an anticyclone drifting east and a cyclone drifting west."""
    t = pd.date_range("2020-01-01", periods=12, freq="D")
    lat = np.arange(20, 40, 0.25)
    lon = np.arange(-80, -50, 0.25)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    frames = []
    for i in range(len(t)):
        warm = 0.3 * np.exp(-((yy - 30) ** 2 + (xx - (-65 + 0.1 * i)) ** 2) / 0.8)
        cold = -0.25 * np.exp(-((yy - 25) ** 2 + (xx - (-70 - 0.05 * i)) ** 2) / 0.6)
        frames.append(warm + cold)
    ds = xr.Dataset(
        {"sla": (("time", "latitude", "longitude"), np.array(frames), {"units": "m"})},
        coords={"time": t, "latitude": lat, "longitude": lon},
    )
    path = tmp_path / "sla.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def sst_grid(tmp_path: Path) -> Path:
    """1990-2003 daily SST on a 10x10 grid with a warm blob in summer 2003."""
    t = pd.date_range("1990-01-01", "2003-12-31", freq="D")
    lat = np.arange(30, 35, 0.5)
    lon = np.arange(0, 5, 0.5)
    rng = np.random.default_rng(0)
    doy = t.day_of_year.to_numpy()
    sst = 18 + 5 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    data = sst[:, None, None] + rng.normal(0, 0.4, (len(t), len(lat), len(lon)))
    blob = np.exp(-((lat[:, None] - 32.5) ** 2 + (lon[None, :] - 2.5) ** 2) / 2)
    hot = (t >= "2003-06-10") & (t <= "2003-08-20")
    data[hot] += 3.0 * blob
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), data.astype("float32"))},
        coords={"time": t, "lat": lat, "lon": lon},
    )
    path = tmp_path / "sst.nc"
    ds.to_netcdf(path)
    return path


async def test_eddy_track_finds_both_polarities(sla_grid: Path) -> None:
    out = await call(
        "eddy_track", grid=str(sla_grid), wavelength_km=None, min_lifetime_days=5
    )
    assert out["identified"]["anticyclonic"] >= 10
    assert out["identified"]["cyclonic"] >= 10
    assert set(out["tracks_by_polarity"]) == {"anticyclonic", "cyclonic"}
    tracks = pd.read_parquet(out["outputs"][0]["path"])
    warm = tracks[tracks["polarity"] == "anticyclonic"].sort_values("time")
    assert (
        warm["longitude"].iloc[-1] > warm["longitude"].iloc[0]
    )  # drifts east as planted


async def test_mhw_track_and_metrics_find_the_2003_event(sst_grid: Path) -> None:
    tracked = await call(
        "mhw_track",
        grid=str(sst_grid),
        baseline=[1990, 2003],
        percentile=90,
        min_duration=10,
    )
    assert tracked["events_kept"] >= 1
    series = next(o for o in tracked["outputs"] if o["path"].endswith(".parquet"))
    metrics = await call(
        "mhw_events", event_series=series["path"], period=["2003-05-01", "2003-09-30"]
    )
    assert metrics["events"] >= 1
    top = metrics["top_events"][0]
    assert top["start"][:7] in ("2003-06", "2003-07")
    assert top["duration_days"] >= 30
    assert top["category"] in ("Strong", "Severe", "Extreme")
