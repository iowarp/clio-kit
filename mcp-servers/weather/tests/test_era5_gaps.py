"""ERA5 fetch: time steps empty in the store are reported, not passed off as data."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from fastmcp import Client

from weather_mcp.implementation import fetch
from weather_mcp.implementation.common import Source
from weather_mcp.server import mcp


def _grid() -> xr.Dataset:
    time = pd.date_range("2026-11-08", periods=4, freq="12h")
    values = np.ones((4, 2, 2))
    values[2] = np.nan
    ds = xr.Dataset(
        {"mean_sea_level_pressure": (("time", "latitude", "longitude"), values)},
        coords={"time": time, "latitude": [10.0, 20.0], "longitude": [130.0, 140.0]},
    )
    ds.attrs["arco_valid_time_stop_era5t"] = "2026-11-11"
    return ds


def test_empty_times_lists_steps_with_no_values() -> None:
    assert fetch.empty_times(_grid()) == ["2026-11-09T00:00:00"]


async def test_era5_fetch_reports_empty_steps(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        fetch,
        "era5_arco",
        lambda **_: (_grid(), Source(name="ERA5", kind="product")),
    )
    async with Client(mcp) as client:
        reply = await client.call_tool(
            "era5_fetch",
            {
                "variables": ["mean_sea_level_pressure"],
                "box": [130, 140, 10, 20],
                "start": "2026-11-08",
                "end": "2026-11-09T12",
                "output_dir": str(tmp_path),
            },
        )
    data = reply.structured_content
    assert data["empty_times"] == ["2026-11-09T00:00:00"]
    assert "1 of 4 time steps are empty" in data["summary"]
    prov = json.loads(next(tmp_path.glob("*.provenance.json")).read_text())
    assert "2026-11-11" in prov["notes"][0]
