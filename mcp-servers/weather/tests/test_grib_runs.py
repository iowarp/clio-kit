"""Several GRIB forecast files, one run each, are scored as runs (init x lead)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from weather_mcp.implementation import grids, verification


def _run(init: str, hours: list[int], value: float) -> xr.Dataset:
    lat = np.array([10.0, 9.75])
    lon = np.array([130.0, 130.25])
    steps = pd.to_timedelta(hours, unit="h")
    data = np.full((len(hours), 2, 2), value, dtype="float32")
    return xr.Dataset(
        {"msl": (("step", "latitude", "longitude"), data)},
        coords={
            "time": np.datetime64(init),
            "step": steps,
            "latitude": lat,
            "longitude": lon,
        },
    )


def test_runs_stack_along_their_run_time_and_absent_leads_are_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    runs = {
        "a.grib2": _run("2026-11-06T12", [0, 6, 12], 101000.0),
        "b.grib2": _run("2026-11-07T00", [0, 6], 101100.0),
    }
    for name in runs:
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(grids, "_grib_file", lambda path, _v: runs[Path(path).name])
    fc = grids.open_grid(str(tmp_path / "*.grib2"), ["msl"])
    assert fc["msl"].dims[:2] == ("time", "step")
    times = pd.date_range("2026-11-06T12", "2026-11-07T12", freq="6h")
    ref = xr.Dataset(
        {
            "msl": (
                ("time", "latitude", "longitude"),
                np.full((len(times), 2, 2), 101000.0),
            )
        },
        coords={"time": times, "latitude": [10.0, 9.75], "longitude": [130.0, 130.25]},
    )
    table, _map, numbers = verification.score(
        fc,
        ref,
        variable="msl",
        reference_variable="msl",
        box=None,
        climatology=None,
        map_lead_hours=None,
    )
    # Two runs, five scored (run, lead) pairs: run b has no 12 h lead, and it is
    # skipped rather than scored as empty. Run a is 0 Pa off, run b 100 Pa.
    assert numbers["initialisations"] == 2
    assert len(table) == 5
    by_lead = {row["lead_hours"]: row["rmse"] for row in numbers["by_lead"]}
    assert by_lead == pytest.approx({0.0: 50.0, 6.0: 50.0, 12.0: 0.0})


def test_a_glob_with_no_file_holding_the_variable_is_refused(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    for name in ("a.grib2", "b.grib2"):
        (tmp_path / name).write_bytes(b"")
    monkeypatch.setattr(grids, "_grib_file", lambda _p, _v: xr.Dataset())
    with pytest.raises(grids.WeatherToolError, match="none of the GRIB files"):
        grids.open_grid(str(tmp_path / "*.grib2"), ["msl"])
