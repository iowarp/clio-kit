"""Each output a person checks by eye names the component that shows it."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from ocean_mcp.implementation import views


def test_a_profile_table_is_shown_as_a_profile_stack(tmp_path: Path) -> None:
    table = pd.DataFrame(
        {
            "profile": ["a", "a", "b"],
            "time": pd.to_datetime(["2026-01-01"] * 3),
            "latitude": [10.0, 10.0, 11.0],
            "longitude": [60.0, 60.0, 61.0],
            "pressure": [5.0, 10.0, 5.0],
            "temperature": [28.0, 27.5, 28.2],
            "salinity": [36.1, 36.2, 36.0],
        }
    )
    view = views.table_view("profiles", tmp_path / "p.parquet", table)
    assert view is not None and view["component"] == "clio.profile-stack.v1"
    assert view["profileField"] == "profile" and view["verticalField"] == "pressure"
    assert view["verticalUnit"] == "dbar" and view["invertVertical"] is True
    assert view["valueField"] == "temperature"
    assert view["variables"] == ["temperature", "salinity"]
    assert view["dataUri"] == str(tmp_path / "p.parquet")


def test_tracks_and_points_are_map_layers_and_a_summary_has_no_view(
    tmp_path: Path,
) -> None:
    tracks = pd.DataFrame(
        {
            "track": [1, 1],
            "time": pd.to_datetime(["2026-01-01", "2026-01-02"]),
            "latitude": [10.0, 11.0],
            "longitude": [140.0, 139.0],
        }
    )
    view = views.table_view("tracks", tmp_path / "t.parquet", tracks)
    layer = view["layers"][0]
    assert view["component"] == "clio.layered-map.v1" and layer["kind"] == "tracks"
    assert layer["trackField"] == "track" and layer["timeField"] == "time"
    points = pd.DataFrame(
        {"station": ["X"], "latitude": [1.0], "longitude": [2.0], "v": [3.0]}
    )
    point = views.table_view("points", tmp_path / "s.parquet", points)["layers"][0]
    assert point["kind"] == "points" and point["idField"] == "station"
    summary = pd.DataFrame({"model": ["A"], "rmse": [1.0]})
    assert views.table_view("table", tmp_path / "x.parquet", summary) is None


def test_a_field_is_a_grid_layer_and_a_section_its_own_view(tmp_path: Path) -> None:
    field = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.zeros((2, 3, 4)))},
        coords={
            "time": [0, 1],
            "latitude": [0.0, 1.0, 2.0],
            "longitude": [0.0, 1, 2, 3],
        },
    )
    path = tmp_path / "f.nc"
    field.to_netcdf(path)
    layer = views.grid_file_view("grid", path)["layers"][0]
    assert layer["kind"] == "grid" and layer["variable"] == "sst"
    assert layer["gridUri"] == str(path)
    section = xr.Dataset(
        {"CT": (("pressure", "distance"), np.zeros((3, 2)))},
        coords={"pressure": [0.0, 10, 20], "distance": [0.0, 5.0]},
    )
    view = views.grid_view("section", tmp_path / "s.nc", section)
    assert view["component"] == "clio.section.v1" and view["variable"] == "CT"
