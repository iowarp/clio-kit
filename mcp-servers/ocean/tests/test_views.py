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
    assert "colormap" not in layer
    bias = xr.Dataset(
        {"bias": (("latitude", "longitude"), np.array([[-1.0, 2.0], [0.5, -0.2]]))},
        coords={"latitude": [0.0, 1.0], "longitude": [0.0, 1.0]},
    )
    assert (
        views.grid_view("grid", tmp_path / "b.nc", bias)["layers"][0]["colormap"]
        == "balance"
    )
    assert layer["gridUri"] == str(path)
    section = xr.Dataset(
        {"CT": (("pressure", "distance"), np.zeros((3, 2)))},
        coords={"pressure": [0.0, 10, 20], "distance": [0.0, 5.0]},
    )
    view = views.grid_view("section", tmp_path / "s.nc", section)
    assert view["component"] == "clio.section.v1" and view["variable"] == "CT"


def test_particle_trajectories_are_tracks_too(tmp_path: Path) -> None:
    """The particles tool names its track column ``particle`` (Q9: no track view)."""
    table = pd.DataFrame(
        {
            "particle": [0, 0, 1, 1],
            "time": pd.to_datetime(["2026-09-16", "2026-09-17"] * 2),
            "longitude": [-64.2, -64.3, -64.1, -64.0],
            "latitude": [31.7, 31.8, 31.6, 31.5],
        }
    )
    view = views.table_view("tracks", tmp_path / "p.parquet", table)
    assert view is not None
    [layer] = view["layers"]
    assert layer["kind"] == "tracks" and layer["trackField"] == "particle"


def test_a_transport_shows_its_mean_section(tmp_path: Path) -> None:
    """Q4: the transport arrived with no view; its 2-D section is now shown."""
    ds = xr.Dataset(
        {
            "volume_transport": (("time",), np.ones(3)),
            "normal_velocity": (("time", "depth", "distance"), np.ones((3, 4, 5))),
            "normal_velocity_mean": (("depth", "distance"), np.ones((4, 5))),
        },
        coords={
            "time": np.arange(3),
            "depth": np.arange(4.0),
            "distance": np.arange(5.0),
        },
    )
    view = views.grid_view("transport", tmp_path / "t.nc", ds)
    assert view is not None and view["component"] == "clio.section.v1"
    assert view["variables"] == ["normal_velocity_mean"]
    assert (
        views.grid_view("transport", tmp_path / "t.nc", ds[["volume_transport"]])
        is None
    )


def test_a_route_view_is_general_and_carries_its_context(tmp_path: Path) -> None:
    """Any ordered plan (cruise, glider, sampling line) gets one map: line, stops, context."""
    feature = {"id": "feature", "kind": "tracks", "label": "Forecast track"}
    view = views.route_view(
        tmp_path / "route.parquet",
        tmp_path / "wp.parquet",
        "Glider mission",
        stop_field="waypoint",
        stop_label="Waypoints",
        context=[feature],
        replan_event="mission.replan",
    )
    assert [layer["id"] for layer in view["layers"]] == ["feature", "route", "stops"]
    stops = view["layers"][2]
    assert stops["idField"] == stops["labelField"] == "waypoint"
    assert view["action"]["event"]["name"] == "mission.replan"


def test_a_glider_track_is_a_map_linked_to_its_profiles(tmp_path: Path) -> None:
    positions = pd.DataFrame(
        {
            "profile": [1, 2, 3],
            "time": pd.to_datetime(["2020-09-08", "2020-09-09", "2020-09-10"]),
            "latitude": [18.1, 18.0, 17.9],
            "longitude": [-64.8, -64.7, -64.6],
            "transect": [2, 2, -1],
            "leg": ["transect 2", "transect 2", "between transects"],
        }
    )
    path = tmp_path / "ru29_positions.parquet"
    view = views.positions_view(path, positions, "ru29 2020", title="ru29 track")
    assert view is not None
    (layer,) = view["layers"]
    assert view["component"] == "clio.layered-map.v1"
    assert layer["kind"] == "points" and layer["idField"] == "profile"
    assert layer["categoryField"] == "leg" and layer["timeField"] == "time"
    # The profile stack of the same deployment binds the same selection by profile id.
    assert view["selection"] == views.profile_selection("ru29 2020")["selection"]
    assert view["selection"] == {"path": "/selection/ru29-2020-profile"}
    assert view["selectionField"] == "profile"
    assert views.positions_view(path, positions.drop(columns="latitude"), "k") is None
