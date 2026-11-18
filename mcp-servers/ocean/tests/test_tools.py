"""Tools through a real MCP client: results, outputs, provenance, typed errors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from ocean_mcp.server import mcp

EXPECTED_TOOLS = {
    "ctd_read",
    "teos10",
    "water_masses",
    "argo_fetch",
    "glider_fetch",
    "profile_metrics",
    "section",
    "transport",
    "overturning",
    "harmonics",
    "spectra",
    "sea_level_decompose",
    "particles",
    "eddy_track",
    "track_crossings",
    "swath_fetch",
    "swath_compare",
    "mhw_point",
    "mhw_track",
    "mhw_events",
    "bloom_phenology",
    "mixed_layer_budget",
    "cruise_plan",
    "drift",
    "colocate",
    "mom6_preflight",
}


async def call(tool: str, **arguments: object) -> dict:
    async with Client(mcp) as client:
        reply = await client.call_tool(tool, arguments)
    assert reply.content[0].text == reply.structured_content["summary"]
    return reply.structured_content


async def test_tool_set_and_titles() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == EXPECTED_TOOLS
    for tool in tools:
        assert tool.title and len(tool.title) <= 24 and "(" not in tool.title, tool.name
        assert tool.description and len(tool.description) <= 500, tool.name


def _provenance(output: dict) -> dict:
    return json.loads(Path(output["provenance"]).read_text())


async def test_casts_to_water_masses_to_section(
    cast_files: list[Path], whp_file: Path
) -> None:
    read = await call("ctd_read", paths=[str(whp_file)], name="a16n")
    table = read["outputs"][0]["path"]
    assert read["profiles"] == 1 and read["formats"] == ["whp-exchange"]
    masses = await call("water_masses", profiles=table)
    assert masses["definition"]["name"] == "atlantic_emery_meincke_1986"
    assert {c["water_mass"] for c in masses["cores"]} >= {
        "Mediterranean Water",
        "North Atlantic Deep Water",
    }
    record = _provenance(masses["outputs"][0])
    assert record["schema"] == "clio.provenance/1" and record["method"][
        "citation"
    ].startswith("Emery")
    casts = await call("ctd_read", paths=[str(p) for p in cast_files], name="casts")
    sec = await call(
        "section",
        variables=["CT", "SA"],
        profiles=casts["outputs"][0]["path"],
        order="latitude",
    )
    assert sec["columns"] == 3 and sec["path"]["type"] == "LineString"
    assert Path(sec["outputs"][0]["path"]).suffix == ".nc"


async def test_profile_metrics_reports_criterion(cast_files: list[Path]) -> None:
    casts = await call("ctd_read", paths=[str(p) for p in cast_files])
    m = await call(
        "profile_metrics",
        profiles=casts["outputs"][0]["path"],
        criterion="temperature_threshold",
        threshold=0.2,
    )
    assert m["method"]["mld"]["criterion"] == "temperature_threshold"
    assert 50 < m["mld_median_dbar"] < 90


async def test_transport_states_approximation(velocity_grid: Path) -> None:
    out = await call(
        "transport",
        grid=str(velocity_grid),
        path=[[-1, 0], [1, 0]],
        temperature="thetao",
        product="synthetic",
    )
    assert "approximate" in out["summary"]
    assert _provenance(out["outputs"][0])["notes"]


async def test_mhw_point_finds_planted_event(sst_series: Path) -> None:
    out = await call(
        "mhw_point",
        series=str(sst_series),
        column="sst",
        baseline=[1983, 2012],
        period=["2022-01-01", "2022-12-31"],
    )
    assert out["events"] >= 1
    starts = [e["start"] for e in out["top_events"]]
    assert any(s.startswith(("2022-06", "2022-05")) for s in starts)
    assert out["method"]["baseline"] == [1983, 2012]


async def test_particles_follow_uniform_flow(surface_flow: Path) -> None:
    out = await call(
        "particles",
        grid=str(surface_flow),
        station=[2.0, 0.0],
        days=5,
        direction="forward",
        count=50,
        radius_km=1.0,
        dt_minutes=60,
        output_hours=24,
    )
    # 0.1 m/s for 5 days is 43.2 km.
    assert out["median_displacement_km"] == pytest.approx(43.2, rel=0.05)
    trajectories = pd.read_parquet(out["outputs"][0]["path"])
    assert trajectories["particle"].nunique() == 50


async def test_a_second_particle_run_keeps_its_own_count(surface_flow: Path) -> None:
    """Parcels numbers particles across runs in one process; each run still counts its own."""
    for _ in range(2):
        out = await call(
            "particles",
            grid=str(surface_flow),
            station=[2.0, 0.0],
            days=2,
            direction="forward",
            count=20,
            radius_km=1.0,
            dt_minutes=60,
            output_hours=24,
        )
        trajectories = pd.read_parquet(out["outputs"][0]["path"])
        assert trajectories["particle"].nunique() == 20
        assert out["kept_to_end"] == 20


async def test_cruise_plan_retimes_moved_stations() -> None:
    out = await call(
        "cruise_plan",
        port=[0, 0],
        depart="2026-11-20T06:00Z",
        days_available=1,
        stations=[
            {"name": "A", "longitude": 1, "latitude": 0},
            {"name": "B", "longitude": 1, "latitude": 1},
        ],
    )
    assert out["mode"] == "re-timed the given stations"
    assert out["fits"] is False and "does NOT fit" in out["summary"]
    # The plan is shown as the ship's line from port, through the named stations, home.
    stations, route = out["outputs"]
    assert route["role"] == "route"
    line = pd.read_parquet(route["path"])
    assert list(zip(line["longitude"], line["latitude"], strict=True)) == [
        (0, 0),
        (1, 0),
        (1, 1),
        (0, 0),
    ]
    view = stations["view"]
    assert view["component"] == "clio.layered-map.v1"
    route_layer, station_layer = view["layers"]
    assert route_layer["kind"] == "tracks" and route_layer["dataUri"] == route["path"]
    assert (
        station_layer["labelField"] == "station" and station_layer["editable"] is True
    )
    # Moving a station is recorded and sent to the agent to replan.
    assert view["edits"]["path"].startswith("/edits/")
    assert view["action"]["event"]["context"]["moved"] == view["edits"]


async def test_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match=r"^bad_input: "):
        await call("ctd_read", paths=[str(tmp_path / "nothing*.cnv")])
    with pytest.raises(ToolError, match=r"^bad_input: unknown definition"):
        frame = pd.DataFrame(
            {
                "profile": ["a"],
                "time": [pd.Timestamp("2020")],
                "latitude": [0.0],
                "longitude": [0.0],
                "pressure": [1.0],
                "temperature": [1.0],
                "salinity": [35.0],
            }
        )
        path = tmp_path / "p.parquet"
        frame.to_parquet(path)
        await call("water_masses", profiles=str(path), definition="pacific_nowhere")


async def test_swath_download_without_login_names_the_credential(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in ("EARTHDATA_USERNAME", "EARTHDATA_PASSWORD", "EARTHDATA_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    with pytest.raises(ToolError, match=r"^credential_missing: NASA Earthdata Login"):
        await call(
            "swath_fetch",
            box=[-75, -70, 35, 40],
            start="2024-01-01",
            end="2024-01-03",
            download=True,
        )


async def test_bloom_phenology_returns_a_year_without_a_start_as_null(
    tmp_path: Path,
) -> None:
    # One year with a spring bloom, one already above its threshold from 1 January.
    t = pd.date_range("2020-01-01", "2021-12-31", freq="D", tz="UTC")
    doy = t.day_of_year.to_numpy()
    bloom = 0.3 + 2.0 * np.exp(-((doy - 100) ** 2) / (2 * 12**2))
    early = np.where(doy < 120, 1.0 + doy / 100.0, 0.2)
    chl = np.where(t.year == 2020, bloom, early)
    series = tmp_path / "chl.csv"
    pd.DataFrame({"time": t, "chl": chl}).to_csv(series, index=False)
    data = await call(
        "bloom_phenology",
        method="median_plus_5pct",
        series=str(series),
        column="chl",
        output_dir=str(tmp_path),
    )
    by_year = {row["year"]: row for row in data["starts"]}
    assert by_year[2020]["bloom_start"].startswith("2020-03")
    assert by_year[2021]["bloom_start"] is None
    assert "not observed" in by_year[2021]["reason"]


async def test_particles_say_what_their_spread_means_and_can_diffuse(
    surface_flow: Path,
) -> None:
    """Q9: 500 particles moved as one, and the result called it a spread."""
    common = {
        "grid": str(surface_flow),
        "station": [2.0, 0.0],
        "days": 5,
        "direction": "forward",
        "count": 50,
        "radius_km": 1.0,
        "dt_minutes": 60,
        "output_hours": 24,
    }
    still = await call("particles", **common)
    assert "not an uncertainty" in still["summary"]
    # A start time with a zone (as an agent writes it, "...Z") is read as UTC (Q9 rerun).
    spread = await call(
        "particles", diffusivity_m2s=100.0, start="2021-01-01T00:00:00Z", **common
    )
    assert "random walk" in spread["summary"]
    # sqrt(2 K t) for 100 m2/s over 5 days is about 9 km per axis.
    assert spread["end_spread_km"] > 2 * still["end_spread_km"]
    assert spread["median_displacement_km"] == pytest.approx(43.2, rel=0.2)
    with pytest.raises(ToolError, match="out_of_range"):
        await call("particles", start="2020-06-01", **common)


def _glider_like(tmp_path: Path) -> Path:
    """Two transect casts on pressure levels, one sample flagged as a spike."""
    rows = []
    for k, profile in enumerate(("p1", "p2")):
        for pressure in (5.0, 15.0, 25.0, 35.0):
            rows.append(
                {
                    "profile": profile,
                    "time": pd.Timestamp("2020-09-15", tz="UTC")
                    + pd.Timedelta(hours=k),
                    "latitude": 17.9,
                    "longitude": -64.8 + 0.05 * k,
                    "pressure": pressure,
                    "temperature": 28.0 - pressure / 10,
                    "salinity": 35.5,
                    "transect": 18,
                    "suspect": "temperature spike"
                    if (k, pressure) == (1, 15.0)
                    else "",
                }
            )
    path = tmp_path / "glider.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    return path


async def test_a_section_drops_the_samples_a_fetch_flagged(tmp_path: Path) -> None:
    profiles = _glider_like(tmp_path)
    out = await call(
        "section", variables=["CT"], profiles=str(profiles), output_dir=str(tmp_path)
    )
    assert out["suspect_dropped"] == {"temperature spike": 1}
    assert "1 suspect samples dropped" in out["summary"]
    kept = await call(
        "section",
        variables=["CT"],
        profiles=str(profiles),
        keep_suspect=True,
        name="kept",
        output_dir=str(tmp_path),
    )
    assert kept["suspect_dropped"] == {}


async def test_teos10_reports_a_non_finite_n2_instead_of_failing(
    tmp_path: Path,
) -> None:
    rows = [
        {
            "profile": "p1",
            "time": pd.Timestamp("2020-09-15", tz="UTC"),
            "latitude": 17.9,
            "longitude": -64.8,
            "pressure": pressure,
            "temperature": 28.0 - pressure / 10,
            "salinity": 35.5,
        }
        # A repeated pressure gives a zero layer thickness and an infinite N².
        for pressure in (5.0, 15.0, 15.0, 25.0, 35.0)
    ]
    path = tmp_path / "dup.parquet"
    pd.DataFrame(rows).to_parquet(path, index=False)
    out = await call(
        "teos10", profiles=str(path), buoyancy_frequency=True, output_dir=str(tmp_path)
    )
    assert out["N2_max"] is None or np.isfinite(out["N2_max"])
    assert out["N2_non_finite"] >= 0
