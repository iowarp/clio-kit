"""Weather tools against known answers, through a real MCP client."""

from __future__ import annotations

from pathlib import Path

import metpy.calc as mpcalc
import numpy as np
import pandas as pd
import pytest
import xarray as xr
from fastmcp import Client
from fastmcp.exceptions import ToolError
from metpy.units import units

from weather_mcp.implementation import climate, verification
from weather_mcp.server import mcp

EXPECTED = {
    "era5_fetch",
    "ensemble_fetch",
    "exceedance",
    "threshold_events",
    "cross_section",
    "soundings",
    "detect_blobs",
    "stitch_blobs",
    "blob_stats",
    "climate_index",
    "composite",
    "evaluate_run",
    "cmip_change",
    "score",
    "station_fetch",
    "station_verify",
    "ibtracs_fetch",
    "track_score",
}


async def call(tool: str, **arguments: object) -> dict:
    async with Client(mcp) as client:
        reply = await client.call_tool(tool, arguments)
    assert reply.content[0].text == reply.structured_content["summary"]
    return reply.structured_content


async def test_tools_and_titles() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} >= EXPECTED
    for tool in tools:
        assert tool.title and len(tool.title) <= 24 and "(" not in tool.title, tool.name


async def test_exceedance_counts_members(ensemble_point: Path) -> None:
    out = await call(
        "exceedance", ensemble=str(ensemble_point), variable="tp", thresholds=[30.0]
    )
    # Member k rains k mm/day: 20 of 51 members (31..50) exceed 30 mm in a day.
    assert out["max_probability_gt_30"] == pytest.approx(20 / 51)
    # Over three days member k has 3k mm: members 11..50 exceed 30 mm.
    assert out["probability_total_gt_30"] == pytest.approx(40 / 51)
    assert out["windows"] == 3


async def test_threshold_events_find_planted_heat_wave(hourly_t2m: Path) -> None:
    out = await call(
        "threshold_events",
        series=str(hourly_t2m),
        column="t2m",
        baseline=[1991, 2020],
        daily_statistic="max",
        percentile=90,
        min_duration=3,
        offset=-273.15,
        period=["2022-01-01", "2022-12-31"],
    )
    starts = [e["start"][:10] for e in out["longest"]]
    assert any("2022-07-09" <= s <= "2022-07-11" for s in starts)
    planted = next(
        e for e in out["longest"] if "2022-07-09" <= e["start"][:10] <= "2022-07-11"
    )
    assert 5 <= planted["duration_days"] <= 8
    assert out["method"]["baseline"] == [1991, 2020]


async def test_soundings_match_metpy_directly(pressure_levels: Path) -> None:
    out = await call(
        "soundings",
        grid=str(pressure_levels),
        points=[{"name": "A", "longitude": -95, "latitude": 35}],
    )
    row = out["indices"][0]
    with xr.open_dataset(pressure_levels) as ds:
        col = ds.sel(longitude=-95, latitude=35)
        p = col["level"].to_numpy() * units.hPa
        t = (col["temperature"].to_numpy() * units.K).to("degC")
        td = mpcalc.dewpoint_from_specific_humidity(
            p, col["specific_humidity"].to_numpy() * units("kg/kg")
        ).to("degC")
        cape, _ = mpcalc.surface_based_cape_cin(p, t, td)
    assert row["sbcape"] == pytest.approx(float(cape.m), rel=1e-6)
    assert row["sbcape"] > 100  # warm, moist surface under a standard lapse rate
    profiles = pd.read_parquet(out["outputs"][0]["path"])
    assert {"pressure", "temperature_c", "dewpoint_c", "parcel_temperature_c"} <= set(
        profiles.columns
    )


async def test_cross_section_follows_path(pressure_levels: Path) -> None:
    out = await call(
        "cross_section",
        grid=str(pressure_levels),
        path=[[-99, 35], [-91, 35]],
        steps_per_leg=20,
    )
    assert out["length_km"] == pytest.approx(
        8 * 111.2 * np.cos(np.radians(35)), rel=0.02
    )
    ds = xr.open_dataset(out["outputs"][0]["path"])
    assert {
        "potential_temperature",
        "equivalent_potential_temperature",
        "normal_wind",
        "tangential_wind",
    } <= set(ds.data_vars)
    # Eastward path, westerly 10 m/s wind: almost all along-path. The legs are geodesics,
    # not parallels, so a zonal wind keeps a small (about 2 degree) cross-path part.
    assert float(ds["tangential_wind"].mean()) == pytest.approx(10.0, abs=0.1)
    assert float(abs(ds["normal_wind"]).max()) < 0.5


async def test_score_bias_and_rmse(forecast_and_truth: tuple[Path, Path]) -> None:
    fc, truth = forecast_and_truth
    out = await call(
        "score",
        forecast=str(fc),
        reference=str(truth),
        variable="t2m",
        map_lead_hours=24,
        model="test",
    )
    for row in out["by_lead"]:
        assert row["bias"] == pytest.approx(1.0, abs=1e-9)
        assert row["rmse"] == pytest.approx(1.0, abs=1e-9)
    assert len(out["outputs"]) == 2


def test_fair_crps_known_value() -> None:
    # Two members at 0 and 2, observation 1: mean|X-y| = 1, pair term 2*2/(2*2*1) = 1, CRPS = 0.
    assert verification.fair_crps(np.array([[0.0], [2.0]]), np.array([1.0]))[
        0
    ] == pytest.approx(0.0)
    assert verification.fair_crps(np.array([[1.0], [1.0]]), np.array([3.0]))[
        0
    ] == pytest.approx(2.0)


async def test_track_score_one_degree_offset(tmp_path: Path) -> None:
    times = pd.date_range("2026-09-01", periods=9, freq="6h", tz="UTC")
    best = pd.DataFrame(
        {
            "track": "X",
            "name": "TEST",
            "time": times,
            "latitude": np.linspace(15, 20, 9),
            "longitude": np.linspace(-50, -60, 9),
            "mslp_hpa": 980.0,
            "wind_kt": 80.0,
        }
    )
    fc = best.assign(
        model="m1",
        lead_hours=np.arange(9) * 6.0,
        latitude=best["latitude"] + 1.0,
        mslp_hpa=990.0,
    )
    bp, fp = tmp_path / "best.parquet", tmp_path / "fc.parquet"
    best.to_parquet(bp)
    fc.to_parquet(fp)
    out = await call("track_score", forecasts=[str(fp)], best_track=str(bp))
    assert out["mean_position_error_km"]["m1"] == pytest.approx(111.2, abs=1.0)
    assert all(r["mslp_error_hpa"] == pytest.approx(10.0) for r in out["by_model_lead"])


def test_oni_phase_rule() -> None:
    t = pd.date_range("1980-01-01", "2000-12-01", freq="MS")
    anomaly = np.where((t >= "1990-06-01") & (t <= "1991-06-01"), 1.5, 0.0)
    lat = np.arange(-6, 7, 2.0)
    lon = np.arange(-172, -117, 4.0)
    sst = 27 + anomaly[:, None, None] + 0 * lat[None, :, None] + 0 * lon[None, None, :]
    da = xr.DataArray(
        sst,
        coords={"time": t, "latitude": lat, "longitude": lon},
        dims=["time", "latitude", "longitude"],
    )
    table = climate.nino34(da, (1980, 2000))
    during = table[(table["time"] >= "1990-09-01") & (table["time"] <= "1991-03-01")]
    assert (during["phase"] == "el_nino").all()
    assert (table[table["time"] < "1989-01-01"]["phase"] == "neutral").all()


def test_nao_recovers_planted_dipole() -> None:
    years = np.arange(1980, 2021)
    t = pd.date_range("1979-12-01", "2021-02-01", freq="MS")
    lat = np.arange(20, 81, 5.0)
    lon = np.arange(-90, 41, 10.0)
    rng = np.random.default_rng(5)
    index = rng.normal(0, 1, len(years))
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    dipole = -np.exp(-((yy - 65) ** 2 + (xx + 20) ** 2) / 300) + np.exp(
        -((yy - 38) ** 2 + (xx + 20) ** 2) / 300
    )
    data = np.zeros((len(t), len(lat), len(lon)))
    for i, when in enumerate(t):
        season_year = when.year + (1 if when.month == 12 else 0)
        if season_year in years:
            data[i] = (
                1013
                + 5 * index[season_year - years[0]] * dipole
                + rng.normal(0, 0.3, dipole.shape)
            )
        else:
            data[i] = 1013
    da = xr.DataArray(
        data,
        coords={"time": t, "latitude": lat, "longitude": lon},
        dims=["time", "latitude", "longitude"],
    )
    table, _pattern, explained = climate.nao_eof(da, "DJF")
    joined = table.set_index("year")["nao"].reindex(years)
    assert (
        np.corrcoef(joined, index)[0, 1] > 0.95
    )  # positive NAO = low pressure over Iceland, as planted
    assert explained > 0.5


async def test_evaluate_run_bias(tmp_path: Path) -> None:
    t = pd.date_range("2000-01-01", "2002-12-01", freq="MS")
    lat = np.arange(-60, 61, 10.0)
    lon = np.arange(-180, 180, 10.0)
    ref = xr.Dataset(
        {
            "tas": (
                ("time", "lat", "lon"),
                280
                + np.random.default_rng(1).normal(0, 1, (len(t), len(lat), len(lon))),
            )
        },
        coords={"time": t, "lat": lat, "lon": lon},
    )
    run = ref + 0.5
    rp, mp = tmp_path / "ref.nc", tmp_path / "run.nc"
    ref.to_netcdf(rp)
    run.to_netcdf(mp)
    out = await call(
        "evaluate_run",
        run=str(mp),
        reference=str(rp),
        variable="tas",
        regions={"tropics": [-180, 180, -20, 20]},
    )
    assert out["global_mean_bias"] == pytest.approx(0.5, abs=1e-6)
    assert out["seasonal_cycle_rmse_by_region"]["tropics"] == pytest.approx(
        0.5, abs=1e-6
    )


async def test_errors_are_typed(tmp_path: Path) -> None:
    with pytest.raises(ToolError, match=r"^bad_input: "):
        await call(
            "exceedance",
            ensemble=str(tmp_path / "none.nc"),
            variable="tp",
            thresholds=[1.0],
        )
    with pytest.raises(ToolError, match=r"^bad_input: give a point or a box"):
        await call("ensemble_fetch", param="tp", steps=[0, 24])


async def test_detect_blobs_tracks_a_moving_blob(tmp_path: Path, te_bin: str) -> None:
    t = pd.date_range("2026-01-01", periods=12, freq="6h")
    lat = np.arange(20, 50.1, 0.5)
    lon = np.arange(-150, -110.1, 0.5)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    u = np.zeros((len(t), len(lat), len(lon)))
    for i in range(len(t)):
        u[i] = 600 * np.exp(-(((yy - 35) / 2) ** 2 + ((xx - (-145 + 2 * i)) / 8) ** 2))
    ds = xr.Dataset(
        {
            "viwve": (("time", "lat", "lon"), u),
            "viwvn": (("time", "lat", "lon"), np.zeros_like(u)),
        },
        coords={"time": t, "lat": lat, "lon": lon},
    )
    path = tmp_path / "ivt.nc"
    ds.to_netcdf(path)
    out = await call(
        "detect_blobs",
        grid=str(path),
        variable="ivt",
        threshold="ivt,>=,250,0",
        derive_magnitude=["viwve", "viwvn"],
        min_time="4",
        regional=True,
    )
    assert out["objects"] == 1
    assert out["objects_table"][0]["steps"] == 12


async def test_tc_track_follows_a_planted_low(tmp_path: Path, te_bin: str) -> None:
    init = np.datetime64("2026-09-01T00:00")
    leads = (np.arange(0, 13) * 6).astype("timedelta64[h]").astype("timedelta64[ns]")
    lat = np.arange(5, 35.1, 0.5)
    lon = np.arange(-80, -40.1, 0.5)
    yy, xx = np.meshgrid(lat, lon, indexing="ij")
    msl = np.zeros((1, len(leads), len(lat), len(lon)))
    for i in range(len(leads)):
        cy, cx = 15 + 0.5 * i, -50 - 1.0 * i
        msl[0, i] = 101300 - 4000 * np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / 4.0))
    ds = xr.Dataset(
        {
            "msl": (("time", "lead_time", "lat", "lon"), msl),
            "u10m": (("time", "lead_time", "lat", "lon"), np.zeros_like(msl)),
            "v10m": (("time", "lead_time", "lat", "lon"), np.zeros_like(msl)),
        },
        coords={"time": [init], "lead_time": leads, "lat": lat, "lon": lon},
    )
    path = tmp_path / "fc.nc"
    ds.to_netcdf(path)
    out = await call(
        "tc_track",
        forecast=str(path),
        model="synthetic",
        start=[-50, 15],
        min_time="24h",
    )
    assert out["tracks"] == 1
    track = pd.read_parquet(out["outputs"][0]["path"]).sort_values("time")
    assert track["longitude"].iloc[0] == pytest.approx(-50, abs=0.6)
    assert track["longitude"].iloc[-1] == pytest.approx(-62, abs=0.6)
    assert track["mslp_hpa"].min() == pytest.approx(973.0, abs=1.0)


async def test_gpu_tracker_refuses_without_cuda(tmp_path: Path) -> None:
    with pytest.raises(
        ToolError, match=r"^backend_unavailable: Earth2Studio's TCTracker"
    ):
        await call(
            "tc_track",
            forecast=str(tmp_path / "x.nc"),
            model="m",
            backend="earth2studio",
        )
