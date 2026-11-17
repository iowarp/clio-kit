"""Implementation tests against known answers."""

from __future__ import annotations

import json
from pathlib import Path

import gsw
import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_mcp.implementation import (
    cruise,
    events,
    hydrography,
    model_checks,
    sections,
    timeseries,
)
from ocean_mcp.implementation import transport as transport_impl
from ocean_mcp.implementation.common import OceanToolError
from ocean_mcp.implementation.profiles_io import read_casts, read_whp


def test_cnv_reader_parses_positions_and_channels(cast_files: list[Path]) -> None:
    table, formats = read_casts(cast_files)
    assert formats == ["sbe-cnv"]
    assert table["profile"].nunique() == 3
    assert set(table["latitude"].round(3)) == {30.0, 31.0, 32.0}
    assert (table["longitude"] == -40.0).all()
    assert table["time"].iloc[0] == pd.Timestamp("2013-08-05 12:00:00")


def test_whp_reader_drops_bad_flags(whp_file: Path) -> None:
    table = read_whp(whp_file)
    assert table["profile"].iloc[0] == "33RO20130803-007-1"
    assert np.isnan(table.loc[table["pressure"] == 1100.0, "temperature"]).all()
    assert table["time"].iloc[0] == pd.Timestamp("2013-08-05 15:30")


def test_teos10_matches_gsw_check_value() -> None:
    # Reference Salinity of SP 35 is 35.16504 g/kg; SA adds the regional anomaly (< 0.002 here).
    table = pd.DataFrame(
        {
            "profile": ["a"],
            "time": [pd.Timestamp("2020")],
            "latitude": [0.0],
            "longitude": [0.0],
            "pressure": [0.0],
            "temperature": [10.0],
            "salinity": [35.0],
        }
    )
    out = hydrography.add_teos10(table)
    assert gsw.SR_from_SP(35.0) == pytest.approx(35.16504, abs=1e-5)
    assert out["SA"].iloc[0] == pytest.approx(35.16504, abs=2e-3)
    assert out["SA"].iloc[0] == pytest.approx(
        gsw.SA_from_SP(35.0, 0.0, 0.0, 0.0), abs=1e-12
    )
    assert out["CT"].iloc[0] == pytest.approx(
        gsw.CT_from_t(out["SA"].iloc[0], 10.0, 0.0), abs=1e-9
    )


def test_mixed_layer_depth_recovers_planted_depth(cast_files: list[Path]) -> None:
    table = hydrography.add_teos10(read_casts(cast_files)[0])
    metrics = hydrography.profile_metrics(
        table,
        criterion="temperature_threshold",
        threshold=0.2,
        reference_pressure=10.0,
        stratification_range=(20, 200),
    )
    mlds = dict(zip(metrics["profile"], metrics["mld_dbar"], strict=True))
    # 0.2 °C below a 0.05 °C/dbar gradient is 4 dbar below the planted MLD.
    assert mlds["cast00"] == pytest.approx(54.0, abs=1.0)
    assert mlds["cast01"] == pytest.approx(84.0, abs=1.0)
    assert mlds["cast02"] == pytest.approx(124.0, abs=1.0)


def test_water_masses_from_builtin_definition(whp_file: Path) -> None:
    _, masses, citation = hydrography.parse_definition("atlantic_emery_meincke_1986")
    assert "Emery" in citation
    table, _counts = hydrography.assign_water_masses(read_whp(whp_file), masses)
    by_p = dict(zip(table["pressure"], table["water_mass"], strict=True))
    assert by_p[1000.0] == "Mediterranean Water"
    assert by_p[2500.0] == "North Atlantic Deep Water"
    assert by_p[1100.0] == "missing"  # bad flag removed the temperature


def test_custom_definition_rejects_malformed() -> None:
    with pytest.raises(OceanToolError) as exc:
        hydrography.parse_definition({"name": "x", "masses": [{"name": "A"}]})
    assert exc.value.reason == "bad_input"


def test_section_from_profiles_orders_by_latitude(cast_files: list[Path]) -> None:
    table = read_casts(cast_files)[0]
    ds = sections.section_from_profiles(
        table, ["temperature"], order="latitude", pressure_step=10
    )
    assert ds.sizes["distance"] == 3
    assert ds["distance"].values[-1] == pytest.approx(
        222.4, abs=1.0
    )  # two degrees of latitude
    assert json.loads(ds.attrs["path"])["type"] == "LineString"


def test_section_transport_of_uniform_flow(velocity_grid: Path) -> None:
    ds = xr.open_dataset(velocity_grid)
    # West to east along the equator, 2 degrees (about 222 km); positive is to the right (south).
    _out, numbers = transport_impl.section_transport(
        ds,
        [[-1.0, 0.0], [1.0, 0.0]],
        u="uo",
        v="vo",
        temperature="thetao",
        step_km=5.0,
        reference_temperature=0.0,
    )
    expected_sv = -0.1 * 222_390 * 1000 / 1e6  # northward flow is to the left: negative
    assert numbers["mean_volume_transport_sv"] == pytest.approx(expected_sv, rel=0.03)
    assert "approximate" in numbers["approximation"]
    heat = numbers["mean_heat_transport_pw"]
    assert heat == pytest.approx(1025 * 3992 * expected_sv * 1e6 * 10 / 1e15, rel=0.03)


def test_overturning_depth_space(velocity_grid: Path) -> None:
    ds = xr.open_dataset(velocity_grid)
    psi = transport_impl.overturning(
        ds,
        transport=None,
        velocity="vo",
        space="depth",
        density=None,
        density_bins=None,
    )
    _series, numbers = transport_impl.overturning_at(psi, 0.0)
    width = 6_371_000 * np.deg2rad(20.25)  # 81 cells of 0.25 degrees at the equator
    assert numbers["mean_max_sv"] == pytest.approx(0.1 * width * 1000 / 1e6, rel=0.02)


def test_harmonics_recovers_constituents(tide_series: Path) -> None:
    data = timeseries.read_series(tide_series, ["height"])
    table, _recon, numbers = timeseries.harmonic_analysis(
        data,
        height="height",
        u=None,
        v=None,
        latitude=40.0,
        constituents=["M2", "K1", "S2", "O1"],
        method="ols",
    )
    rows = table.set_index("constituent")
    assert rows.loc["M2", "amplitude"] == pytest.approx(1.0, abs=0.01)
    assert rows.loc["K1", "amplitude"] == pytest.approx(0.3, abs=0.02)
    assert numbers["variance_explained"] > 0.99


def test_rotary_spectrum_inertial_peak_is_clockwise() -> None:
    lat = 45.0
    f = 2 * 7.2921e-5 * np.sin(np.deg2rad(lat))
    t = pd.date_range("2020-01-01", periods=24 * 64, freq="h", tz="UTC")
    s = (t - t[0]).total_seconds().to_numpy()
    # Northern-hemisphere inertial motion turns clockwise: u = cos(ft), v = -sin(ft).
    frame = pd.DataFrame({"u": np.cos(f * s), "v": -np.sin(f * s)}, index=t)
    table, numbers = timeseries.spectra(
        frame, u="u", v="v", segment_days=16, latitude=lat
    )
    assert table["clockwise"].max() > 100 * table["anticlockwise"].max()
    assert numbers["peak_frequency_cpd"]["clockwise"] == pytest.approx(
        numbers["inertial_frequency_cpd"], abs=0.08
    )


def test_sea_level_local_part_is_land_motion() -> None:
    t = pd.date_range("1995-01-01", "2020-12-01", freq="MS", tz="UTC")
    years = (t - t[0]).days / 365.25
    alt = pd.Series(3.0 * years + 20 * np.sin(2 * np.pi * years), index=t)
    gauge = alt + 2.0 * years  # 2 mm/yr subsidence shows as extra relative rise
    _, numbers = timeseries.sea_level_decompose(gauge, alt)
    assert numbers["local_trend"]["trend_per_year"] == pytest.approx(2.0, abs=0.05)


def test_hobday_detects_planted_heatwave(sst_series: Path) -> None:
    s = timeseries.read_series(sst_series, ["sst"])["sst"]
    frame = events.climatology(
        s, baseline=(1983, 2012), percentile=90, window_days=11, smooth_days=31
    )
    found = events.detect_events(frame, min_duration=5, max_gap=2)
    planted = found[(found["start"] >= "2022-05-25") & (found["start"] <= "2022-06-20")]
    assert len(planted) == 1
    event = planted.iloc[0]
    assert 20 <= event["duration_days"] <= 40
    assert event["intensity_max"] == pytest.approx(2.0, abs=0.6)
    # Noise sd 0.3 puts the 90th percentile ~0.38 above climatology; a +2 peak is > 4x that.
    assert event["intensity_ratio"] > 4
    assert event["category"] == "Extreme"


def test_hobday_refuses_short_baseline(sst_series: Path) -> None:
    s = timeseries.read_series(sst_series, ["sst"])["sst"]
    with pytest.raises(OceanToolError, match="baseline"):
        events.climatology(
            s, baseline=(2022, 2022), percentile=90, window_days=11, smooth_days=31
        )


def test_bloom_start_median_plus_five_percent() -> None:
    t = pd.date_range("2010-01-01", "2014-12-31", freq="D", tz="UTC")
    doy = t.day_of_year.to_numpy()
    chl = 0.3 + 2.0 * np.exp(-((doy - 100) ** 2) / (2 * 15**2))
    table, numbers = events.bloom_start(
        pd.Series(chl, index=t), method="median_plus_5pct", season=(1, 7), smooth_days=1
    )
    starts = pd.to_datetime(table["bloom_start"]).dt.day_of_year
    assert starts.between(50, 80).all()
    assert numbers["years_with_start"] == 5


def test_mixed_layer_budget_closes_for_flux_only_warming() -> None:
    t = pd.date_range("2022-06-01", periods=31, freq="D")
    h = 20.0
    q = 200.0
    warming_per_day = q / (1025 * 3992 * h) * 86400
    frame = pd.DataFrame(
        {"T": 20 + warming_per_day * np.arange(31), "Q": q, "H": h}, index=t
    )
    _table, numbers = events.mixed_layer_budget(
        frame,
        temperature="T",
        net_flux="Q",
        shortwave=None,
        mld="H",
        start="2022-06-01",
        end="2022-07-01",
    )
    assert numbers["residual_contribution_K"] == pytest.approx(0.0, abs=1e-6)
    assert numbers["surface_flux_fraction"] == pytest.approx(1.0, abs=0.05)


def test_cruise_schedule_and_feature_plan() -> None:
    depart = pd.Timestamp("2026-11-20T06:00", tz="UTC")
    _table, numbers = cruise.schedule(
        [{"longitude": 1.0, "latitude": 0.0}],
        port=(0.0, 0.0),
        depart=depart,
        speed_knots=10.0,
        station_hours=2.0,
        days_available=1.0,
    )
    assert numbers["steaming_km"] == pytest.approx(222.4, abs=0.5)
    assert numbers["days_used"] == pytest.approx((222.4 / 18.52 + 2) / 24, abs=0.01)
    track = pd.DataFrame(
        {
            "time": pd.date_range("2026-11-20", periods=11, freq="D", tz="UTC"),
            "longitude": np.linspace(2.0, 1.0, 11),
            "latitude": 0.0,
        }
    )
    plan, numbers = cruise.plan_across_feature(
        track,
        port=(0.0, 0.0),
        depart=depart,
        speed_knots=10.0,
        station_hours=3.0,
        days_available=5.0,
        stations=5,
        line_length_km=100.0,
        bearing_degrees=0.0,
    )
    assert len(plan) == 5
    assert numbers["fits"]
    assert 1.8 < numbers["feature_position_at_arrival"]["longitude"] < 2.0


def test_drift_flags_a_warming_level() -> None:
    t = pd.date_range("2000-01-01", periods=24, freq="MS")
    z = np.array([5.0, 500.0])
    lat = np.array([0.0, 1.0])
    lon = np.array([0.0, 1.0])
    temp = np.zeros((24, 2, 2, 2))
    temp[:, 0] = 15.0
    temp[:, 1] = 5.0 + 0.01 * np.arange(24)[:, None, None]
    ds = xr.Dataset(
        {"thetao": (("time", "depth", "latitude", "longitude"), temp)},
        coords={"time": t, "depth": z, "latitude": lat, "longitude": lon},
    )
    _, verdicts = model_checks.drift(
        ds,
        temperature="thetao",
        salinity=None,
        u=None,
        v=None,
        window=12,
        tolerance={"thetao": 0.05},
    )
    states = dict(zip(verdicts["level"], verdicts["equilibrated"], strict=True))
    assert states[5.0] is True or states[5.0] == True
    assert not states[500.0]


def test_colocate_reports_known_bias(velocity_grid: Path) -> None:
    ds = xr.open_dataset(velocity_grid)
    obs = pd.DataFrame(
        {
            "profile": ["p"] * 3,
            "time": pd.to_datetime(["2020-01-02"] * 3, utc=True),
            "latitude": [1.0] * 3,
            "longitude": [2.0] * 3,
            "pressure": [10.0, 100.0, 600.0],
            "temperature": [9.0, 9.5, 10.0],
        }
    )
    _pairs, _stats, numbers = model_checks.colocate(
        ds,
        obs,
        model_variable="thetao",
        observed_variable="temperature",
        depth_bands=[0, 50, 200, 1000],
        region_step_degrees=5,
        max_time_offset_days=5,
    )
    assert numbers["matched"] == 3
    assert numbers["overall_bias"] == pytest.approx(0.5, abs=1e-6)


def test_cache_dir_is_under_the_configured_root(tmp_path, monkeypatch) -> None:
    from ocean_mcp.implementation.common import cache_dir

    monkeypatch.setenv("OCEAN_MCP_CACHE_DIR", str(tmp_path / "c"))
    assert cache_dir("argopy") == tmp_path / "c" / "argopy"
    assert (tmp_path / "c" / "argopy").is_dir()
    monkeypatch.delenv("OCEAN_MCP_CACHE_DIR")
    monkeypatch.setenv("OCEAN_MCP_OUTPUT_DIR", str(tmp_path / "out"))
    assert cache_dir("argopy") == tmp_path / "out" / ".cache" / "argopy"
