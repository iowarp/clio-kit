"""mom6_preflight on a synthetic run directory with planted mistakes."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from ocean_mcp.implementation import mom6

from .test_tools import call


def make_run(root: Path, *, channel: bool, segment_end: str = "2020-03-01") -> Path:
    run = root / "run"
    inputs = run / "INPUT"
    inputs.mkdir(parents=True)
    depth = np.full((20, 24), 1000.0)
    depth[:, :8] = 0.0  # land to the west
    if channel:
        depth[10, 3] = 500.0  # a one-cell channel poking into the land ...
        depth[10, 4:8] = 500.0  # ... joined to the ocean along row 10
    depth[2, 2] = 300.0  # an isolated wet cell
    xr.Dataset({"depth": (("ny", "nx"), depth)}).to_netcdf(inputs / "ocean_topog.nc")
    for k in (1, 2):
        times = pd.date_range("2020-01-01", segment_end, freq="D")
        xr.Dataset(
            {f"u_segment_00{k}": (("time",), np.zeros(len(times)))},
            coords={"time": times},
        ).to_netcdf(inputs / f"forcing_obc_segment_00{k}.nc")
    (run / "MOM_input").write_text(
        'TOPO_FILE = "ocean_topog.nc"\n'
        "MINIMUM_DEPTH = 10.0 ! m\n"
        "OBC_NUMBER_OF_SEGMENTS = 2\n"
        'OBC_SEGMENT_001 = "J=0,I=0:N,FLATHER,ORLANSKI,NUDGED"\n'
        'OBC_SEGMENT_001_DATA = "U=file:forcing_obc_segment_001.nc(u_segment_001)"\n'
        'OBC_SEGMENT_002 = "I=N,J=0:N,FLATHER,ORLANSKI,NUDGED"\n'
        'OBC_SEGMENT_002_DATA = "U=file:forcing_obc_segment_002.nc(u_segment_002)"\n'
        "LAYOUT = 2, 2\n"
        'MASKTABLE = "mask_table.1.2x2"\n'
    )
    (inputs / "mask_table.1.2x2").write_text("1\n2,2\n1,1\n")
    (run / "input.nml").write_text(
        "&ocean_solo_nml\n    date_init = 2020,1,1,0,0,0\n    months = 1\n    days = 0\n/\n"
    )
    return run


def test_planted_channel_and_isolated_cell_are_caught(tmp_path: Path) -> None:
    checks, _facts = mom6.preflight(
        make_run(tmp_path, channel=True), input_dir=None, kept_cells=[]
    )
    status = {c.name: c for c in checks}
    assert status["open_boundaries"].status == "passed"
    assert status["dates_cover_run"].status == "passed"
    kinds = {(item["j"], item["i"]): item["kind"] for item in status["land_mask"].items}
    assert kinds[(2, 2)] == "isolated wet cell"
    assert kinds[(10, 3)] == "one-cell channel"
    # The south-west PE (1,1) is masked out but holds the planted wet cells.
    assert status["layout"].status == "failed"


def test_kept_cells_are_not_flagged(tmp_path: Path) -> None:
    run = make_run(tmp_path, channel=True)
    checks, _ = mom6.preflight(
        run,
        input_dir=None,
        kept_cells=[[10, 3], [10, 4], [10, 5], [10, 6], [10, 7], [2, 2]],
    )
    land = next(c for c in checks if c.name == "land_mask")
    assert land.status == "passed"


def test_short_boundary_files_fail_the_dates_check(tmp_path: Path) -> None:
    checks, _ = mom6.preflight(
        make_run(tmp_path, channel=False, segment_end="2020-01-20"),
        input_dir=None,
        kept_cells=[],
    )
    dates = next(c for c in checks if c.name == "dates_cover_run")
    assert dates.status == "failed"
    assert "run needs" in dates.items[0]


def test_tidal_harmonics_are_not_date_checked(tmp_path: Path) -> None:
    run = make_run(tmp_path, channel=False)
    inputs = run / "INPUT"
    # Amplitude and phase per constituent, with the single time record tidal
    # files usually carry: constant in time, they apply to the whole run.
    xr.Dataset(
        {
            "zamp_segment_001": (("nz", "nx"), np.ones((8, 24))),
            "zphase_segment_001": (("nz", "nx"), np.zeros((8, 24))),
        },
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    ).to_netcdf(inputs / "tides_obc_segment_001.nc")
    text = (
        (run / "MOM_input")
        .read_text()
        .replace(
            'file:forcing_obc_segment_001.nc(u_segment_001)"',
            "file:forcing_obc_segment_001.nc(u_segment_001),"
            'SSHamp=file:tides_obc_segment_001.nc(zamp),SSHphase=file:tides_obc_segment_001.nc(zphase)"',
        )
    )
    (run / "MOM_input").write_text(
        text.replace("(zamp)", "(zamp_segment_001)").replace(
            "(zphase)", "(zphase_segment_001)"
        )
    )
    checks, _ = mom6.preflight(run, input_dir=None, kept_cells=[])
    dates = next(c for c in checks if c.name == "dates_cover_run")
    assert dates.status == "passed", dates.items
    assert "tides_obc_segment_001.nc" in dates.detail


def test_time_varying_field_with_one_record_still_fails(tmp_path: Path) -> None:
    run = make_run(tmp_path, channel=False)
    xr.Dataset(
        {"u_segment_001": (("time",), np.zeros(1))},
        coords={"time": pd.date_range("2020-01-01", periods=1)},
    ).to_netcdf(run / "INPUT" / "forcing_obc_segment_001.nc", mode="w")
    checks, _ = mom6.preflight(run, input_dir=None, kept_cells=[])
    dates = next(c for c in checks if c.name == "dates_cover_run")
    assert dates.status == "failed"
    assert any("forcing_obc_segment_001.nc" in item for item in dates.items)


async def test_tool_reports_partial_with_failed_checks(tmp_path: Path) -> None:
    out = await call("mom6_preflight", run_dir=str(make_run(tmp_path, channel=True)))
    assert out["status"] == "partial" and out["outcome"] == "checks_failed"
    assert "land_mask" in out["summary"]
    assert out["parameters_read"]["OBC_NUMBER_OF_SEGMENTS"] == "2"


def test_missing_input_raises_typed(tmp_path: Path) -> None:
    from ocean_mcp.implementation.common import OceanToolError

    with pytest.raises(OceanToolError, match="no MOM_input"):
        mom6.preflight(tmp_path, input_dir=None, kept_cells=[])


def test_surface_forcing_without_a_boundary_layer_scheme_fails() -> None:
    forced = {"BUOY_CONFIG": "file", "KD": "1.0E-05"}
    check = mom6.surface_mixing_check(forced)
    assert check.status == "failed" and "top layer" in check.detail
    assert "shortwave is absorbed at the surface" in check.detail
    assert (
        mom6.surface_mixing_check({**forced, "ENERGETICS_SFC_PBL": "True"}).status
        == "passed"
    )
    assert mom6.surface_mixing_check({**forced, "USE_KPP": "T"}).status == "passed"
    assert mom6.surface_mixing_check({"BUOY_CONFIG": "zero"}).status == "not_checked"
    assert mom6.surface_mixing_check({}).status == "not_checked"


def test_the_live_tasmania_configuration_is_caught() -> None:
    # The keys of the forced 20-day Tasmania run that warmed 2 °C too fast.
    params = {
        "BUOY_CONFIG": "file",
        "WIND_CONFIG": "file",
        "KD": "1.0E-05",
        "KV": "1.0E-04",
    }
    assert mom6.surface_mixing_check(params).status == "failed"
