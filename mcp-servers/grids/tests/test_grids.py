"""Edits, logs and replay on a small synthetic bathymetry, through the real server."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import xarray as xr
from fastmcp import Client
from fastmcp.exceptions import ToolError

from grids_mcp import editing
from grids_mcp.server import EDITOR_URI, mcp


def bathymetry(path: Path, *, deeper: float = 0.0) -> Path:
    """A 1-degree basin: depth positive down, land (0) on the borders, an island
    at 10E 5N, and a two-cell lake at 2-3E 8N cut off from the open water."""

    lat = np.arange(0.5, 10, 1.0)
    lon = np.arange(0.5, 20, 1.0)
    depth = np.full((lat.size, lon.size), 1000.0 + deeper)
    depth[0, :] = depth[-1, :] = depth[:, 0] = depth[:, -1] = 0.0
    depth[5, 10] = 0.0  # island
    depth[6:9, 1:5] = 0.0  # land around the lake
    depth[7, 2:4] = 40.0  # the lake
    depth[3, 15] = 8.0  # a shallow cell
    xr.Dataset(
        {"depth": (("lat", "lon"), depth, {"units": "m", "positive": "down"})},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    return path


async def call(name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    async with Client(mcp) as client:
        result = await client.call_tool(name, arguments)
    assert result.structured_content is not None
    return result.structured_content


async def test_open_describes_the_grid_and_names_the_editor(tmp_path: Path) -> None:
    base = bathymetry(tmp_path / "topo.nc")
    out = await call("grid_open", {"path": str(base)})
    assert out["grid"]["shape"] == [10, 20]
    assert out["grid"]["value_range"] == [0.0, 1000.0]
    assert out["output"].endswith("topo.edited.nc")
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        html = await client.read_resource(EDITOR_URI)
    assert tools["grid_open"].meta["ui"]["resourceUri"] == EDITOR_URI
    assert "grid_edit" in html[0].text and "ui/initialize" in html[0].text


async def test_edits_apply_in_order_and_the_log_records_hashes(tmp_path: Path) -> None:
    base = bathymetry(tmp_path / "topo.nc")
    before = base.read_bytes()
    edits = [
        {"op": "clip", "min": 10, "wet": "positive", "note": "minimum depth"},
        {"op": "fill_enclosed", "wet": "positive", "land_value": 0},
        {"op": "set", "value": 0, "region": {"points": [[12.4, 5.6]]}, "note": "close"},
        {"op": "set", "value": 500, "region": {"box": [10.1, 5.1, 10.9, 5.9]}},
    ]
    out = await call("grid_edit", {"path": str(base), "edits": edits})
    assert [edit["cells_changed"] for edit in out["edits"]] == [1, 2, 1, 1]
    result = xr.open_dataset(out["output"])["depth"].values
    assert result[3, 15] == 10.0  # clipped up to the minimum depth
    assert (result[7, 2:4] == 0).all()  # the lake is land
    assert result[5, 12] == 0.0 and result[5, 10] == 500.0  # closed cell, opened island
    log = json.loads(Path(out["log"]).read_text())
    assert log["schema"] == editing.LOG_SCHEMA
    assert log["base"]["hash"] == out["base_hash"] != out["result_hash"]
    assert log["result"]["hash"] == out["result_hash"]
    assert base.read_bytes() == before  # the base is never changed


async def test_undo_is_the_list_without_its_last_edit(tmp_path: Path) -> None:
    base = bathymetry(tmp_path / "topo.nc")
    one = {"op": "set", "value": 0, "region": {"box": [4, 2, 6, 3]}}
    two = {"op": "set", "value": 7, "region": {"box": [14, 2, 16, 3]}}
    both = await call("grid_edit", {"path": str(base), "edits": [one, two]})
    first = await call("grid_edit", {"path": str(base), "edits": [one]})
    again = await call("grid_edit", {"path": str(base), "edits": [one]})
    assert first["result_hash"] == again["result_hash"] != both["result_hash"]


async def test_replay_reports_a_regenerated_base(tmp_path: Path) -> None:
    base = bathymetry(tmp_path / "topo.nc")
    edits = [
        {"op": "fill_enclosed", "wet": "positive", "land_value": 0},
        {"op": "set", "value": 0, "region": {"points": [[12.4, 5.6]]}},
    ]
    first = await call("grid_edit", {"path": str(base), "edits": edits})
    same = await call(
        "grid_replay",
        {"log": first["log"], "path": str(base), "output": str(tmp_path / "same.nc")},
    )
    assert same["replay"]["same_base"] is True
    assert same["result_hash"] == first["result_hash"]

    regenerated = bathymetry(tmp_path / "topo2.nc", deeper=50.0)
    replay = await call("grid_replay", {"log": first["log"], "path": str(regenerated)})
    assert replay["replay"]["same_base"] is False
    assert "a different base" in replay["summary"]
    result = xr.open_dataset(replay["output"])["depth"].values
    assert (result[7, 2:4] == 0).all() and result[5, 12] == 0.0


async def test_typed_refusals(tmp_path: Path) -> None:
    base = bathymetry(tmp_path / "topo.nc")
    cases = [
        ({"path": str(base), "edits": [{"op": "paint"}]}, "bad_input: edits[0].op"),
        (
            {"path": str(base), "edits": [{"op": "set"}]},
            "bad_input: edits[0]: set needs",
        ),
        (
            {"path": str(base), "edits": [], "output": str(base)},
            "bad_input: output must",
        ),
        (
            {"path": str(base), "edits": [], "expect_base_hash": "0" * 16},
            "bad_input: the base grid changed",
        ),
        ({"path": str(tmp_path / "none.nc"), "edits": []}, "bad_input:"),
    ]
    async with Client(mcp) as client:
        for arguments, prefix in cases:
            with pytest.raises(ToolError) as caught:
                await client.call_tool("grid_edit", arguments)
            assert str(caught.value).startswith(prefix), str(caught.value)


def test_box_across_the_dateline_and_window_stride(tmp_path: Path) -> None:
    lat = np.arange(-80.0, 81, 20)
    lon = np.arange(0.0, 360, 30)
    path = tmp_path / "g.nc"
    xr.Dataset(
        {"z": (("lat", "lon"), np.zeros((lat.size, lon.size)))},
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    grid = editing.open_grid(path)
    mask = editing.region_mask(grid, {"box": [170, -10, -170, 10]})
    assert sorted(set(grid.lon[mask].tolist())) == [180.0]
    view = editing.window(grid, np.arange(9 * 12.0).reshape(9, 12), 8)
    assert view["shape"] == [5, 6] and view["stride"] == [2, 2]


async def test_the_clean_bathymetry_prompt_walks_the_tools() -> None:
    async with Client(mcp) as client:
        prompts = {prompt.name for prompt in await client.list_prompts()}
        result = await client.get_prompt(
            "clean_bathymetry", {"path": "/runs/topo.nc", "variable": "depth"}
        )
    assert prompts == {"clean_bathymetry"}
    text = result.messages[0].content.text
    assert "/runs/topo.nc" in text and "variable depth" in text
    for name in ("grid_open", "grid_edit", "fill_enclosed", "grid_replay"):
        assert name in text
