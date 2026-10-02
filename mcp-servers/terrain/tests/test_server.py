"""In-memory MCP tests for the terrain server.

Exercises the tools, resource, and prompt through ``fastmcp.Client`` so the
full registration + schema path is covered, plus error handling via ToolError.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from terrain_mcp.server import mcp


@pytest.mark.asyncio
async def test_tools_registered() -> None:
    async with Client(mcp) as client:
        tools = {t.name for t in await client.list_tools()}
    assert {"dem_terrain", "pointcloud_read"} <= tools


@pytest.mark.asyncio
async def test_resource_and_prompt_registered() -> None:
    async with Client(mcp) as client:
        resources = {str(r.uri) for r in await client.list_resources()}
        prompts = {p.name for p in await client.list_prompts()}
    assert "terrain://capabilities" in resources
    assert "terrain_suitability_workflow" in prompts


@pytest.mark.asyncio
async def test_dem_terrain_runs_on_csv(dem_csv: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "dem_terrain",
            {
                "filepath": str(dem_csv),
                "cell_size": 1.0,
                "slope_max_degrees": 90.0,
            },
        )
    data = result.data
    assert data["ok"] is True
    assert data["shape"] == [5, 5]
    assert data["valid_cell_count"] == 25
    # Tilted plane: every finite cell satisfies a 90-degree slope cap.
    assert data["suitable_cell_count"] == 25
    assert data["elevation"]["count"] == 25
    assert data["slope_degrees"]["max"] is not None


@pytest.mark.asyncio
async def test_dem_terrain_elevation_filter(dem_csv: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool(
            "dem_terrain",
            {"filepath": str(dem_csv), "elevation_min": 14.0},
        )
    data = result.data
    # Columns with elevation >= 14 are cols 2,3,4 -> 3 cols * 5 rows = 15 cells.
    assert data["suitable_cell_count"] == 15


@pytest.mark.asyncio
async def test_dem_terrain_reads_npz(dem_npz: Path) -> None:
    async with Client(mcp) as client:
        result = await client.call_tool("dem_terrain", {"filepath": str(dem_npz)})
    data = result.data
    assert data["ok"] is True
    assert data["metadata"]["array_key"] == "dem"
    assert data["shape"] == [4, 4]


@pytest.mark.asyncio
async def test_pointcloud_read_runs_and_writes_dem(
    points_csv: Path, tmp_path: Path
) -> None:
    out_dem = tmp_path / "gridded.csv"
    async with Client(mcp) as client:
        result = await client.call_tool(
            "pointcloud_read",
            {
                "filepath": str(points_csv),
                "grid_cell_size": 1.0,
                "output_dem_path": str(out_dem),
            },
        )
    data = result.data
    assert data["ok"] is True
    assert data["point_count"] == 16
    assert data["grid_shape"] == [4, 4]
    assert data["filled_cell_count"] == 16
    assert data["output_dem_path"] == str(out_dem)
    assert out_dem.is_file()
    written = np.loadtxt(out_dem, delimiter=",")
    assert written.shape == (4, 4)


@pytest.mark.asyncio
async def test_pointcloud_then_dem_roundtrip(points_csv: Path, tmp_path: Path) -> None:
    out_dem = tmp_path / "roundtrip.csv"
    async with Client(mcp) as client:
        await client.call_tool(
            "pointcloud_read",
            {"filepath": str(points_csv), "output_dem_path": str(out_dem)},
        )
        dem_result = await client.call_tool("dem_terrain", {"filepath": str(out_dem)})
    assert dem_result.data["ok"] is True
    assert dem_result.data["shape"] == [4, 4]


@pytest.mark.asyncio
async def test_dem_terrain_missing_file_raises_tool_error() -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool("dem_terrain", {"filepath": "/no/such/dem.csv"})


@pytest.mark.asyncio
async def test_dem_terrain_bad_cell_size_raises_tool_error(dem_csv: Path) -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "dem_terrain", {"filepath": str(dem_csv), "cell_size": 0.0}
            )


@pytest.mark.asyncio
async def test_pointcloud_read_missing_file_raises_tool_error() -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError):
            await client.call_tool(
                "pointcloud_read", {"filepath": "/no/such/points.csv"}
            )


@pytest.mark.asyncio
async def test_capabilities_resource_payload() -> None:
    async with Client(mcp) as client:
        contents = await client.read_resource("terrain://capabilities")
    import json

    payload = json.loads(contents[0].text)
    assert payload["tools"] == ["dem_terrain", "pointcloud_read"]
    assert "geotiff" in payload["dem_formats"]["optional"]


@pytest.mark.asyncio
async def test_slope_next_to_nodata_is_excluded_not_computed_from_a_fill(
    tmp_path: Path,
) -> None:
    rows, cols = np.mgrid[0:20, 0:30]
    plane = 100.0 + 0.5 * cols  # slope = atan(0.5) = 26.565 degrees everywhere
    plane[10, 15] = -9999.0
    path = tmp_path / "nodata.npy"
    np.save(path, plane)
    async with Client(mcp) as client:
        data = (
            await client.call_tool(
                "dem_terrain", {"filepath": str(path), "nodata": -9999}
            )
        ).data
        strict = (
            await client.call_tool(
                "dem_terrain",
                {"filepath": str(path), "nodata": -9999, "slope_max_degrees": 30},
            )
        ).data
    assert data["valid_cell_count"] == 599
    # The no-data cell and its four gradient neighbours have no defined slope.
    assert data["slope_degrees"]["count"] == data["aspect_degrees"]["count"] == 595
    assert data["slope_degrees"]["min"] == pytest.approx(26.565, abs=1e-3)
    assert data["slope_degrees"]["max"] == pytest.approx(26.565, abs=1e-3)
    assert data["suitable_cell_count"] == 599  # no slope criterion: elevation only
    assert strict["suitable_cell_count"] == 595


@pytest.mark.asyncio
async def test_dem_terrain_rejects_inverted_elevation_range(dem_csv: Path) -> None:
    async with Client(mcp) as client:
        with pytest.raises(ToolError, match="elevation_min"):
            await client.call_tool(
                "dem_terrain",
                {"filepath": str(dem_csv), "elevation_min": 200, "elevation_max": 100},
            )


@pytest.mark.asyncio
async def test_pointcloud_read_says_which_columns_were_xyz(
    dem_csv: Path, points_csv: Path
) -> None:
    async with Client(mcp) as client:
        grid = await client.call_tool("pointcloud_read", {"filepath": str(dem_csv)})
        named = await client.call_tool("pointcloud_read", {"filepath": str(points_csv)})
    assert grid.data["metadata"]["xyz_columns"] == (
        "no column names: columns 1, 2, 3 of 5 read as x, y, z"
    )
    assert named.data["metadata"]["xyz_columns"] == "header columns named x, y, z"


@pytest.mark.asyncio
async def test_geotiff_error_does_not_suggest_an_impossible_install(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "rasterio", None)  # force ImportError
    path = tmp_path / "dem.tif"
    path.write_bytes(b"II*\x00")
    async with Client(mcp) as client:
        with pytest.raises(ToolError) as excinfo:
            await client.call_tool("dem_terrain", {"filepath": str(path)})
    message = str(excinfo.value)
    assert "not included in this installation" in message and "CSV" in message
    assert "pip install" not in message


def test_server_version_matches_release_manifest() -> None:
    import re

    manifest = (Path(__file__).parents[1] / "clio-server.toml").read_text()
    assert mcp.version == re.search(r'^version = "(.+)"', manifest, re.M).group(1)
