"""Known-answer tests: squares in UTM, flat rasters, and a 2 m surface change."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from geoprocess_mcp.server import mcp

UTM = "EPSG:32610"
X0, Y0 = 750000.0, 4320000.0  # near Lake Tahoe


async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    async with Client(mcp) as client:
        return (await client.call_tool(name, args)).data


def square(x: float, y: float, size: float) -> list[list[float]]:
    return [[x, y], [x + size, y], [x + size, y + size], [x, y + size], [x, y]]


def layer(
    path: Path, polygons: list[tuple[list[list[float]], dict[str, Any]]], crs: str = UTM
) -> str:
    import geopandas as gpd
    from shapely.geometry import Polygon

    frame = gpd.GeoDataFrame(
        [props for _, props in polygons],
        geometry=[Polygon(ring) for ring, _ in polygons],
        crs=crs,
    )
    frame.to_file(path, driver="GPKG")
    return str(path)


def raster(
    path: Path,
    values: np.ndarray,
    *,
    res: float = 10.0,
    nodata: float = -9999.0,
    crs: str = UTM,
) -> str:
    import rasterio
    from rasterio.transform import from_origin

    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=values.shape[0],
        width=values.shape[1],
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(X0, Y0 + values.shape[0] * res, res, res),
        nodata=nodata,
    ) as out:
        out.write(values.astype("float32"), 1)
    return str(path)


async def test_dissolve_counts_overlap_once(tmp_path: Path) -> None:
    # Two 1 km squares overlapping by half: covered 150 ha, layer sum 200 ha.
    habitat = layer(
        tmp_path / "habitat.gpkg",
        [
            (square(X0, Y0, 1000), {"kind": "deer"}),
            (square(X0 + 500, Y0, 1000), {"kind": "deer"}),
        ],
    )
    result = await call(
        "overlay_features", {"operation": "dissolve", "source": habitat}
    )
    assert result["area"]["covered_area_ha"] == pytest.approx(150, rel=0.01)
    assert result["overlap_removed_ha"] == pytest.approx(50, rel=0.02)
    assert result["feature_count"] == 1
    assert (
        "149.888 ha" in result["summary"]
    )  # equal-area, not the UTM grid's nominal 150 ha


async def test_intersection_of_two_layers_reports_area_by_attribute(
    tmp_path: Path,
) -> None:
    habitat = layer(tmp_path / "h.gpkg", [(square(X0, Y0, 1000), {"kind": "owl"})])
    fire = layer(
        tmp_path / "f.gpkg", [(square(X0 + 600, Y0, 1000), {"fire": "Caldor"})]
    )
    result = await call(
        "overlay_features",
        {"operation": "intersection", "source": habitat, "other": fire, "by": "fire"},
    )
    assert result["area"]["groups"]["Caldor"] == pytest.approx(40, rel=0.02)


async def test_clip_to_bbox_keeps_inside_features(tmp_path: Path) -> None:
    from pyproj import Transformer

    habitat = layer(
        tmp_path / "h.gpkg",
        [
            (square(X0, Y0, 1000), {"n": 1}),
            (square(X0 + 50000, Y0, 1000), {"n": 2}),
        ],
    )
    to_wgs = Transformer.from_crs(UTM, "EPSG:4326", always_xy=True)
    w, s = to_wgs.transform(X0 - 100, Y0 - 100)
    e, n = to_wgs.transform(X0 + 2000, Y0 + 2000)
    result = await call("clip_features", {"source": habitat, "bbox": [w, s, e, n]})
    assert result["feature_count"] == 1 and result["input_features"] == 2
    saved = json.loads(Path(result["output_path"]).read_text())
    assert saved["features"][0]["properties"]["n"] == 1


async def test_zonal_statistics_on_the_raster_grid(tmp_path: Path) -> None:
    values = np.zeros((100, 100))
    values[:, :50] = 10.0  # west half 10, east half 20
    values[:, 50:] = 20.0
    values[:10, :] = -9999.0  # a strip without data
    dem = raster(tmp_path / "dem.tif", values)
    zones = layer(
        tmp_path / "z.gpkg",
        [
            (square(X0, Y0, 400), {"zone": "west"}),
            (square(X0 + 600, Y0, 400), {"zone": "east"}),
            (square(X0 + 5000, Y0, 400), {"zone": "away"}),
        ],
    )
    result = await call(
        "zonal_statistics", {"raster": dem, "zones": zones, "by": "zone"}
    )
    rows = {r["zone"]: r for r in result["rows"]}
    assert rows["west"]["mean"] == 10.0 and rows["east"]["mean"] == 20.0
    assert rows["west"]["cells"] == 1600 and rows["west"]["coverage"] == pytest.approx(
        1.0
    )
    assert rows["away"]["cells"] == 0 and "outside" in rows["away"]["note"]
    assert "1 have no raster data" in result["summary"]


async def test_points_sample_nearest_cell_and_buffer(tmp_path: Path) -> None:
    import geopandas as gpd
    from shapely.geometry import Point

    values = np.arange(10000, dtype=float).reshape(100, 100)
    dem = raster(tmp_path / "dem.tif", values)
    stations = tmp_path / "stations.gpkg"
    gpd.GeoDataFrame(
        {"code": ["P1", "P2"]},
        geometry=[Point(X0 + 505, Y0 + 495), Point(X0 + 99999, Y0)],
        crs=UTM,
    ).to_file(stations, driver="GPKG")
    nearest = await call(
        "zonal_statistics", {"raster": dem, "zones": str(stations), "id_field": "code"}
    )
    rows = {r["point"]: r for r in nearest["rows"]}
    assert rows["P1"]["value"] == values[50, 50]
    assert rows["P2"]["value"] is None and nearest["missing"] == 1
    buffered = await call(
        "zonal_statistics",
        {"raster": dem, "zones": str(stations), "id_field": "code", "buffer_m": 20},
    )
    assert buffered["method"] == "mean within 20.0 m"
    assert {r["point"]: r for r in buffered["rows"]}["P1"]["cells"] > 1


async def test_raster_difference_resamples_and_flags_datums(tmp_path: Path) -> None:
    before = raster(tmp_path / "before.tif", np.full((100, 100), 1900.0))
    after = raster(tmp_path / "after.tif", np.full((50, 50), 1902.0), res=20.0)
    result = await call("raster_difference", {"first": after, "second": before})
    # 'after' defines the grid: 50 x 50 at 20 m.
    assert result["shape"] == [50, 50]
    assert result["difference"]["mean"] == pytest.approx(2.0)
    assert result["net_volume_m3"] == pytest.approx(2.0 * 1000 * 1000)
    assert any("vertical datum" in c for c in result["caveats"])
    reverse = await call("raster_difference", {"first": before, "second": after})
    assert any("coarser" in c for c in reverse["caveats"])


async def test_reproject_vector_and_raster(tmp_path: Path) -> None:
    habitat = layer(tmp_path / "h.gpkg", [(square(X0, Y0, 1000), {"n": 1})])
    moved = await call("reproject", {"source": habitat, "dst_crs": "EPSG:4326"})
    assert moved["source_crs"] == "EPSG:32610" and moved["bounds_wgs84"][
        0
    ] == pytest.approx(-120.11, abs=0.01)
    dem = raster(tmp_path / "dem.tif", np.full((10, 10), 5.0))
    grid = await call(
        "reproject", {"source": dem, "dst_crs": "EPSG:32611", "resampling": "bilinear"}
    )
    assert grid["target_crs"] == "EPSG:32611" and Path(grid["output_path"]).exists()


async def test_bad_inputs_are_typed_failures(tmp_path: Path) -> None:
    habitat = layer(tmp_path / "h.gpkg", [(square(X0, Y0, 10), {"n": 1})])
    with pytest.raises(ToolError, match="operation must be one of"):
        await call("overlay_features", {"operation": "merge", "source": habitat})
    with pytest.raises(ToolError, match="needs a second layer"):
        await call("overlay_features", {"operation": "intersection", "source": habitat})
    with pytest.raises(ToolError, match="No such"):
        await call(
            "clip_features",
            {"source": str(tmp_path / "missing.geojson"), "bbox": [0, 0, 1, 1]},
        )


async def test_tools_have_compact_titles() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert len(tools) == 5
    assert all(t.title and len(t.title) <= 24 and "(" not in t.title for t in tools)


async def test_outputs_carry_provenance_with_input_digests(tmp_path: Path) -> None:
    before = raster(tmp_path / "before.tif", np.full((10, 10), 1.0))
    after = raster(tmp_path / "after.tif", np.full((10, 10), 3.0))
    result = await call("raster_difference", {"first": after, "second": before})
    (entry,) = result["outputs"]
    record = json.loads(Path(entry["provenance"]).read_text())
    assert (
        record["schema"] == "clio.provenance/1"
        and record["tool"] == "raster_difference"
    )
    assert [s["name"] for s in record["sources"]] == ["after.tif", "before.tif"]
    assert all(s["version"].startswith("sha256:") for s in record["sources"])
    assert any("vertical datum" in note for note in record["notes"])
