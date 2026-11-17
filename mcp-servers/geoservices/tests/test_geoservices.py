"""Behaviour of each protocol against local fixtures; no public service is contacted."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from geoservices_mcp.implementation import arcgis, common, ogc
from geoservices_mcp.implementation.common import GeoServiceError, Region
from geoservices_mcp.server import mcp

from .conftest import Site

TAHOE = [-120.30, 38.70, -119.80, 39.35]


async def call(name: str, args: dict[str, Any]) -> dict[str, Any]:
    async with Client(mcp) as client:
        return (await client.call_tool(name, args)).data


# --- common -----------------------------------------------------------------------------


def test_region_parses_lists_and_strings_and_refuses_inverted_boxes() -> None:
    assert Region.parse("-120.3, 38.7, -119.8, 39.35") == Region(
        -120.3, 38.7, -119.8, 39.35
    )
    assert Region.parse(None) is None
    with pytest.raises(GeoServiceError, match="west < east"):
        Region.parse([10, 0, 5, 1])
    with pytest.raises(GeoServiceError, match="four numbers"):
        Region.parse([1, 2, 3])


@pytest.mark.parametrize(
    ("url", "kind"),
    [
        (
            "https://maps.example.org/server/rest/services/Wildlife/MapServer/8",
            "arcgis",
        ),
        (
            "https://s.arcgis.com/x/arcgis/rest/services/Perims/FeatureServer/0/query?f=json",
            "arcgis",
        ),
        (
            "https://gs.example.org/geoserver/ows?service=WCS&request=GetCapabilities",
            "wcs",
        ),
        ("https://gs.example.org/geoserver/wfs", "wfs"),
        ("https://gs.example.org/geoserver/ows", "ows"),
        ("https://bucket.example/PROJ/ept.json", "ept"),
        ("https://x.example/a.copc.laz", "copc"),
        ("https://topo.example/raster/TAHOE/TAHOE_be.vrt", "raster"),
        ("https://topo.example/raster/NE/NE_be/w001001.adf", "raster"),
        ("https://earth-search.example/v1/search", "stac"),
        (
            'https://usgs.entwine.io/data/view.html?r="https://bucket.example/PROJ"',
            "ept",
        ),
    ],
)
def test_classify_names_the_protocol(url: str, kind: str) -> None:
    assert common.classify(url) == kind


def test_entwine_viewer_link_unwraps_to_ept_json() -> None:
    url = "https://usgs.entwine.io/data/view.html?r=%22https://s3.example/usgs-lidar-public/CA_X%22"
    assert (
        common.unwrap_viewer(url)
        == "https://s3.example/usgs-lidar-public/CA_X/ept.json"
    )


def test_outputs_stay_inside_the_artifacts_root(artifacts: Path) -> None:
    path = common.output_path("../../etc/passwd", "x", ".geojson")
    assert path.parent == artifacts.resolve() and path.name == "passwd.geojson"


# --- ArcGIS -----------------------------------------------------------------------------


def test_malformed_declared_extent_is_flagged_not_used() -> None:
    box, caveat = arcgis.extent_wgs84(
        {
            "xmin": -180,
            "ymin": -90,
            "xmax": 427603.1,
            "ymax": 4285332.4,
            "spatialReference": {"wkid": 4326},
        }
    )
    assert box is None and "not valid longitude/latitude" in caveat


def test_projected_extent_is_transformed() -> None:
    box, caveat = arcgis.extent_wgs84(
        {
            "xmin": 750000,
            "ymin": 4319000,
            "xmax": 751000,
            "ymax": 4320000,
            "spatialReference": {"wkid": 26910},
        }
    )
    assert caveat is None and box[0] == pytest.approx(-120.1138, abs=1e-3)


def test_esri_rings_become_polygons_with_holes() -> None:
    outer = [[0, 0], [0, 10], [10, 10], [10, 0], [0, 0]]  # clockwise: outer
    hole = [[2, 2], [4, 2], [4, 4], [2, 4], [2, 2]]  # counter-clockwise: hole
    (feature,) = arcgis.esri_to_geojson(
        [{"geometry": {"rings": [outer, hole]}, "attributes": {"a": 1}}]
    )
    assert feature["geometry"] == {
        "type": "MultiPolygon",
        "coordinates": [[outer, hole]],
    }


def _layer(site: Site, features: int, server_max: int = 2) -> None:
    path = "/arcgis/rest/services/Fires/FeatureServer/0"
    site.json(
        path,
        {
            "name": "Fires",
            "type": "Feature Layer",
            "geometryType": "esriGeometryPoint",
            "maxRecordCount": server_max,
            "fields": [{"name": "NAME", "type": "esriFieldTypeString"}],
            "extent": {
                "xmin": -180,
                "ymin": -90,
                "xmax": 427603,
                "ymax": 4285332,
                "spatialReference": {"wkid": 4326},
            },
        },
    )

    def query(q: dict[str, str]) -> tuple[int, str, bytes]:
        if q.get("returnCountOnly"):
            return 200, "application/json", json.dumps({"count": features}).encode()
        if q.get("returnExtentOnly"):
            return (
                200,
                "application/json",
                json.dumps(
                    {
                        "extent": {
                            "xmin": -120.2,
                            "ymin": 38.9,
                            "xmax": -120.0,
                            "ymax": 39.1,
                        }
                    }
                ).encode(),
            )
        start, size = int(q["resultOffset"]), int(q["resultRecordCount"])
        batch = [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [-120.1, 39.0 + i / 100]},
                "properties": {"NAME": f"fire {i}"},
            }
            for i in range(start, min(start + size, features))
        ]
        more = start + size < features
        return (
            200,
            "application/json",
            json.dumps(
                {
                    "type": "FeatureCollection",
                    "features": batch,
                    "properties": {"exceededTransferLimit": more},
                }
            ).encode(),
        )

    site.routes[path + "/query"] = query


async def test_features_page_past_the_server_limit(site: Site) -> None:
    _layer(site, features=5, server_max=2)
    url = site.base + "/arcgis/rest/services/Fires/FeatureServer/0"
    result = await call("fetch_features", {"url": url, "bbox": TAHOE})
    assert result["feature_count"] == 5 and result["paging"]["pages"] == 3
    assert result["paging"]["complete"] is True
    saved = json.loads(Path(result["output_path"]).read_text())
    assert [f["properties"]["NAME"] for f in saved["features"]] == [
        f"fire {i}" for i in range(5)
    ]
    limited = await call("fetch_features", {"url": url, "max_features": 3})
    assert limited["feature_count"] == 3 and limited["paging"]["complete"] is False
    assert "limit 3 reached" in limited["summary"]


async def test_describe_reports_measured_extent_and_coverage(site: Site) -> None:
    _layer(site, features=4)
    result = await call(
        "describe_geo_service",
        {
            "url": site.base + "/arcgis/rest/services/Fires/FeatureServer/0",
            "bbox": TAHOE,
        },
    )
    assert result["declared_extent"] is None and "not valid" in result["extent_caveat"]
    assert result["data_extent"] == [-120.2, 38.9, -120.0, 39.1]
    assert result["record_count"] == 4 and result["coverage"] == "overlaps"
    count = await call(
        "fetch_features",
        {
            "url": site.base + "/arcgis/rest/services/Fires/FeatureServer/0",
            "count_only": True,
        },
    )
    assert count["count"] == 4


async def test_service_error_envelope_is_a_typed_failure(site: Site) -> None:
    site.json(
        "/arcgis/rest/services/Bad/FeatureServer/0",
        {"error": {"code": 400, "message": "Invalid query"}},
    )
    with pytest.raises(ToolError, match="refused by the service: 400 Invalid query"):
        await call(
            "fetch_features",
            {"url": site.base + "/arcgis/rest/services/Bad/FeatureServer/0"},
        )


# --- OGC -------------------------------------------------------------------------------

DESCRIBE = """<wcs:CoverageDescriptions xmlns:wcs="http://www.opengis.net/wcs/2.0"
 xmlns:gml="http://www.opengis.net/gml/3.2"><wcs:CoverageDescription>
 <gml:boundedBy><gml:Envelope srsName="http://www.opengis.net/def/crs/EPSG/0/3310"
  axisLabels="X Y" srsDimension="2"><gml:lowerCorner>-189990 -326400</gml:lowerCorner>
  <gml:upperCorner>202890 442800</gml:upperCorner></gml:Envelope></gml:boundedBy>
 <gml:domainSet><gml:RectifiedGrid><gml:limits><gml:GridEnvelope><gml:low>0 0</gml:low>
 <gml:high>13095 25639</gml:high></gml:GridEnvelope></gml:limits></gml:RectifiedGrid></gml:domainSet>
 </wcs:CoverageDescription></wcs:CoverageDescriptions>"""


def test_wcs_description_reads_gml_envelope(site: Site) -> None:
    site.bytes("/geoserver/ows", DESCRIBE.encode(), "application/xml")
    result = ogc.wcs_describe(site.base + "/geoserver/ows", "fuel")
    assert result["crs"] == "EPSG:3310" and result["grid_size"] == [13096, 25640]
    assert result["wgs84_extent"][0] == pytest.approx(-122.288, abs=1e-2)


def test_wcs_exception_report_is_refused(site: Site) -> None:
    site.bytes(
        "/geoserver/ows",
        b'<ows:ExceptionReport xmlns:ows="http://www.opengis.net/ows/2.0"><ows:Exception>'
        b"<ows:ExceptionText>No such coverage</ows:ExceptionText></ows:Exception></ows:ExceptionReport>",
        "application/xml",
    )
    with pytest.raises(GeoServiceError, match="No such coverage"):
        ogc.wcs_describe(site.base + "/geoserver/ows", "missing")


# --- rasters ---------------------------------------------------------------------------


def _tile(
    path: Path, crs: str, left: float, top: float, value: float, size: int = 200
) -> None:
    import rasterio
    from rasterio.transform import from_origin

    data = np.full((size, size), value, dtype="float32")
    data[:, -20:] = -9999.0  # an unsurveyed strip, as on real tile edges
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=size,
        width=size,
        count=1,
        dtype="float32",
        crs=crs,
        transform=from_origin(left, top, 5, 5),
        nodata=-9999.0,
    ) as out:
        out.write(data, 1)


async def test_raster_window_mosaics_two_utm_zones_and_reports_decimation(
    tmp_path: Path,
) -> None:
    west, east = tmp_path / "zone10.tif", tmp_path / "zone11.tif"
    _tile(west, "EPSG:26910", 760000, 4300000, 2000.0)  # just west of -120
    _tile(east, "EPSG:26911", 238000, 4300000, 2500.0)  # just east of -120
    result = await call(
        "read_raster_window",
        {
            "sources": [str(west), str(east)],
            "bbox": [-120.02, 38.80, -119.97, 38.82],
            "max_cells": 20000,
        },
    )
    assert result["crs"] == "EPSG:26910" and result["sources_used"] == 2
    assert result["decimation"] > 1
    assert result["statistics"]["min"] == pytest.approx(2000.0, abs=1)
    assert result["statistics"]["max"] == pytest.approx(2500.0, abs=1)
    assert 0 < result["valid_fraction"] < 1


async def test_raster_window_outside_the_data_is_a_typed_failure(
    tmp_path: Path,
) -> None:
    tile = tmp_path / "t.tif"
    _tile(tile, "EPSG:26910", 760000, 4300000, 1.0)
    with pytest.raises(ToolError, match="does not cover it"):
        await call(
            "read_raster_window", {"sources": [str(tile)], "bbox": [-100, 30, -99, 31]}
        )


async def test_remote_cog_is_read_by_range_requests(site: Site, tmp_path: Path) -> None:
    tile = tmp_path / "remote.tif"
    _tile(tile, "EPSG:26910", 760000, 4300000, 1234.0)
    site.bytes("/dem/remote.tif", tile.read_bytes(), "image/tiff")
    result = await call(
        "read_raster_window",
        {
            "sources": site.base + "/dem/remote.tif",
            "bbox": [-120.02, 38.80, -120.005, 38.81],
        },
    )
    assert result["statistics"]["mean"] == pytest.approx(1234.0)


# --- point clouds ----------------------------------------------------------------------


def _ept(site: Site, tmp_path: Path) -> None:
    import laspy

    root = [-13376500.0, 4746000.0, 1800.0, -13374500.0, 4748000.0, 3800.0]

    def node(name: str, n: int, seed: int) -> None:
        rng = np.random.default_rng(seed)
        header = laspy.LasHeader(point_format=6, version="1.4")
        header.scales, header.offsets = [0.01] * 3, root[:3]
        las = laspy.LasData(header)
        las.x = rng.uniform(root[0], root[3], n)
        las.y = rng.uniform(root[1], root[4], n)
        las.z = rng.uniform(1900, 2100, n)
        las.classification = np.where(np.arange(n) % 2 == 0, 2, 1).astype(np.uint8)
        path = tmp_path / f"{name}.laz"
        las.write(path)
        site.bytes(f"/lidar/P/ept-data/{name}.laz", path.read_bytes())

    site.json(
        "/lidar/P/ept.json",
        {
            "bounds": root,
            "boundsConforming": root,
            "dataType": "laszip",
            "hierarchyType": "json",
            "points": 5000,
            "span": 128,
            "srs": {"authority": "EPSG", "horizontal": "3857"},
            "schema": [{"name": "X"}, {"name": "Y"}, {"name": "Z"}],
        },
    )
    hierarchy = {
        "0-0-0-0": 1000,
        **{f"1-{x}-{y}-0": 1000 for x in (0, 1) for y in (0, 1)},
    }
    site.json("/lidar/P/ept-hierarchy/0-0-0-0.json", hierarchy)
    node("0-0-0-0", 1000, 1)
    for i, (x, y) in enumerate([(0, 0), (0, 1), (1, 0), (1, 1)]):
        node(f"1-{x}-{y}-0", 1000, 10 + i)


async def test_ept_subset_reads_full_depth_when_it_fits(
    site: Site, tmp_path: Path
) -> None:
    _ept(site, tmp_path)
    result = await call(
        "read_point_cloud",
        {
            "url": site.base + "/lidar/P/ept.json",
            "bbox": [-120.16, 39.17, -120.145, 39.18],
            "max_points": 100000,
        },
    )
    assert result["complete"] is True and result["depth"] == 1
    assert 0 < result["point_count"] < 5000
    assert set(result["classes"]) <= {"1", "2"}


async def test_ept_subset_stops_at_the_budget_and_says_so(
    site: Site, tmp_path: Path
) -> None:
    _ept(site, tmp_path)
    result = await call(
        "read_point_cloud",
        {
            "url": site.base + "/lidar/P/ept.json",
            "bbox": [-120.17, 39.16, -120.14, 39.19],
            "max_points": 1000,
            "classifications": [2],
        },
    )
    assert result["complete"] is False and result["depth"] == 0
    assert "next level would exceed" in result["limit_reason"]
    assert set(result["classes"]) == {"2"}
    assert "points saved; stopped at octree depth 0" in result["summary"]


async def test_tools_have_compact_titles() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert {t.name for t in tools} == {
        "describe_geo_service",
        "fetch_features",
        "read_raster_window",
        "read_point_cloud",
        "search_stac",
    }
    assert all(t.title and len(t.title) <= 24 and "(" not in t.title for t in tools)
