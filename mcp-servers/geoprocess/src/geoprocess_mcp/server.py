#!/usr/bin/env python3
"""Geoprocess MCP server: the core geospatial operations between acquisition and science.

Five tools: reproject a vector or raster, clip features, dissolve and overlay
polygons with areas in hectares, zonal statistics of a raster over polygons (or
values at points), and raster differencing. Areas are computed in an equal-area
CRS and named; co-registration and datum caveats are reported, never assumed.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Annotated, Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from pydantic import Field

from .implementation import raster_ops, vector
from .implementation.vector import GeoProcessError, output_path

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

RASTER_SUFFIXES = (".tif", ".tiff", ".vrt", ".img", ".asc", ".nc")

mcp: FastMCP = FastMCP(
    "geoprocess",
    version="1.0.0",
    instructions=(
        "Core geospatial operations on local files. reproject changes a vector's or "
        "raster's CRS. clip_features cuts features to a polygon or bbox. "
        "overlay_features dissolves overlapping polygons (so totals are covered area, "
        "not a sum of layers) or intersects/unions two layers, with hectares. "
        "zonal_statistics summarises a raster within polygons, or samples it at points "
        "(nearest cell or a buffer). raster_difference subtracts two rasters on the "
        "first one's grid and reports datum and unit caveats."
    ),
)

ANN = {"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True}


def _fail(name: str, exc: Exception) -> ToolError:
    if isinstance(exc, GeoProcessError):
        return ToolError(str(exc))
    logger.exception("%s failed", name)
    return ToolError(f"{name} failed: {type(exc).__name__}: {exc}")


def _stem(path: str) -> str:
    return Path(path).stem[:40] or "result"


@mcp.tool(
    name="reproject",
    title="Reproject",
    description=(
        "Reproject a vector file (GeoJSON, shapefile, GeoPackage, zipped shapefile) or a "
        "raster (GeoTIFF, VRT, IMG) to another CRS, e.g. EPSG:32610 or EPSG:4326. Rasters "
        "take an optional resolution and resampling (nearest for categories, bilinear "
        "or average for continuous values). Reports source and target CRS and the "
        "raster's vertical datum when it declares one."
    ),
    annotations=ANN,
    tags={"geospatial", "crs", "reprojection"},
)
async def reproject_tool(
    source: Annotated[str, Field(description="Path to the vector or raster file.")],
    dst_crs: Annotated[str, Field(description="Target CRS, e.g. EPSG:32610.")],
    source_crs: Annotated[
        str | None, Field(description="CRS for a vector file that declares none.")
    ] = None,
    resolution: Annotated[
        float | None, Field(description="Raster cell size in target units.")
    ] = None,
    resampling: Annotated[
        str, Field(description="Raster resampling method.")
    ] = "nearest",
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """Reproject a vector or raster."""
    try:
        if source.lower().endswith(RASTER_SUFFIXES):
            out = _out(output_path, f"{_stem(source)}_reprojected", ".tif")
            result = raster_ops.reproject_raster(
                source, dst_crs, out, resolution, resampling
            )
        else:
            out = _out(output_path, f"{_stem(source)}_reprojected", ".geojson")
            result = vector.reproject_vector(source, dst_crs, out, source_crs)
        result["summary"] = (
            f"Reprojected {result['source_crs']} to {result['target_crs']}"
        )
        return result
    except Exception as exc:
        raise _fail("reproject", exc) from exc


@mcp.tool(
    name="clip_features",
    title="Clip Features",
    description=(
        "Clip vector features to a polygon layer (mask) or a bbox [west, south, east, "
        "north] in degrees, save GeoJSON, and report how many features remain and, for "
        "polygons, the covered area in hectares."
    ),
    annotations=ANN,
    tags={"geospatial", "clip", "vector"},
)
async def clip_features_tool(
    source: Annotated[str, Field(description="Vector file to clip.")],
    mask: Annotated[str | None, Field(description="Polygon file to clip to.")] = None,
    bbox: Annotated[
        list[float] | None, Field(description="[west, south, east, north] degrees.")
    ] = None,
    source_crs: Annotated[
        str | None, Field(description="CRS if the file declares none.")
    ] = None,
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """Clip features."""
    try:
        out = _out(output_path, f"{_stem(source)}_clipped", ".geojson")
        result = vector.clip(source, mask, bbox, out, source_crs)
        result["summary"] = (
            f"{result['feature_count']} of {result['input_features']} features inside "
            f"{result['clipped_by']}"
        )
        return result
    except Exception as exc:
        raise _fail("clip_features", exc) from exc


@mcp.tool(
    name="overlay_features",
    title="Overlay Features",
    description=(
        "Polygon overlay with honest areas. operation=dissolve merges overlapping "
        "polygons (optionally per attribute 'by'), so the total is the area covered, "
        "and reports how much overlap was removed. intersection, union, difference, "
        "symmetric_difference and identity combine 'source' with 'other'. Areas are in "
        "hectares computed in an equal-area CRS."
    ),
    annotations=ANN,
    tags={"geospatial", "overlay", "dissolve", "area"},
)
async def overlay_features_tool(
    operation: Annotated[
        str,
        Field(
            description="dissolve, intersection, union, difference, symmetric_difference or identity."
        ),
    ],
    source: Annotated[str, Field(description="Polygon file.")],
    other: Annotated[
        str | None, Field(description="Second polygon file (not for dissolve).")
    ] = None,
    by: Annotated[
        str | None, Field(description="Attribute to dissolve or total areas by.")
    ] = None,
    source_crs: Annotated[
        str | None, Field(description="CRS if the file declares none.")
    ] = None,
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """Dissolve or overlay polygons."""
    try:
        out = _out(output_path, f"{_stem(source)}_{operation}", ".geojson")
        result = vector.overlay(operation, source, other, by, out, source_crs)
        area = result["area"]
        result["summary"] = (
            f"{operation}: {result['feature_count']} features covering "
            f"{area['covered_area_ha']:,} ha"
            + (
                f" ({result['overlap_removed_ha']:,} ha of overlap removed)"
                if result.get("overlap_removed_ha")
                else ""
            )
        )
        return result
    except Exception as exc:
        raise _fail("overlay_features", exc) from exc


@mcp.tool(
    name="zonal_statistics",
    title="Zonal Statistics",
    description=(
        "Summarise a raster inside polygons (count, min, max, mean, std, sum and the "
        "share of each zone with data), optionally grouped by an attribute, or sample it "
        "at point features (nearest cell, or the mean within buffer_m metres). Zones are "
        "rasterized on the raster's own grid."
    ),
    annotations={**ANN, "readOnlyHint": True},
    tags={"geospatial", "zonal-statistics", "raster", "sampling"},
)
async def zonal_statistics_tool(
    raster: Annotated[str, Field(description="Raster file.")],
    zones: Annotated[str, Field(description="Polygon or point file.")],
    by: Annotated[
        str | None, Field(description="Polygon attribute to group zones by.")
    ] = None,
    buffer_m: Annotated[
        float, Field(description="Point buffer radius in metres (0 = nearest cell).")
    ] = 0.0,
    id_field: Annotated[
        str | None, Field(description="Point attribute naming each point.")
    ] = None,
    band: Annotated[int, Field(description="Raster band.")] = 1,
    all_touched: Annotated[
        bool, Field(description="Count every cell a polygon touches.")
    ] = False,
) -> dict[str, Any]:
    """Zonal statistics or point sampling."""
    try:
        frame = vector.load(zones)
        if len(frame) and frame.geom_type.eq("Point").all():
            result = raster_ops.sample_points(raster, zones, buffer_m, band, id_field)
            result["summary"] = (
                f"Sampled {result['count']} points ({result['method']}); "
                f"{result['missing']} without data"
            )
        else:
            result = raster_ops.zonal_statistics(raster, zones, by, band, all_touched)
            empty = sum(1 for r in result["rows"] if not r["cells"])
            result["summary"] = f"Statistics for {result['zone_count']} zones" + (
                f"; {empty} have no raster data" if empty else ""
            )
        return result
    except Exception as exc:
        raise _fail("zonal_statistics", exc) from exc


@mcp.tool(
    name="raster_difference",
    title="Raster Difference",
    description=(
        "Subtract one raster from another (first minus second) on the first raster's "
        "grid, e.g. two DEMs for elevation change or snow depth, save the difference "
        "GeoTIFF and report its statistics, overlap, net volume for projected grids, and "
        "caveats when vertical datums, units or resolutions differ."
    ),
    annotations=ANN,
    tags={"geospatial", "raster", "change-detection", "dem"},
)
async def raster_difference_tool(
    first: Annotated[
        str, Field(description="Raster to subtract from (defines the grid).")
    ],
    second: Annotated[str, Field(description="Raster to subtract.")],
    band: Annotated[int, Field(description="Band in both rasters.")] = 1,
    resampling: Annotated[
        str,
        Field(
            description="How second is resampled: bilinear, nearest, average, cubic."
        ),
    ] = "bilinear",
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """Difference two rasters."""
    try:
        out = _out(output_path, f"{_stem(first)}_minus_{_stem(second)}", ".tif")
        result = raster_ops.raster_difference(first, second, out, band, resampling)
        d = result["difference"]
        result["summary"] = (
            f"Difference over {result['overlap_fraction']:.0%} of the grid"
            + (
                f": mean {d['mean']:.4g}, range {d['min']:.4g} to {d['max']:.4g}"
                if d
                else "; no overlap"
            )
            + (f"; {len(result['caveats'])} caveat(s)" if result["caveats"] else "")
        )
        return result
    except Exception as exc:
        raise _fail("raster_difference", exc) from exc


def _out(name: str | None, default: str, suffix: str) -> Path:
    return output_path(name, default, suffix)


@mcp.resource("geoprocess://capabilities")
def capabilities() -> dict[str, Any]:
    """Operations, formats and conventions."""
    return {
        "tools": [
            "reproject",
            "clip_features",
            "overlay_features",
            "zonal_statistics",
            "raster_difference",
        ],
        "vector_formats": ["GeoJSON", "shapefile (.shp or .zip)", "GeoPackage"],
        "raster_formats": ["GeoTIFF", "VRT", "IMG", "ASCII grid"],
        "area_crs": vector.EQUAL_AREA,
        "outputs": "files under CLIO_KIT_ARTIFACTS",
    }


@mcp.prompt()
def habitat_area(habitat: str, boundary: str) -> list[Message]:
    """Guided workflow: covered habitat area inside a boundary."""
    return [
        Message(
            f"Clip {habitat} to {boundary} with clip_features, dissolve the result with "
            "overlay_features so overlapping polygons are counted once, and report the "
            "covered area in hectares with the CRS used."
        )
    ]


def main() -> None:
    """Entry point for the geoprocess MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
