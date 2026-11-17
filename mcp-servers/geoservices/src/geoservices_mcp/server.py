#!/usr/bin/env python3
"""Geoservices MCP server: reach geospatial data where it is served.

Five tools turn a service or remote file into data on disk, bounded by a region:
describe a service, fetch vector features (ArcGIS REST, OGC WFS), read a raster
window (GeoTIFF/COG, VRT, ArcGrid, IMG over HTTP; OGC WCS subsets), search STAC,
and read a point-cloud subset (Entwine/EPT, COPC). Every result says how complete
it is; a limit that was hit is reported with the reason, never hidden.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any

from dotenv import load_dotenv
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from pydantic import Field

from .implementation import arcgis, ogc, pointcloud, raster, stac
from .implementation.common import (
    GeoServiceError,
    Region,
    classify,
    output_path,
    slug,
    unwrap_viewer,
)
from .implementation.describe import describe_any, features_summary
from .implementation.provenance import Source, output_entry, write_provenance

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)
load_dotenv()

mcp: FastMCP = FastMCP(
    "geoservices",
    version="1.0.0",
    instructions=(
        "Fetch geospatial data from where it is served, bounded by a region given as "
        "[west, south, east, north] in degrees. Start with describe_geo_service to learn "
        "what a URL is (ArcGIS layer list, WFS feature types, WCS coverages, raster "
        "header, point-cloud extent) and whether it covers the region. Then use "
        "fetch_features for vector layers, read_raster_window for rasters and WCS, "
        "read_point_cloud for Entwine/EPT and COPC lidar, and search_stac for STAC. "
        "Declared extents can be wrong: fetch_features with count_only answers "
        "'does this layer have data here?' from the data itself."
    ),
)

READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}
WRITES = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}
Bbox = Annotated[
    list[float] | str | None,
    Field(description="Region [west, south, east, north] in WGS84 degrees."),
]


def _fail(name: str, exc: Exception) -> ToolError:
    if isinstance(exc, GeoServiceError):
        return ToolError(str(exc))
    logger.exception("%s failed", name)
    return ToolError(f"{name} failed: {type(exc).__name__}: {exc}")


@mcp.tool(
    name="describe_geo_service",
    title="Describe Service",
    description=(
        "Identify a geospatial URL and describe it: ArcGIS FeatureServer/MapServer/"
        "ImageServer layers, fields, record count and real data extent; OGC WFS feature "
        "types; WCS coverages (filter by words) or one coverage's grid; a remote raster's "
        "CRS, size, resolution and units; an Entwine/EPT or COPC point cloud's extent and "
        "point count; a STAC catalog. With bbox, reports whether the declared or measured "
        "extent overlaps it. Entwine viewer links are followed to their point cloud."
    ),
    annotations=READ_ONLY,
    tags={"geospatial", "arcgis", "ogc", "stac", "lidar", "coverage"},
)
async def describe_geo_service_tool(
    url: Annotated[
        str, Field(description="Service, layer, coverage, raster or point-cloud URL.")
    ],
    item: Annotated[
        str | None,
        Field(
            description="Optional WFS type name or WCS coverage id to describe in detail."
        ),
    ] = None,
    match: Annotated[
        str,
        Field(
            description="Words that WCS coverage ids must contain, to filter long lists."
        ),
    ] = "",
    bbox: Bbox = None,
) -> dict[str, Any]:
    """Describe what a geospatial URL serves and whether it covers a region."""
    try:
        return describe_any(url, item=item, match=match, region=Region.parse(bbox))
    except Exception as exc:
        raise _fail("describe_geo_service", exc) from exc


@mcp.tool(
    name="fetch_features",
    title="Fetch Features",
    description=(
        "Fetch vector features from an ArcGIS FeatureServer/MapServer layer (bbox and SQL "
        "where clause) or an OGC WFS feature type (bbox), paging through the service's "
        "record limit, and save them as a GeoJSON FeatureCollection in EPSG:4326. "
        "count_only returns just the number of matching features, which is the reliable "
        "way to test coverage when a layer's declared extent is wrong."
    ),
    annotations=WRITES,
    tags={"geospatial", "arcgis", "wfs", "features", "geojson"},
)
async def fetch_features_tool(
    url: Annotated[
        str, Field(description="ArcGIS layer URL (…/FeatureServer/0) or WFS URL.")
    ],
    bbox: Bbox = None,
    where: Annotated[str, Field(description="ArcGIS SQL where clause.")] = "1=1",
    type_name: Annotated[
        str | None, Field(description="WFS feature type name.")
    ] = None,
    out_fields: Annotated[
        str, Field(description="ArcGIS fields, comma-separated, or *.")
    ] = "*",
    max_features: Annotated[
        int, Field(description="Feature limit (1 to 50000).")
    ] = 5000,
    count_only: Annotated[
        bool, Field(description="Only count matching features.")
    ] = False,
    output_path: Annotated[
        str | None,
        Field(description="Output file name inside the artifacts directory."),
    ] = None,
) -> dict[str, Any]:
    """Fetch features by region and save GeoJSON, or count them."""
    try:
        region = Region.parse(bbox)
        kind = classify(url)
        max_features = max(1, min(int(max_features), 50_000))
        if count_only:
            if kind != "arcgis":
                raise GeoServiceError("count_only is supported for ArcGIS layers")
            n = arcgis.count(url, region, where)
            return {
                "url": url,
                "region": region.as_list() if region else None,
                "where": where,
                "count": n,
                "summary": f"{n} features match"
                + (" inside the region" if region else ""),
            }
        if kind == "arcgis":
            features, paging = arcgis.query(
                url, region, where, out_fields, max_features
            )
            name = url.rstrip("/").rsplit("/", 2)[-2]
        elif kind in {"wfs", "ows"}:
            features, paging = ogc.wfs_query(url, type_name or "", region, max_features)
            name = type_name or "wfs"
        else:
            raise GeoServiceError(
                f"fetch_features reads ArcGIS layers and WFS; this URL looks like {kind!r}. "
                "Use describe_geo_service to see what it is."
            )
        path = _out(output_path, f"{slug(name)}_features", ".geojson")
        path.write_text(json.dumps({"type": "FeatureCollection", "features": features}))
        sidecar = write_provenance(
            path,
            tool="fetch_features",
            sources=[
                Source(
                    name=name,
                    kind="service",
                    reference=url,
                    subset={
                        "bbox": region.as_list() if region else None,
                        "where": where,
                        "type_name": type_name,
                        "out_fields": out_fields,
                    },
                )
            ],
            method={"protocol": kind, "max_features": max_features, **paging},
            packages=["httpx"],
            notes=[]
            if paging.get("complete")
            else [f"stopped at {max_features} features"],
        )
        return {
            "url": url,
            "region": region.as_list() if region else None,
            "where": where,
            "output_path": str(path),
            "outputs": [
                output_entry(path, "features", "application/geo+json", sidecar)
            ],
            **features_summary(features),
            "paging": paging,
            "summary": f"{len(features)} features saved"
            + ("" if paging.get("complete") else f" (limit {max_features} reached)"),
        }
    except Exception as exc:
        raise _fail("fetch_features", exc) from exc


@mcp.tool(
    name="read_raster_window",
    title="Read Raster Window",
    description=(
        "Read only the part of one or more rasters inside a region and save it as one "
        "GeoTIFF: remote or local GeoTIFF/COG, VRT mosaics, ArcGrid (.adf) and IMG tiles "
        "through ranged HTTP reads, or an OGC WCS coverage subset (give coverage_id). "
        "Tiles in different CRSs are warped to one. The output is decimated to max_cells "
        "and the decimation is reported. Returns CRS, shape, resolution, units, valid "
        "fraction and statistics."
    ),
    annotations=WRITES,
    tags={"geospatial", "raster", "geotiff", "cog", "vrt", "wcs", "dem"},
)
async def read_raster_window_tool(
    sources: Annotated[
        list[str] | str,
        Field(
            description="Raster URL(s) or path(s), or a WCS service URL with coverage_id."
        ),
    ],
    bbox: Annotated[
        list[float] | str, Field(description="[west, south, east, north] degrees.")
    ],
    coverage_id: Annotated[str | None, Field(description="WCS coverage id.")] = None,
    max_cells: Annotated[
        int, Field(description="Cell budget for the output.")
    ] = 4_000_000,
    dst_crs: Annotated[
        str | None, Field(description="Output CRS, e.g. EPSG:32610.")
    ] = None,
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """Bounded raster read by region."""
    try:
        region = Region.parse(bbox)
        if region is None:
            raise GeoServiceError("bbox is required for a bounded raster read")
        items = [sources] if isinstance(sources, str) else list(sources)
        path = _out(
            output_path,
            f"{slug(coverage_id or items[0].rsplit('/', 1)[-1])}_window",
            ".tif",
        )
        if coverage_id or classify(items[0]) in {"wcs", "ows"}:
            if not coverage_id:
                raise GeoServiceError(
                    "Give coverage_id for a WCS (describe the service to list them)"
                )
            fetched = path.with_name(path.stem + "_wcs.tif")
            got = ogc.wcs_subset(items[0], coverage_id, region, fetched)
            try:
                result = raster.read_window(
                    [str(fetched)], region, path, max_cells=max_cells, dst_crs=dst_crs
                )
            finally:
                fetched.unlink(missing_ok=True)
            result["source"] = f"WCS {coverage_id}"
            result["wcs_bytes"] = got["bytes"]
        else:
            if classify(items[0]) == "arcgis":
                raise GeoServiceError(
                    "ArcGIS ImageServer export is not supported yet; use the layer's WCS or "
                    "a GeoTIFF download if the record offers one"
                )
            result = raster.read_window(
                items, region, path, max_cells=max_cells, dst_crs=dst_crs
            )
        sidecar = write_provenance(
            path,
            tool="read_raster_window",
            sources=[
                Source(
                    name=coverage_id or item.rsplit("/", 1)[-1],
                    kind="service" if coverage_id else "file",
                    reference=item,
                    subset={"bbox": region.as_list(), "coverage_id": coverage_id},
                )
                for item in items
            ],
            method={
                key: result.get(key)
                for key in (
                    "crs",
                    "resolution",
                    "native_resolution",
                    "decimation",
                    "nodata",
                )
            }
            | {"max_cells": max_cells},
            packages=["rasterio", "numpy"],
            notes=[
                f"unreadable source {s['source']}: {s['reason']}"
                for s in result.get("sources_unreadable", [])
            ]
            + ([] if result.get("units") else ["the source declares no units"]),
        )
        result["outputs"] = [output_entry(path, "raster", "image/tiff", sidecar)]
        stats = result.get("statistics") or {}
        result["summary"] = (
            f"{result['shape'][0]}x{result['shape'][1]} cells at {result['resolution']} "
            f"({'decimated x' + str(result['decimation']) if result['decimation'] > 1 else 'native resolution'})"
            + (
                f", values {stats['min']:.4g} to {stats['max']:.4g}"
                if stats
                else "; no valid cells: the sources' bounds overlap the region but hold "
                "no data there (outside the surveyed footprint or masked)"
            )
        )
        return result
    except Exception as exc:
        raise _fail("read_raster_window", exc) from exc


@mcp.tool(
    name="read_point_cloud",
    title="Read Point Cloud",
    description=(
        "Read the lidar points inside a region from an Entwine Point Tiles (ept.json) "
        "dataset, such as USGS 3DEP's public bucket (Entwine viewer links are accepted), "
        "or a COPC .copc.laz file, and save them as LAZ. Reads the finest octree level "
        "that fits max_points and reports the level, point spacing and whether the result "
        "is full density. Optional classification filter (2 = ground)."
    ),
    annotations=WRITES,
    tags={"geospatial", "lidar", "point-cloud", "ept", "copc", "3dep"},
)
async def read_point_cloud_tool(
    url: Annotated[
        str, Field(description="ept.json, Entwine viewer link, or .copc.laz URL.")
    ],
    bbox: Annotated[
        list[float] | str, Field(description="[west, south, east, north] degrees.")
    ],
    max_points: Annotated[
        int, Field(description="Point budget (1000 to 20 million).")
    ] = 2_000_000,
    classifications: Annotated[
        list[int] | None, Field(description="Keep only these ASPRS classes, e.g. [2].")
    ] = None,
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """Point-cloud subset by region."""
    try:
        region = Region.parse(bbox)
        if region is None:
            raise GeoServiceError("bbox is required for a point-cloud read")
        target = unwrap_viewer(url)
        kind = classify(target)
        path = _out(
            output_path, f"{slug(target.rstrip('/').rsplit('/', 2)[-2])}_points", ".laz"
        )
        if kind == "ept":
            result = pointcloud.ept_subset(
                target,
                region,
                path,
                max_points=max_points,
                classifications=classifications,
            )
        elif kind == "copc":
            result = pointcloud.copc_subset(
                target,
                region,
                path,
                max_points=max_points,
                classifications=classifications,
            )
        else:
            raise GeoServiceError(
                f"read_point_cloud reads EPT (ept.json) and COPC; this URL looks like {kind!r}"
            )
        sidecar = write_provenance(
            path,
            tool="read_point_cloud",
            sources=[
                Source(
                    name=target.rstrip("/").rsplit("/", 2)[-2],
                    kind="service",
                    reference=target,
                    subset={
                        "bbox": region.as_list(),
                        "classifications": classifications,
                    },
                )
            ],
            method={
                key: result.get(key)
                for key in ("format", "depth", "spacing", "complete", "nodes_read")
            }
            | {"max_points": max_points},
            packages=["laspy", "lazrs", "pyproj"],
            notes=[result["limit_reason"]] if result.get("limit_reason") else [],
        )
        result["outputs"] = [
            output_entry(path, "points", "application/vnd.laszip", sidecar)
        ]
        result["summary"] = f"{result['point_count']:,} points saved" + (
            "" if result["complete"] else f"; {result['limit_reason']}"
        )
        return result
    except Exception as exc:
        raise _fail("read_point_cloud", exc) from exc


@mcp.tool(
    name="search_stac",
    title="Search STAC",
    description=(
        "Find STAC items by region, time interval (e.g. 2020-01-01/2021-12-31) and "
        "collection, through a STAC API search or a bounded walk of a static catalog. "
        "Saves the items as GeoJSON and returns each item's id, time, bbox and asset links."
    ),
    annotations=WRITES,
    tags={"geospatial", "stac", "catalog", "search"},
)
async def search_stac_tool(
    url: Annotated[
        str, Field(description="STAC API root, /search, catalog or collection URL.")
    ],
    bbox: Bbox = None,
    datetime: Annotated[
        str | None, Field(description="Interval start/end; '..' is open.")
    ] = None,
    collections: Annotated[
        list[str] | None, Field(description="Collection ids.")
    ] = None,
    limit: Annotated[int, Field(description="Item limit (1 to 1000).")] = 100,
    output_path: Annotated[str | None, Field(description="Output file name.")] = None,
) -> dict[str, Any]:
    """STAC item search."""
    try:
        region = Region.parse(bbox)
        items, how = stac.search(url, region, datetime, collections, limit)
        path = _out(output_path, "stac_items", ".geojson")
        path.write_text(json.dumps({"type": "FeatureCollection", "features": items}))
        sidecar = write_provenance(
            path,
            tool="search_stac",
            sources=[
                Source(
                    name="STAC",
                    kind="service",
                    reference=url,
                    subset={
                        "bbox": region.as_list() if region else None,
                        "datetime": datetime,
                        "collections": collections,
                    },
                )
            ],
            method={"limit": limit, **how},
            packages=["httpx"],
        )
        return {
            "url": url,
            "region": region.as_list() if region else None,
            "datetime": datetime,
            "output_path": str(path),
            "outputs": [output_entry(path, "items", "application/geo+json", sidecar)],
            "item_count": len(items),
            "items": [stac._summary(i) for i in items[:50]],
            **how,
            "summary": f"{len(items)} STAC items",
        }
    except Exception as exc:
        raise _fail("search_stac", exc) from exc


def _out(name: str | None, default: str, suffix: str):
    return output_path(name, default, suffix)


@mcp.resource("geoservices://capabilities")
def capabilities() -> dict[str, Any]:
    """Protocols, formats and limits."""
    return {
        "tools": [
            "describe_geo_service",
            "fetch_features",
            "read_raster_window",
            "read_point_cloud",
            "search_stac",
        ],
        "protocols": [
            "ArcGIS REST FeatureServer/MapServer (query, count, extent)",
            "OGC WFS 2.0 (GeoJSON output)",
            "OGC WCS 2.0.1 (GetCoverage subset)",
            "STAC API item search and static catalogs",
            "GDAL /vsicurl/ ranged reads: GeoTIFF/COG, VRT, ArcGrid, IMG",
            "Entwine Point Tiles (LAZ nodes)",
            "COPC",
        ],
        "limits": {
            "features": 50_000,
            "raster_sources": raster.MAX_SOURCES,
            "raster_cells": 50_000_000,
            "points": 20_000_000,
            "ept_nodes": pointcloud.MAX_NODES,
        },
        "outputs": "files under CLIO_KIT_ARTIFACTS (GeoJSON, GeoTIFF, LAZ)",
    }


@mcp.prompt()
def fetch_region(url: str, bbox: str) -> list[Message]:
    """Guided workflow: describe a service, check coverage, fetch the region."""
    return [
        Message(
            f"Describe {url} with describe_geo_service and bbox {bbox}. If it covers the "
            "region, fetch the data with the matching tool (fetch_features, "
            "read_raster_window, read_point_cloud or search_stac) and report what was "
            "fetched, its completeness and any limits reached."
        )
    ]


def main() -> None:
    """Entry point for the geoservices MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
