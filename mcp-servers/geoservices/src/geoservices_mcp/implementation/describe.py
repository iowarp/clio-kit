"""describe_geo_service dispatch and GeoJSON feature summaries."""

from __future__ import annotations

from collections import Counter
from typing import Any

from . import arcgis, ogc, pointcloud, raster
from .common import GeoServiceError, Region, classify, get_json, unwrap_viewer


def _coverage(extent: list[float] | None, region: Region | None) -> str | None:
    """overlaps / disjoint / undetermined, from an extent already in WGS84."""
    if region is None:
        return None
    if not extent:
        return "undetermined"
    w, s, e, n = extent
    if w < region.east and region.west < e and s < region.north and region.south < n:
        return "overlaps"
    return "disjoint"


def raster_header(url: str) -> dict[str, Any]:
    import rasterio
    from rasterio.warp import transform_bounds

    with rasterio.Env(**raster.GDAL_ENV), rasterio.open(raster.gdal_path(url)) as ds:
        box = (
            [
                round(v, 6)
                for v in transform_bounds(
                    ds.crs, "EPSG:4326", *ds.bounds, densify_pts=21
                )
            ]
            if ds.crs
            else None
        )
        return {
            "kind": "raster",
            "url": url,
            "driver": ds.driver,
            "crs": ds.crs.to_string() if ds.crs else None,
            "size": [ds.height, ds.width],
            "bands": ds.count,
            "dtype": ds.dtypes[0],
            "resolution": [abs(ds.res[0]), abs(ds.res[1])],
            "nodata": ds.nodata,
            "units": (ds.units or [None])[0],
            "overviews": len(ds.overviews(1)),
            "tiled": bool(ds.profile.get("tiled")),
            "wgs84_extent": box,
        }


def describe_any(
    url: str, *, item: str | None = None, match: str = "", region: Region | None = None
) -> dict[str, Any]:
    url = unwrap_viewer(url)
    kind = classify(url)
    if kind == "arcgis":
        result = arcgis.describe(url)
        extent = result.get("data_extent") or result.get("declared_extent")
    elif kind == "wfs":
        result = ogc.wfs_describe(url, item)
        extent = result["feature_types"][0]["wgs84_extent"] if item else None
    elif kind == "wcs":
        result = ogc.wcs_describe(url, item, match)
        extent = result.get("wgs84_extent")
    elif kind == "ows":
        # GeoServer's /ows answers both; a named item decides which one is meant.
        try:
            result = (
                ogc.wcs_describe(url, item, match) if item else ogc.wfs_describe(url)
            )
        except GeoServiceError:
            result = ogc.wfs_describe(url, item)
        extent = result.get("wgs84_extent")
    elif kind == "ept":
        result = pointcloud.ept_describe(url)
        extent = result["wgs84_extent"]
    elif kind in {"raster", "copc"}:
        result = raster_header(url) if kind == "raster" else _copc_header(url)
        extent = result.get("wgs84_extent")
    elif kind == "stac":
        doc = get_json(url, what="STAC document")
        spatial = ((doc.get("extent") or {}).get("spatial") or {}).get("bbox") or [None]
        result = {
            "kind": "stac",
            "url": url,
            "type": doc.get("type"),
            "id": doc.get("id"),
            "title": doc.get("title"),
            "search": any(link.get("rel") == "search" for link in doc.get("links", [])),
            "children": sum(
                link.get("rel") == "child" for link in doc.get("links", [])
            ),
            "wgs84_extent": spatial[0][:4] if spatial[0] else None,
        }
        extent = result["wgs84_extent"]
    else:
        raise GeoServiceError(
            f"Not a recognised geospatial service or raster: {url}. Supported: ArcGIS REST "
            "layers, OGC WFS/WCS, STAC, GeoTIFF/COG/VRT/ArcGrid/IMG, EPT and COPC."
        )
    coverage = _coverage(extent, region)
    if coverage:
        result["region"] = region.as_list() if region else None
        result["coverage"] = coverage
        if coverage == "undetermined":
            result["coverage_reason"] = result.get("extent_caveat") or (
                "no usable extent; describe one item, or count features in the region"
            )
    return result


def _copc_header(url: str) -> dict[str, Any]:
    import laspy
    from pyproj import Transformer

    with laspy.CopcReader.open(url) as reader:
        crs = reader.header.parse_crs()
        mins, maxs = reader.header.mins, reader.header.maxs
        box = (
            Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform_bounds(
                mins[0], mins[1], maxs[0], maxs[1], densify_pts=21
            )
            if crs
            else None
        )
        return {
            "kind": "copc",
            "url": url,
            "points": int(reader.header.point_count),
            "crs": crs.to_string() if crs else None,
            "wgs84_extent": [round(v, 6) for v in box] if box else None,
            "z_range": [float(mins[2]), float(maxs[2])],
        }


def features_summary(features: list[dict[str, Any]]) -> dict[str, Any]:
    """Counts, geometry types, property names, measured bounds and a short preview."""
    types = Counter((f.get("geometry") or {}).get("type") or "None" for f in features)
    xs: list[float] = []
    ys: list[float] = []

    def walk(coords: Any) -> None:
        if (
            isinstance(coords, (list, tuple))
            and coords
            and isinstance(coords[0], (int, float))
        ):
            xs.append(float(coords[0]))
            ys.append(float(coords[1]))
        elif isinstance(coords, (list, tuple)):
            for c in coords:
                walk(c)

    for feature in features:
        walk((feature.get("geometry") or {}).get("coordinates"))
    names: list[str] = []
    for feature in features[:50]:
        for key in feature.get("properties") or {}:
            if key not in names:
                names.append(key)
    return {
        "feature_count": len(features),
        "geometry_types": dict(types),
        "properties": names[:60],
        "bounds": [
            round(min(xs), 6),
            round(min(ys), 6),
            round(max(xs), 6),
            round(max(ys), 6),
        ]
        if xs
        else None,
        "preview": [
            {k: v for k, v in list((f.get("properties") or {}).items())[:12]}
            for f in features[:5]
        ],
    }
