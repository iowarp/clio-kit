"""ArcGIS REST: describe services and layers, and query features with paging."""

from __future__ import annotations

import math
from collections.abc import Iterable
from itertools import pairwise
from typing import Any

from .common import (
    ARCGIS_RE,
    GeoServiceError,
    Region,
    ResponseTooLarge,
    get_json,
    strip_query,
)

MAX_LIST = 100


def split_url(url: str) -> tuple[str, str | None]:
    """Return (service root, layer id or None) for a FeatureServer/MapServer/ImageServer URL."""
    base = strip_query(url).rstrip("/")
    if base.lower().endswith("/query"):
        base = base[: -len("/query")]
    match = ARCGIS_RE.search(base)
    if not match:
        raise GeoServiceError(
            f"Not an ArcGIS FeatureServer/MapServer/ImageServer URL: {url}"
        )
    layer = match.group(2)
    root = base if layer is None else base.rsplit("/", 1)[0]
    return root, layer


def _crs_of(spatial_reference: dict[str, Any] | None) -> str | None:
    if not spatial_reference:
        return None
    wkid = spatial_reference.get("latestWkid") or spatial_reference.get("wkid")
    if wkid:
        return f"EPSG:{3857 if int(wkid) == 102100 else int(wkid)}"
    return spatial_reference.get("wkt")


def extent_wgs84(
    extent: dict[str, Any] | None,
) -> tuple[list[float] | None, str | None]:
    """Convert a declared ArcGIS extent to WGS84, or say why it cannot be trusted."""
    if not extent or "xmin" not in extent:
        return None, "the service declares no extent"
    values = [float(extent[k]) for k in ("xmin", "ymin", "xmax", "ymax")]
    if any(math.isnan(v) or abs(v) > 1e15 for v in values):
        return None, "the declared extent is empty or not a number"
    crs = _crs_of(extent.get("spatialReference"))
    if crs is None:
        return None, "the declared extent has no spatial reference"
    if crs == "EPSG:4326":
        west, south, east, north = values
        if not (-180 <= west <= east <= 180 and -90 <= south <= north <= 90):
            return None, (
                f"the declared extent {values} is not valid longitude/latitude "
                "(mixed or projected units); ask the data instead (count or extent query)"
            )
        return [west, south, east, north], None
    from pyproj import Transformer

    try:
        transformer = Transformer.from_crs(crs, "EPSG:4326", always_xy=True)
        box = transformer.transform_bounds(
            values[0], values[1], values[2], values[3], densify_pts=21
        )
    except Exception as exc:  # noqa: BLE001 - pyproj raises several unrelated types
        return None, f"the declared extent could not be transformed from {crs}: {exc}"
    if any(not math.isfinite(v) for v in box):
        return None, f"the declared extent does not transform from {crs}"
    return [round(v, 6) for v in box], None


def describe(url: str) -> dict[str, Any]:
    """Describe a service root (its layers) or one layer (fields, extent, counts)."""
    root, layer = split_url(url)
    kind = ARCGIS_RE.search(root + ("/" + layer if layer else ""))
    service_type = kind.group(1) if kind else "FeatureServer"
    target = f"{root}/{layer}" if layer is not None else root
    info = get_json(target, {"f": "json"}, what=f"ArcGIS describe {target}")
    result: dict[str, Any] = {
        "kind": "arcgis",
        "service_type": service_type,
        "url": target,
    }
    if layer is None:
        layers = [
            {
                "id": item.get("id"),
                "name": item.get("name"),
                "type": item.get("type", "layer"),
            }
            for item in (info.get("layers") or []) + (info.get("tables") or [])
        ]
        declared, caveat = extent_wgs84(info.get("fullExtent") or info.get("extent"))
        result.update(
            name=info.get("serviceDescription")
            or info.get("mapName")
            or root.rsplit("/", 2)[-2],
            layers=layers[:MAX_LIST],
            layer_count=len(layers),
            declared_extent=declared,
            crs=_crs_of(info.get("spatialReference") or {}),
        )
        if caveat:
            result["extent_caveat"] = caveat
        if service_type == "ImageServer":
            result["raster"] = {
                "bands": info.get("bandCount"),
                "pixel_type": info.get("pixelType"),
                "cell_size": [info.get("pixelSizeX"), info.get("pixelSizeY")],
            }
        return result
    declared, caveat = extent_wgs84(info.get("extent"))
    fields = [
        {
            "name": f.get("name"),
            "type": str(f.get("type", "")).replace("esriFieldType", ""),
        }
        for f in info.get("fields") or []
    ]
    result.update(
        name=info.get("name"),
        layer_type=info.get("type"),
        geometry_type=str(info.get("geometryType") or "").replace("esriGeometry", "")
        or None,
        fields=fields[:MAX_LIST],
        field_count=len(fields),
        max_record_count=info.get("maxRecordCount"),
        declared_extent=declared,
        crs=_crs_of((info.get("extent") or {}).get("spatialReference")),
        time_extent=_time_extent(info.get("timeInfo")),
    )
    if caveat:
        result["extent_caveat"] = caveat
    if info.get("type") in {"Feature Layer", "Table"}:
        query = f"{target}/query"
        try:
            result["record_count"] = get_json(
                query,
                {"where": "1=1", "returnCountOnly": "true", "f": "json"},
                what="count",
            ).get("count")
            data_extent = get_json(
                query,
                {
                    "where": "1=1",
                    "returnExtentOnly": "true",
                    "outSR": "4326",
                    "f": "json",
                },
                what="extent query",
            ).get("extent")
            measured, measure_caveat = extent_wgs84(
                {**(data_extent or {}), "spatialReference": {"wkid": 4326}}
                if data_extent and "spatialReference" not in data_extent
                else data_extent
            )
            result["data_extent"] = measured
            if measure_caveat:
                result["data_extent_caveat"] = measure_caveat
        except GeoServiceError as exc:
            result["data_extent_caveat"] = (
                f"the layer did not answer a count/extent query: {exc}"
            )
    return result


def _time_extent(info: dict[str, Any] | None) -> list[str] | None:
    if not info or not info.get("timeExtent"):
        return None
    from datetime import UTC, datetime

    return [
        datetime.fromtimestamp(ms / 1000, tz=UTC).date().isoformat()
        if ms is not None
        else ""
        for ms in info["timeExtent"]
    ]


def _envelope(region: Region | None) -> dict[str, str]:
    if region is None:
        return {}
    return {
        "geometry": ",".join(str(v) for v in region.as_list()),
        "geometryType": "esriGeometryEnvelope",
        "inSR": "4326",
        "spatialRel": "esriSpatialRelIntersects",
    }


def count(url: str, region: Region | None, where: str) -> int:
    """How many features match: the reliable coverage test when extents are wrong."""
    root, layer = split_url(url)
    if layer is None:
        raise GeoServiceError(
            "Give a layer URL (…/FeatureServer/<id>) to count features"
        )
    data = get_json(
        f"{root}/{layer}/query",
        {"where": where, "returnCountOnly": "true", "f": "json", **_envelope(region)},
        what="ArcGIS count",
    )
    return int(data.get("count", 0))


def query(
    url: str,
    region: Region | None,
    where: str,
    out_fields: str,
    max_features: int,
    page_size: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Page through a layer query; returns GeoJSON features and a paging record."""
    root, layer = split_url(url)
    if layer is None:
        raise GeoServiceError(
            "This is a service root; describe it to list its layers, then query one layer URL"
        )
    target = f"{root}/{layer}/query"
    info = get_json(f"{root}/{layer}", {"f": "json"}, what="ArcGIS layer")
    server_max = int(info.get("maxRecordCount") or page_size)
    page = max(1, min(page_size, server_max, max_features))
    features: list[dict[str, Any]] = []
    pages, offset, exceeded, fmt, reductions = 0, 0, False, "geojson", 0
    while len(features) < max_features:
        params = {
            "where": where,
            "outFields": out_fields,
            "outSR": "4326",
            "returnGeometry": "true",
            "resultOffset": str(offset),
            "resultRecordCount": str(min(page, max_features - len(features))),
            "f": fmt,
            **_envelope(region),
        }
        try:
            data = get_json(target, params, what="ArcGIS query")
        except ResponseTooLarge:
            # Detailed polygons: fewer per page, never a silently smaller result.
            if page == 1:
                raise
            page = max(1, page // 2)
            reductions += 1
            continue
        except GeoServiceError:
            if fmt == "geojson" and pages == 0:
                fmt = "json"  # older MapServer layers: Esri JSON, converted below and recorded
                continue
            raise
        batch = (
            data.get("features", [])
            if fmt == "geojson"
            else list(esri_to_geojson(data.get("features", [])))
        )
        pages += 1
        features.extend(batch)
        exceeded = bool(
            data.get("exceededTransferLimit")
            or (data.get("properties") or {}).get("exceededTransferLimit")
        )
        if not batch or (len(batch) < page and not exceeded):
            break
        offset += len(batch)
        if pages >= 1000:
            break
    paging = {
        "pages": pages,
        "page_size": page,
        "server_max_record_count": server_max,
        "format": "GeoJSON" if fmt == "geojson" else "Esri JSON converted to GeoJSON",
        "complete": len(features) < max_features and not (exceeded and pages >= 1000),
        **(
            {"page_size_reduced": f"halved {reductions} times: pages exceeded 64 MiB"}
            if reductions
            else {}
        ),
    }
    return features[:max_features], paging


def _signed_area(ring: list[list[float]]) -> float:
    return sum((x2 - x1) * (y2 + y1) for (x1, y1, *_), (x2, y2, *_) in pairwise(ring))


def esri_to_geojson(features: Iterable[dict[str, Any]]) -> Iterable[dict[str, Any]]:
    """Convert Esri JSON features; clockwise rings are outer rings in Esri's convention."""
    for feature in features:
        geometry = feature.get("geometry") or {}
        shape: dict[str, Any] | None
        if "x" in geometry:
            shape = {"type": "Point", "coordinates": [geometry["x"], geometry["y"]]}
        elif "points" in geometry:
            shape = {"type": "MultiPoint", "coordinates": geometry["points"]}
        elif "paths" in geometry:
            shape = {"type": "MultiLineString", "coordinates": geometry["paths"]}
        elif "rings" in geometry:
            polygons: list[list[list[list[float]]]] = []
            for ring in geometry["rings"]:
                if _signed_area(ring) >= 0 or not polygons:
                    polygons.append([ring])
                else:
                    polygons[-1].append(ring)
            shape = {"type": "MultiPolygon", "coordinates": polygons}
        else:
            shape = None
        yield {
            "type": "Feature",
            "geometry": shape,
            "properties": feature.get("attributes", {}),
        }


__all__ = ["count", "describe", "esri_to_geojson", "extent_wgs84", "query", "split_url"]
