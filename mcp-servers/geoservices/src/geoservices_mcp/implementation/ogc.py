"""OGC WFS (features) and WCS (coverage subsets): describe and bounded requests."""

from __future__ import annotations

import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import httpx

from .common import (
    MAX_JSON_BYTES,
    GeoServiceError,
    Region,
    client,
    get_json,
    get_text,
    http_error,
    raise_for_status,
    strip_query,
)

MAX_LIST = 100


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _find_all(node: ET.Element, name: str) -> list[ET.Element]:
    return [child for child in node.iter() if _local(child.tag) == name]


def _text(node: ET.Element, name: str) -> str:
    for child in node.iter():
        if _local(child.tag).casefold() == name.casefold() and child.text:
            return child.text.strip()
    return ""


def _base(url: str) -> tuple[str, dict[str, str]]:
    """Service base URL and any vendor parameters to keep (e.g. GeoServer workspaces)."""
    query = {k: v[0] for k, v in parse_qs(urlsplit(url).query).items()}
    keep = {
        k: v
        for k, v in query.items()
        if k.lower() not in {"service", "request", "version"}
    }
    return strip_query(url), keep


def _corners(node: ET.Element) -> list[float] | None:
    lower, upper = _text(node, "LowerCorner"), _text(node, "UpperCorner")
    if not lower or not upper:
        return None
    a, b = [float(v) for v in lower.split()[:2]], [float(v) for v in upper.split()[:2]]
    return [a[0], a[1], b[0], b[1]]


# --- WFS ------------------------------------------------------------------------------


def wfs_describe(url: str, type_name: str | None = None) -> dict[str, Any]:
    """List feature types with their WGS84 boxes (from GetCapabilities)."""
    base, extra = _base(url)
    text = get_text(
        base,
        {**extra, "service": "WFS", "request": "GetCapabilities", "version": "2.0.0"},
        what="WFS GetCapabilities",
    )
    root = _parse(text, "WFS GetCapabilities")
    types = []
    for node in _find_all(root, "FeatureType"):
        name = _text(node, "Name")
        if type_name and name != type_name:
            continue
        box = None
        for child in node:
            if _local(child.tag) == "WGS84BoundingBox":
                box = _corners(child)
        types.append(
            {
                "name": name,
                "title": _text(node, "Title"),
                "wgs84_extent": box,
                "default_crs": _text(node, "DefaultCRS") or _text(node, "DefaultSRS"),
            }
        )
    if type_name and not types:
        raise GeoServiceError(f"The WFS has no feature type named {type_name!r}")
    return {
        "kind": "wfs",
        "url": base,
        "feature_types": types[:MAX_LIST],
        "feature_type_count": len(types),
    }


def wfs_query(
    url: str,
    type_name: str,
    region: Region | None,
    max_features: int,
    page_size: int = 1000,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """GetFeature as GeoJSON in EPSG:4326, paged with startIndex/count."""
    if not type_name:
        raise GeoServiceError(
            "Give type_name (describe the WFS to list its feature types)"
        )
    base, extra = _base(url)
    features: list[dict[str, Any]] = []
    pages = 0
    page = max(1, min(page_size, max_features))
    while len(features) < max_features:
        params = {
            **extra,
            "service": "WFS",
            "version": "2.0.0",
            "request": "GetFeature",
            "typeNames": type_name,
            "outputFormat": "application/json",
            "srsName": "EPSG:4326",
            "count": str(min(page, max_features - len(features))),
            "startIndex": str(len(features)),
        }
        if region:
            # WFS 2.0 with EPSG:4326 uses lat/lon axis order; the urn form makes it explicit.
            params["bbox"] = (
                f"{region.south},{region.west},{region.north},{region.east},"
                "urn:ogc:def:crs:EPSG::4326"
            )
        data = get_json(base, params, what="WFS GetFeature")
        batch = data.get("features", []) if isinstance(data, dict) else []
        pages += 1
        features.extend(batch)
        if len(batch) < page or pages >= 1000:
            break
    for feature in features:
        _swap_if_latlon(feature, region)
    return features[:max_features], {
        "pages": pages,
        "page_size": page,
        "format": "GeoJSON",
        "complete": len(features) < max_features,
    }


def _swap_if_latlon(feature: dict[str, Any], region: Region | None) -> None:
    """Some servers return EPSG:4326 GeoJSON in lat/lon order. Detect it against the request box."""
    geometry = feature.get("geometry") or {}
    if region is None or geometry.get("type") != "Point":
        return
    x, y = geometry["coordinates"][:2]
    inside = region.west <= x <= region.east and region.south <= y <= region.north
    swapped = region.west <= y <= region.east and region.south <= x <= region.north
    if swapped and not inside:
        geometry["coordinates"] = [y, x, *geometry["coordinates"][2:]]
        feature.setdefault("properties", {})["_axis_order_corrected"] = True


# --- WCS ------------------------------------------------------------------------------


def wcs_describe(
    url: str, coverage_id: str | None = None, pattern: str = ""
) -> dict[str, Any]:
    """List coverages (optionally filtered) or describe one coverage's grid and envelope."""
    base, extra = _base(url)
    if coverage_id:
        text = get_text(
            base,
            {
                **extra,
                "service": "WCS",
                "version": "2.0.1",
                "request": "DescribeCoverage",
                "coverageId": coverage_id,
            },
            what="WCS DescribeCoverage",
        )
        root = _parse(text, "WCS DescribeCoverage")
        envelope = next(iter(_find_all(root, "Envelope")), None)
        if envelope is None:
            raise GeoServiceError(f"WCS returned no envelope for {coverage_id!r}")
        native = _corners(envelope)
        crs = _epsg(envelope.get("srsName", ""))
        result: dict[str, Any] = {
            "kind": "wcs",
            "url": base,
            "coverage_id": coverage_id,
            "crs": crs,
            "native_extent": native,
            "axis_labels": envelope.get("axisLabels", "").split(),
        }
        limits = next(iter(_find_all(root, "GridEnvelope")), None)
        if limits is not None:
            high = [int(v) for v in _text(limits, "high").split()]
            result["grid_size"] = [v + 1 for v in high]
        if native and crs:
            result["wgs84_extent"] = _to_wgs84(native, crs, result["axis_labels"])
        return result
    text = get_text(
        base,
        {**extra, "service": "WCS", "version": "2.0.1", "request": "GetCapabilities"},
        what="WCS GetCapabilities",
    )
    root = _parse(text, "WCS GetCapabilities")
    ids = [_text(node, "CoverageId") for node in _find_all(root, "CoverageSummary")]
    words = [w.casefold() for w in re.findall(r"\w+", pattern)]
    matching = [i for i in ids if all(w in i.casefold() for w in words)]
    return {
        "kind": "wcs",
        "url": base,
        "coverage_count": len(ids),
        "coverages": matching[:MAX_LIST],
        "matching": len(matching),
    }


def _epsg(srs_name: str) -> str | None:
    match = re.search(r"EPSG/0/(\d+)|EPSG::?(\d+)", srs_name)
    return f"EPSG:{match.group(1) or match.group(2)}" if match else None


def _lat_first(crs: str, labels: list[str]) -> bool:
    if labels:
        return (
            labels[0].lower() in {"lat", "latitude", "y", "n", "northing"}
            and crs == "EPSG:4326"
        )
    return crs == "EPSG:4326"


def _to_wgs84(native: list[float], crs: str, labels: list[str]) -> list[float]:
    from pyproj import Transformer

    if _lat_first(crs, labels):
        native = [native[1], native[0], native[3], native[2]]
    if crs == "EPSG:4326":
        return native
    box = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform_bounds(
        *native, densify_pts=21
    )
    return [round(v, 6) for v in box]


def wcs_subset(url: str, coverage_id: str, region: Region, out: Path) -> dict[str, Any]:
    """GetCoverage clipped to the region (in the coverage's native CRS), saved as GeoTIFF."""
    from pyproj import Transformer

    described = wcs_describe(url, coverage_id)
    crs, labels = described.get("crs"), described.get("axis_labels") or ["X", "Y"]
    if not crs:
        raise GeoServiceError(
            f"Coverage {coverage_id!r} declares no CRS; cannot subset by region"
        )
    if crs == "EPSG:4326":
        x0, y0, x1, y1 = region.as_list()
    else:
        x0, y0, x1, y1 = Transformer.from_crs(
            "EPSG:4326", crs, always_xy=True
        ).transform_bounds(*region.as_list(), densify_pts=21)
    xl, yl = (
        (labels[1], labels[0]) if _lat_first(crs, labels) else (labels[0], labels[1])
    )
    base, extra = _base(url)
    params = [
        *extra.items(),
        ("service", "WCS"),
        ("version", "2.0.1"),
        ("request", "GetCoverage"),
        ("coverageId", coverage_id),
        ("format", "image/geotiff"),
        ("subset", f"{xl}({x0},{x1})"),
        ("subset", f"{yl}({y0},{y1})"),
    ]
    try:
        with client() as http, http.stream("GET", base, params=params) as response:
            raise_for_status(response)
            kind = response.headers.get("content-type", "")
            size = 0
            with out.open("wb") as handle:
                for chunk in response.iter_bytes(65536):
                    size += len(chunk)
                    if size > 8 * MAX_JSON_BYTES:
                        raise GeoServiceError(
                            "The coverage subset exceeds 512 MiB; use a smaller region"
                        )
                    handle.write(chunk)
    except httpx.HTTPError as exc:
        out.unlink(missing_ok=True)
        raise http_error(exc, "WCS GetCoverage") from exc
    if "xml" in kind or "html" in kind:
        message = out.read_text(errors="replace")[:400]
        out.unlink(missing_ok=True)
        raise GeoServiceError(
            f"WCS refused the subset: {re.sub(r'<[^>]+>', ' ', message).strip()}"
        )
    return {"described": described, "bytes": size}


def _parse(text: str, what: str) -> ET.Element:
    try:
        root = ET.fromstring(text)
    except ET.ParseError as exc:
        raise GeoServiceError(f"{what} did not return XML: {text[:120]!r}") from exc
    if _local(root.tag) in {"ExceptionReport", "ServiceExceptionReport"}:
        raise GeoServiceError(
            f"{what} was refused: {_text(root, 'ExceptionText') or text[:300]}"
        )
    return root
