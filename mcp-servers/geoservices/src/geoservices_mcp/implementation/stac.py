"""STAC: item search on a STAC API, or a bounded walk of a static catalog."""

from __future__ import annotations

from typing import Any
from urllib.parse import urljoin

import httpx

from .common import GeoServiceError, Region, client, get_json, http_error

MAX_DOCUMENTS = 200


def _overlaps(bbox: list[float] | None, region: Region | None) -> bool:
    if region is None or not bbox:
        return True
    w, s, e, n = (bbox[0], bbox[1], bbox[3], bbox[4]) if len(bbox) == 6 else bbox[:4]
    return w < region.east and region.west < e and s < region.north and region.south < n


def _in_time(item: dict[str, Any], interval: str | None) -> bool:
    if not interval:
        return True
    start, _, end = interval.partition("/")
    props = item.get("properties") or {}
    t0 = props.get("start_datetime") or props.get("datetime") or ""
    t1 = props.get("end_datetime") or props.get("datetime") or ""
    if not t0 and not t1:
        return True  # undated items are reported, not dropped; the summary counts them
    return (not end or end == ".." or t0[: len(end)] <= end) and (
        not start or start == ".." or t1[: len(start)] >= start
    )


def _summary(item: dict[str, Any]) -> dict[str, Any]:
    props = item.get("properties") or {}
    assets = {
        key: {"href": a.get("href"), "type": a.get("type"), "roles": a.get("roles")}
        for key, a in list((item.get("assets") or {}).items())[:12]
    }
    return {
        "id": item.get("id"),
        "collection": item.get("collection"),
        "datetime": props.get("datetime") or props.get("start_datetime"),
        "bbox": item.get("bbox"),
        "assets": assets,
    }


def search(
    url: str,
    region: Region | None,
    interval: str | None,
    collections: list[str] | None,
    limit: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Return matching STAC items (full GeoJSON) and how they were found."""
    limit = max(1, min(int(limit), 1000))
    root = get_json(url, what="STAC document")
    conforms = " ".join(root.get("conformsTo") or [])
    search_link = next(
        (link["href"] for link in root.get("links", []) if link.get("rel") == "search"),
        None,
    )
    if url.rstrip("/").endswith("/search") or "item-search" in conforms or search_link:
        target = (
            url
            if url.rstrip("/").endswith("/search")
            else urljoin(url, search_link or "search")
        )
        body: dict[str, Any] = {"limit": min(limit, 250)}
        if region:
            body["bbox"] = region.as_list()
        if interval:
            body["datetime"] = interval
        if collections:
            body["collections"] = collections
        items: list[dict[str, Any]] = []
        pages = 0
        with client() as http:
            while target and len(items) < limit and pages < 20:
                try:
                    response = http.post(target, json=body)
                    response.raise_for_status()
                    data = response.json()
                except httpx.HTTPError as exc:
                    raise http_error(exc, "STAC search") from exc
                items.extend(data.get("features", []))
                pages += 1
                nxt = next(
                    (l for l in data.get("links", []) if l.get("rel") == "next"), None
                )
                if not nxt:
                    break
                target, body = nxt["href"], nxt.get("body", body)
        return items[:limit], {
            "method": "STAC API search",
            "pages": pages,
            "documents_read": pages,
        }
    # Static catalog: breadth-first over child/item links, bounded.
    queue, seen, items, read = [url], set(), [], 0
    truncated = False
    while queue and len(items) < limit:
        if read >= MAX_DOCUMENTS:
            truncated = True
            break
        href = queue.pop(0)
        if href in seen:
            continue
        seen.add(href)
        doc = root if read == 0 else get_json(href, what="STAC document")
        read += 1
        if doc.get("type") == "Feature":
            if (
                _overlaps(doc.get("bbox"), region)
                and _in_time(doc, interval)
                and (not collections or doc.get("collection") in collections)
            ):
                items.append(doc)
            continue
        extent = ((doc.get("extent") or {}).get("spatial") or {}).get("bbox") or [None]
        if doc.get("type") == "Collection" and not _overlaps(extent[0], region):
            continue
        if doc.get("type") == "FeatureCollection":
            items.extend(
                f for f in doc.get("features", []) if _overlaps(f.get("bbox"), region)
            )
        for link in doc.get("links", []):
            if link.get("rel") in {"child", "item"} and link.get("href"):
                queue.append(urljoin(href, link["href"]))
    if not root.get("links") and root.get("type") not in {
        "Catalog",
        "Collection",
        "Feature",
        "FeatureCollection",
    }:
        raise GeoServiceError("This URL is not a STAC catalog, collection or API")
    return items[:limit], {
        "method": "static catalog walk",
        "documents_read": read,
        "complete": not truncated and not queue,
        "limit_reason": f"stopped after {MAX_DOCUMENTS} catalog documents"
        if truncated
        else None,
    }


__all__ = ["_summary", "search"]
