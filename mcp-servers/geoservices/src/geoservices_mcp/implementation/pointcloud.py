"""Point-cloud subsets from Entwine Point Tiles (EPT) and COPC, read by region.

EPT (USGS 3DEP's ``usgs-lidar-public`` bucket, the target of NDP's Entwine viewer
links) is an octree of LAZ nodes; each deeper level adds points. The reader walks
the octree over the region only, stops at the deepest level whose estimated point
count fits the budget, and says so: the result is complete at that level of detail,
not at full density, unless ``complete`` is true.
"""

from __future__ import annotations

import concurrent.futures
import io
from pathlib import Path
from typing import Any

import httpx
import numpy as np

from .common import GeoServiceError, Region, client, get_json, http_error, unwrap_viewer

DEFAULT_MAX_POINTS = 2_000_000
MAX_NODES = 400
MAX_DEPTH = 24
WORKERS = 8


def _node_bounds(root: list[float], key: tuple[int, int, int, int]) -> list[float]:
    depth, x, y, z = key
    size = [(root[i + 3] - root[i]) / (2**depth) for i in range(3)]
    return [
        root[0] + x * size[0],
        root[1] + y * size[1],
        root[2] + z * size[2],
        root[0] + (x + 1) * size[0],
        root[1] + (y + 1) * size[1],
        root[2] + (z + 1) * size[2],
    ]


def _overlap(node: list[float], want: list[float]) -> float:
    """Fraction of the node's horizontal area inside the wanted box (0 when disjoint)."""
    w = min(node[3], want[2]) - max(node[0], want[0])
    h = min(node[4], want[3]) - max(node[1], want[1])
    if w <= 0 or h <= 0:
        return 0.0
    return (w * h) / ((node[3] - node[0]) * (node[4] - node[1]))


def _ept_crs(meta: dict[str, Any]) -> str:
    srs = meta.get("srs") or {}
    if srs.get("authority") and srs.get("horizontal"):
        return f"{srs['authority']}:{srs['horizontal']}"
    if srs.get("wkt"):
        return str(srs["wkt"])
    raise GeoServiceError("The EPT dataset declares no spatial reference")


def ept_describe(url: str) -> dict[str, Any]:
    url = unwrap_viewer(url)
    meta = get_json(url, what="EPT metadata")
    crs = _ept_crs(meta)
    from pyproj import Transformer

    b = meta.get("boundsConforming") or meta["bounds"]
    box = Transformer.from_crs(crs, "EPSG:4326", always_xy=True).transform_bounds(
        b[0], b[1], b[3], b[4], densify_pts=21
    )
    return {
        "kind": "ept",
        "url": url,
        "points": meta.get("points"),
        "crs": crs if len(crs) < 40 else "WKT",
        "data_type": meta.get("dataType"),
        "wgs84_extent": [round(v, 6) for v in box],
        "z_range": [b[2], b[5]],
        "dimensions": [d.get("name") for d in meta.get("schema", [])][:40],
    }


def _hierarchy(base: str, key: str, http: httpx.Client) -> dict[str, int]:
    try:
        response = http.get(f"{base}/ept-hierarchy/{key}.json")
        response.raise_for_status()
        return {k: int(v) for k, v in response.json().items()}
    except httpx.HTTPError as exc:
        raise http_error(exc, f"EPT hierarchy {key}") from exc


def ept_subset(
    url: str,
    region: Region,
    out: Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    classifications: list[int] | None = None,
) -> dict[str, Any]:
    """Read the points inside ``region`` at the finest level that fits ``max_points``."""
    import laspy
    from pyproj import CRS, Transformer

    url = unwrap_viewer(url)
    meta = get_json(url, what="EPT metadata")
    if meta.get("dataType") != "laszip":
        raise GeoServiceError(
            f"EPT data type {meta.get('dataType')!r} is not supported; only LAZ (laszip) nodes"
        )
    crs = _ept_crs(meta)
    base = url.rsplit("/", 1)[0]
    root = [float(v) for v in meta["bounds"]]
    want = list(
        Transformer.from_crs("EPSG:4326", crs, always_xy=True).transform_bounds(
            region.west, region.south, region.east, region.north, densify_pts=21
        )
    )
    max_points = max(1_000, min(int(max_points), 20_000_000))
    with client() as http:
        hierarchy = _hierarchy(base, "0-0-0-0", http)
        level = [(0, 0, 0, 0)]
        chosen: list[str] = []
        estimate, depth, deeper = 0.0, -1, False
        while level and depth < MAX_DEPTH:
            entries = []
            for key in level:
                name = "-".join(map(str, key))
                count = hierarchy.get(name)
                if count == -1:
                    hierarchy.update(_hierarchy(base, name, http))
                    count = hierarchy.get(name)
                if not count:
                    continue
                share = _overlap(_node_bounds(root, key), want)
                if share > 0:
                    entries.append((key, name, count * share))
            if not entries:
                break
            level_estimate = sum(e[2] for e in entries)
            if chosen and (
                estimate + level_estimate > max_points
                or len(chosen) + len(entries) > MAX_NODES
            ):
                deeper = True
                break
            chosen.extend(e[1] for e in entries)
            estimate += level_estimate
            depth = entries[0][0][0]
            level = [
                (d + 1, 2 * x + dx, 2 * y + dy, 2 * z + dz)
                for (d, x, y, z), _, _ in entries
                for dx in (0, 1)
                for dy in (0, 1)
                for dz in (0, 1)
            ]
        if not chosen:
            raise GeoServiceError(
                f"The point cloud has no points in the region {region.as_list()} "
                f"(its extent is {ept_describe(url)['wgs84_extent']})"
            )

        def fetch(name: str) -> Any:
            response = http.get(f"{base}/ept-data/{name}.laz")
            response.raise_for_status()
            return laspy.read(io.BytesIO(response.content))

        parts, failed = [], []
        with concurrent.futures.ThreadPoolExecutor(WORKERS) as pool:
            futures = {pool.submit(fetch, name): name for name in chosen}
            for future in concurrent.futures.as_completed(futures):
                try:
                    parts.append(future.result())
                except (httpx.HTTPError, laspy.errors.LaspyException) as exc:
                    failed.append({"node": futures[future], "reason": str(exc)[:160]})
    return _write_points(
        parts,
        want,
        out,
        crs,
        classifications,
        extra={
            "source": url,
            "format": "EPT",
            "depth": depth,
            "nodes_read": len(parts),
            "nodes_failed": failed,
            "spacing": round(
                (root[3] - root[0]) / float(meta.get("span", 128)) / 2**depth, 3
            ),
            "complete": not deeper and not failed,
            "limit_reason": (
                f"stopped at octree depth {depth}: the next level would exceed "
                f"{max_points:,} points or {MAX_NODES} nodes"
                if deeper
                else None
            ),
            "crs_object": CRS.from_user_input(crs),
        },
    )


def copc_subset(
    url: str,
    region: Region,
    out: Path,
    *,
    max_points: int = DEFAULT_MAX_POINTS,
    classifications: list[int] | None = None,
) -> dict[str, Any]:
    """Query a COPC file by region, adding octree levels while the result fits."""
    import laspy
    from pyproj import Transformer

    try:
        reader = laspy.CopcReader.open(url)
    except Exception as exc:
        raise GeoServiceError(f"COPC file could not be opened: {exc}") from exc
    with reader:
        crs_obj = reader.header.parse_crs()
        if crs_obj is None:
            raise GeoServiceError("The COPC file declares no CRS")
        want = Transformer.from_crs(
            "EPSG:4326", crs_obj, always_xy=True
        ).transform_bounds(
            region.west, region.south, region.east, region.north, densify_pts=21
        )
        bounds = laspy.copc.Bounds(
            mins=np.array([want[0], want[1]]), maxs=np.array([want[2], want[3]])
        )
        points, level, deeper = None, -1, False
        for candidate in range(MAX_DEPTH):
            result = reader.query(bounds=bounds, level_range=range(candidate + 1))
            if points is not None and len(result) > max_points:
                deeper = True
                break
            points, level = result, candidate
            if len(result) == len(reader.query(bounds=bounds)):
                break
    return _write_points(
        [points],
        list(want),
        out,
        crs_obj.to_string(),
        classifications,
        extra={
            "source": url,
            "format": "COPC",
            "depth": level,
            "nodes_read": None,
            "nodes_failed": [],
            "spacing": None,
            "complete": not deeper,
            "limit_reason": f"stopped at level {level} to stay under {max_points:,} points"
            if deeper
            else None,
            "crs_object": crs_obj,
        },
    )


def _write_points(
    parts: list[Any],
    want: list[float],
    out: Path,
    crs: str,
    classifications: list[int] | None,
    extra: dict[str, Any],
) -> dict[str, Any]:
    import laspy

    xs, ys, zs, cls = [], [], [], []
    for las in parts:
        x, y, z = np.asarray(las.x), np.asarray(las.y), np.asarray(las.z)
        c = (
            np.asarray(las.classification)
            if "classification" in las.point_format.dimension_names
            else np.zeros(len(x), np.uint8)
        )
        keep = (x >= want[0]) & (x <= want[2]) & (y >= want[1]) & (y <= want[3])
        if classifications:
            keep &= np.isin(c, classifications)
        xs.append(x[keep])
        ys.append(y[keep])
        zs.append(z[keep])
        cls.append(c[keep])
    x = np.concatenate(xs) if xs else np.array([])
    if x.size == 0:
        raise GeoServiceError(
            "No points remain inside the region (after any class filter)"
        )
    y, z, c = np.concatenate(ys), np.concatenate(zs), np.concatenate(cls)
    header = laspy.LasHeader(point_format=6, version="1.4")
    header.offsets = [float(x.min()), float(y.min()), float(z.min())]
    header.scales = [0.01, 0.01, 0.01]
    crs_object = extra.pop("crs_object")
    try:
        header.add_crs(crs_object)
    except Exception:  # noqa: BLE001 - written without CRS; reported below, not silently
        extra["crs_written"] = False
    las = laspy.LasData(header)
    las.x, las.y, las.z, las.classification = x, y, z, c
    las.write(out)
    classes, counts = np.unique(c, return_counts=True)
    return {
        "output_path": str(out),
        "point_count": int(x.size),
        "crs": crs if len(crs) < 40 else crs_object.name,
        "bounds_native": [
            round(float(v), 2) for v in (x.min(), y.min(), x.max(), y.max())
        ],
        "z": {
            "min": round(float(z.min()), 2),
            "max": round(float(z.max()), 2),
            "mean": round(float(z.mean()), 2),
        },
        "classes": {str(int(k)): int(v) for k, v in zip(classes, counts, strict=True)},
        **extra,
    }
