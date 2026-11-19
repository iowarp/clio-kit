"""Vector operations: load, reproject, clip, dissolve and overlay, with honest areas.

Areas and lengths are always computed in an equal-area or local projected CRS and
reported with the CRS used, never in degrees. Overlapping polygons are dissolved
before area totals, so a total is the area covered, not a sum of layers.
"""

from __future__ import annotations

import json
import os
import re
import tempfile
from pathlib import Path
from typing import Any

EQUAL_AREA = "EPSG:6933"  # WGS 84 / NSIDC EASE-Grid 2.0 Global, an equal-area CRS


class GeoProcessError(ValueError):
    """An operation could not be completed; the message says why."""


def artifacts_root() -> Path:
    configured = os.environ.get("CLIO_KIT_ARTIFACTS", "").strip()
    root = (
        Path(configured).expanduser()
        if configured
        else Path(tempfile.gettempdir()) / "clio-kit-geoprocess"
    )
    root.mkdir(parents=True, exist_ok=True)
    return root.resolve()


def output_path(candidate: str | None, default: str, suffix: str) -> Path:
    root = artifacts_root()
    name = Path(candidate).name if candidate else default
    name = re.sub(r"[^\w.\-]+", "_", name).strip("._") or default
    if not name.lower().endswith(suffix):
        name += suffix
    return root / name


def load(source: str | dict[str, Any], crs: str | None = None) -> Any:
    """Read a vector file (GeoJSON, shapefile, GeoPackage, zipped shapefile) or a GeoJSON dict."""
    import geopandas as gpd

    if isinstance(source, dict):
        frame = gpd.GeoDataFrame.from_features(source.get("features", [source]))
        frame = frame.set_crs(crs or "EPSG:4326")
    else:
        path = Path(source).expanduser()
        if not path.exists():
            raise GeoProcessError(f"No such vector file: {source}")
        frame = gpd.read_file(
            f"zip://{path}" if path.suffix.lower() == ".zip" else path
        )
        if frame.crs is None:
            if crs is None:
                raise GeoProcessError(
                    f"{path.name} declares no CRS; pass source_crs (GeoJSON is EPSG:4326)"
                )
            frame = frame.set_crs(crs)
    frame = frame[~frame.geometry.is_empty & frame.geometry.notna()]
    invalid = int((~frame.geometry.is_valid).sum())
    if invalid:
        frame = frame.set_geometry(frame.geometry.make_valid())
    frame.attrs["repaired"] = invalid
    return frame


def local_metric_crs(frame: Any) -> str:
    """UTM zone of the data's centroid when it spans < 6 degrees, else equal-area."""
    w, s, e, n = frame.to_crs("EPSG:4326").total_bounds
    if e - w <= 6 and n - s <= 20:
        zone = int((((w + e) / 2) + 180) // 6) + 1
        return f"EPSG:{32600 + zone if (s + n) / 2 >= 0 else 32700 + zone}"
    return EQUAL_AREA


def area_table(frame: Any, by: str | None = None) -> dict[str, Any]:
    """Covered area in hectares (overlaps dissolved), overall and per group."""
    metric = frame.to_crs(EQUAL_AREA)
    total = metric.geometry.union_all().area if len(metric) else 0.0
    out: dict[str, Any] = {
        "area_crs": EQUAL_AREA,
        "covered_area_ha": round(total / 1e4, 3),
        "sum_of_feature_areas_ha": round(float(metric.geometry.area.sum()) / 1e4, 3),
    }
    if by:
        if by not in frame.columns:
            raise GeoProcessError(
                f"No attribute {by!r}; available: {list(frame.columns)[:30]}"
            )
        groups = metric.dissolve(by=by)
        out["by"] = by
        out["groups"] = {
            str(k): round(float(v) / 1e4, 3)
            for k, v in sorted(groups.geometry.area.items(), key=lambda kv: -kv[1])[
                :100
            ]
        }
    return out


def save(frame: Any, out: Path) -> dict[str, Any]:
    wgs = (
        frame.to_crs("EPSG:4326")
        if frame.crs and frame.crs.to_epsg() != 4326
        else frame
    )
    out.write_text(wgs.to_json(drop_id=True, default=str))
    bounds = [round(float(v), 6) for v in wgs.total_bounds] if len(wgs) else None
    return {
        "output_path": str(out),
        "feature_count": len(frame),
        "geometry_types": {
            str(k): int(v) for k, v in frame.geom_type.value_counts().items()
        },
        "bounds_wgs84": bounds,
        "columns": [c for c in frame.columns if c != "geometry"][:60],
    }


def reproject_vector(
    source: str, dst_crs: str, out: Path, source_crs: str | None
) -> dict[str, Any]:
    frame = load(source, source_crs)
    src = frame.crs.to_string()
    moved = frame.to_crs(dst_crs)
    # GeoJSON output is WGS84 by specification; other targets are written as GeoPackage.
    if dst_crs.upper() in {"EPSG:4326", "OGC:CRS84"}:
        info = save(moved, out.with_suffix(".geojson"))
    else:
        target = out.with_suffix(".gpkg")
        moved.to_file(target, driver="GPKG")
        info = {
            "output_path": str(target),
            "feature_count": len(moved),
            "bounds_native": [round(float(v), 3) for v in moved.total_bounds],
        }
    return {
        "source_crs": src,
        "target_crs": moved.crs.to_string(),
        "repaired_geometries": frame.attrs["repaired"],
        **info,
    }


def clip(
    source: str,
    mask: str | None,
    bbox: list[float] | None,
    out: Path,
    source_crs: str | None,
) -> dict[str, Any]:
    import geopandas as gpd
    from shapely.geometry import box

    frame = load(source, source_crs)
    if mask:
        shape = load(mask).to_crs(frame.crs).geometry.union_all()
        label = Path(mask).name
    elif bbox:
        shape = gpd.GeoSeries([box(*bbox)], crs="EPSG:4326").to_crs(frame.crs).iloc[0]
        label = f"bbox {bbox}"
    else:
        raise GeoProcessError("Give mask (a polygon file) or bbox")
    clipped = gpd.clip(frame, shape)
    return {
        "clipped_by": label,
        "input_features": len(frame),
        "repaired_geometries": frame.attrs["repaired"],
        **save(clipped, out),
        **(
            {"area": area_table(clipped)}
            if clipped.geom_type.str.contains("Polygon").any()
            else {}
        ),
    }


OVERLAYS = {
    "dissolve",
    "intersection",
    "union",
    "difference",
    "symmetric_difference",
    "identity",
}


def overlay(
    operation: str,
    source: str,
    other: str | None,
    by: str | None,
    out: Path,
    source_crs: str | None,
) -> dict[str, Any]:
    import geopandas as gpd

    if operation not in OVERLAYS:
        raise GeoProcessError(f"operation must be one of {sorted(OVERLAYS)}")
    frame = load(source, source_crs)
    metric = local_metric_crs(frame)
    if operation == "dissolve":
        if by and by not in frame.columns:
            raise GeoProcessError(
                f"No attribute {by!r}; available: {list(frame.columns)[:30]}"
            )
        result = (
            frame.to_crs(metric).dissolve(by=by, as_index=False)
            if by
            else gpd.GeoDataFrame(
                geometry=[frame.to_crs(metric).geometry.union_all()], crs=metric
            )
        )
        before = area_table(frame)
        return {
            "operation": "dissolve",
            "by": by,
            "input_features": len(frame),
            "overlap_removed_ha": round(
                before["sum_of_feature_areas_ha"] - before["covered_area_ha"], 3
            ),
            **save(result, out),
            "area": area_table(result, by),
        }
    if not other:
        raise GeoProcessError(f"{operation} needs a second layer (other)")
    second = load(other).to_crs(metric)
    result = gpd.overlay(
        frame.to_crs(metric), second, how=operation, keep_geom_type=True
    )
    return {
        "operation": operation,
        "input_features": [len(frame), len(second)],
        "processing_crs": metric,
        **save(result, out),
        "area": area_table(result, by),
    }


def describe_columns(frame: Any) -> list[str]:
    return [c for c in frame.columns if c != "geometry"]


__all__ = ["json"]
