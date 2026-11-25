"""Raster operations: reprojection, zonal statistics, point sampling and differencing.

Co-registration is explicit: zonal statistics rasterize polygons on the raster's own
grid; point sampling uses the nearest cell or a buffer mean; differencing resamples
the second raster onto the first raster's grid and reports the method. Vertical
datums are compared, not assumed: a difference between rasters whose vertical
reference differs or is unknown carries a caveat.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np

from .vector import GeoProcessError, load


def _open(path: str) -> Any:
    import rasterio

    local = Path(path).expanduser()
    if not local.exists():
        raise GeoProcessError(f"No such raster: {path}")
    return rasterio.open(local)


def _stats(values: np.ndarray) -> dict[str, float] | None:
    if values.size == 0:
        return None
    return {
        "min": round(float(values.min()), 6),
        "max": round(float(values.max()), 6),
        "mean": round(float(values.mean()), 6),
        "std": round(float(values.std()), 6),
        "sum": round(float(values.sum()), 6),
    }


def _valid(ds: Any, data: np.ndarray) -> np.ndarray:
    mask = np.ones(data.shape, bool)
    if ds.nodata is not None and not (
        isinstance(ds.nodata, float) and math.isnan(ds.nodata)
    ):
        mask &= data != ds.nodata
    if np.issubdtype(data.dtype, np.floating):
        mask &= np.isfinite(data)
    return mask


def _vertical(crs: Any) -> str | None:
    if crs is None:
        return None
    try:
        from pyproj import CRS

        parsed = CRS.from_user_input(crs.to_wkt())
        for sub in parsed.sub_crs_list or []:
            if sub.is_vertical:
                return sub.name
    except Exception:  # noqa: BLE001 - an unparseable CRS means an unknown vertical datum
        return None
    return None


def reproject_raster(
    source: str, dst_crs: str, out: Path, resolution: float | None, resampling: str
) -> dict[str, Any]:
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import calculate_default_transform, reproject

    method = getattr(Resampling, resampling, None)
    if method is None:
        raise GeoProcessError(
            "resampling must be nearest, bilinear, cubic, average, mode, min or max"
        )
    with _open(source) as src:
        transform, width, height = calculate_default_transform(
            src.crs,
            dst_crs,
            src.width,
            src.height,
            *src.bounds,
            resolution=resolution,
        )
        profile = {
            **src.profile,
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "compress": "deflate",
        }
        profile.pop("blockxsize", None), profile.pop("blockysize", None)
        with rasterio.open(out, "w", **profile) as dst:
            for band in range(1, src.count + 1):
                reproject(
                    rasterio.band(src, band),
                    rasterio.band(dst, band),
                    resampling=method,
                    src_nodata=src.nodata,
                    dst_nodata=src.nodata,
                )
        return {
            "output_path": str(out),
            "source_crs": src.crs.to_string(),
            "target_crs": str(dst_crs),
            "shape": [height, width],
            "resolution": [abs(transform.a), abs(transform.e)],
            "resampling": resampling,
            "source_vertical_datum": _vertical(src.crs),
        }


def zonal_statistics(
    raster: str,
    zones: str,
    by: str | None,
    band: int,
    all_touched: bool,
    max_zones: int = 2000,
) -> dict[str, Any]:
    """Per-polygon statistics on the raster's grid, with the share of each zone covered."""
    from rasterio.features import geometry_mask
    from rasterio.windows import from_bounds

    frame = load(zones)
    if len(frame) > max_zones:
        raise GeoProcessError(
            f"{len(frame)} zones exceed {max_zones}; dissolve or filter first"
        )
    if by and by not in frame.columns:
        raise GeoProcessError(
            f"No attribute {by!r}; available: {list(frame.columns)[:30]}"
        )
    rows: list[dict[str, Any]] = []
    with _open(raster) as ds:
        frame = frame.to_crs(ds.crs)
        cell_area = abs(ds.res[0] * ds.res[1])
        if by:
            frame = frame.dissolve(by=by, as_index=False)
        for index, zone in frame.iterrows():
            geom = zone.geometry
            label = str(zone[by]) if by else str(index)
            b = geom.bounds
            window = from_bounds(*b, ds.transform).round_offsets().round_lengths()
            window = (
                window.intersection(
                    from_bounds(*ds.bounds, ds.transform)
                    .round_offsets()
                    .round_lengths()
                )
                if _overlaps(b, ds.bounds)
                else None
            )
            if window is None or window.width < 1 or window.height < 1:
                rows.append(
                    {
                        "zone": label,
                        "cells": 0,
                        "coverage": 0.0,
                        "note": "zone lies outside the raster",
                    }
                )
                continue
            data = ds.read(band, window=window)
            inside = ~geometry_mask(
                [geom],
                out_shape=data.shape,
                transform=ds.window_transform(window),
                all_touched=all_touched,
            )
            valid = inside & _valid(ds, data)
            zone_cells = geom.area / cell_area if ds.crs.is_projected else None
            rows.append(
                {
                    "zone": label,
                    "cells": int(valid.sum()),
                    "coverage": round(float(valid.sum() / zone_cells), 4)
                    if zone_cells
                    else None,
                    **(_stats(data[valid].astype("float64")) or {}),
                }
            )
        units = (ds.units or [None])[band - 1]
    return {
        "raster": raster,
        "zones": zones,
        "by": by,
        "zone_count": len(rows),
        "units": units,
        "all_touched": all_touched,
        "rows": rows,
        "method": "polygons rasterized on the raster grid (cell centres"
        + (" or any touched cell)" if all_touched else ")"),
    }


def _overlaps(a: tuple[float, ...], b: Any) -> bool:
    return a[0] < b.right and b.left < a[2] and a[1] < b.top and b.bottom < a[3]


def sample_points(
    raster: str, points: str, buffer_m: float, band: int, id_field: str | None
) -> dict[str, Any]:
    """Value at each point: nearest cell, or mean within buffer_m metres."""
    frame = load(points)
    if not frame.geom_type.eq("Point").all():
        raise GeoProcessError("sample_points needs point features (stations, plots)")
    rows: list[dict[str, Any]] = []
    with _open(raster) as ds:
        local = frame.to_crs(ds.crs)
        for index, point in local.iterrows():
            label = (
                str(point[id_field])
                if id_field and id_field in local.columns
                else str(index)
            )
            x, y = point.geometry.x, point.geometry.y
            if buffer_m and ds.crs.is_projected:
                r_cells = max(1, math.ceil(buffer_m / abs(ds.res[0])))
                row, col = ds.index(x, y)
                from rasterio.windows import Window

                window = Window(
                    col - r_cells, row - r_cells, 2 * r_cells + 1, 2 * r_cells + 1
                )
                data = ds.read(
                    band, window=window, boundless=True, fill_value=ds.nodata or 0
                )
                yy, xx = np.mgrid[-r_cells : r_cells + 1, -r_cells : r_cells + 1]
                disk = (xx**2 + yy**2) * abs(ds.res[0]) ** 2 <= buffer_m**2
                values = data[disk & _valid(ds, data)].astype("float64")
                rows.append(
                    {
                        "point": label,
                        "cells": int(values.size),
                        "value": round(float(values.mean()), 6)
                        if values.size
                        else None,
                    }
                )
            else:
                value = next(ds.sample([(x, y)], indexes=band))[0]
                ok = bool(_valid(ds, np.array([value])).all())
                rows.append(
                    {
                        "point": label,
                        "value": float(value) if ok else None,
                        **({} if ok else {"note": "no data at this point"}),
                    }
                )
    return {
        "raster": raster,
        "points": points,
        "count": len(rows),
        "method": f"mean within {buffer_m} m" if buffer_m else "nearest cell",
        "rows": rows,
        "missing": sum(r["value"] is None for r in rows),
    }


def raster_difference(
    first: str, second: str, out: Path, band: int, resampling: str
) -> dict[str, Any]:
    """first minus second, on first's grid."""
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.warp import reproject

    caveats: list[str] = []
    with _open(first) as a, _open(second) as b:
        base = a.read(band).astype("float64")
        aligned = np.full(base.shape, np.nan)
        reproject(
            rasterio.band(b, band),
            aligned,
            src_nodata=b.nodata,
            dst_nodata=np.nan,
            dst_transform=a.transform,
            dst_crs=a.crs,
            resampling=getattr(Resampling, resampling),
        )
        valid = _valid(a, a.read(band)) & np.isfinite(aligned)
        diff = np.where(valid, base - aligned, np.nan).astype("float32")
        va, vb = _vertical(a.crs), _vertical(b.crs)
        if va != vb:
            caveats.append(
                f"vertical datums differ: first {va or 'unknown'}, second {vb or 'unknown'}"
            )
        elif va is None:
            caveats.append(
                "neither raster declares a vertical datum; confirm both use the same one"
            )
        ua, ub = (a.units or [None])[band - 1], (b.units or [None])[band - 1]
        if ua != ub:
            caveats.append(f"units differ or are undeclared: first {ua}, second {ub}")
        if abs(b.res[0]) > abs(a.res[0]) * 1.5:
            caveats.append(
                f"second raster is coarser ({abs(b.res[0]):.3g} vs {abs(a.res[0]):.3g}); "
                "differences include resampling error"
            )
        profile = {
            **a.profile,
            "dtype": "float32",
            "count": 1,
            "nodata": np.nan,
            "compress": "deflate",
        }
        with rasterio.open(out, "w", **profile) as dst:
            dst.write(diff, 1)
        cell_area = abs(a.res[0] * a.res[1]) if a.crs and a.crs.is_projected else None
        values = diff[valid].astype("float64")
        return {
            "output_path": str(out),
            "crs": a.crs.to_string(),
            "shape": list(diff.shape),
            "resampling": f"second resampled onto first's grid ({resampling})",
            "overlap_cells": int(valid.sum()),
            "overlap_fraction": round(float(valid.mean()), 4),
            "difference": _stats(values),
            "net_volume_m3": round(float(values.sum() * cell_area), 3)
            if cell_area
            else None,
            "caveats": caveats,
        }
