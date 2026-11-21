"""Bounded reads of remote or local rasters (GeoTIFF/COG, VRT, ArcGrid, IMG) through GDAL.

Only the blocks that intersect the region are fetched (HTTP range requests through
GDAL's /vsicurl/). Several tiles are mosaicked; tiles in different CRSs (3DEP DEMs
straddle UTM zones) are warped to one CRS first. The output is decimated to a cell
budget and the decimation is reported, never hidden.
"""

from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import numpy as np

from .common import GeoServiceError, Region

MAX_SOURCES = 64
#: Full-resolution bytes one read may fetch. Sources without overviews must be read at
#: native resolution even for a coarse result, so a whole-basin window of a 0.5 m mosaic
#: would fetch tens of gigabytes. Past this default a read stops with the numbers and
#: the way through (``max_read_bytes``): whether to fetch that much is the caller's
#: decision, not this server's.
MAX_READ_BYTES = int(
    os.environ.get("GEOSERVICES_MAX_READ_BYTES", str(1536 * 1024 * 1024))
)
DEFAULT_MAX_CELLS = 4_000_000
GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MAX_RETRY": "2",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
    "CPL_VSIL_CURL_USE_HEAD": "NO",
}


def gdal_path(source: str) -> str:
    """Local paths pass through; URLs read through /vsicurl/; ArcGrid opens its header."""
    if source.startswith(("http://", "https://")):
        path = source
        if path.lower().endswith(".adf"):
            path = path.rsplit("/", 1)[0] + "/hdr.adf"
        return "/vsicurl/" + path
    local = Path(source).expanduser()
    if not local.exists():
        raise GeoServiceError(f"No such file: {source}")
    if local.suffix.lower() == ".adf":
        local = local.parent / "hdr.adf"
    return str(local)


def read_window(
    sources: list[str],
    region: Region,
    out: Path,
    *,
    max_cells: int = DEFAULT_MAX_CELLS,
    dst_crs: str | None = None,
    band: int = 1,
    max_read_bytes: int | None = None,
) -> dict[str, Any]:
    """Mosaic the part of ``sources`` inside ``region`` into one GeoTIFF."""
    import rasterio
    from rasterio.transform import from_origin
    from rasterio.warp import transform_bounds

    if not sources:
        raise GeoServiceError("Give at least one raster source (URL or path)")
    unique = list(dict.fromkeys(gdal_path(s) for s in sources))
    if len(unique) > MAX_SOURCES:
        raise GeoServiceError(
            f"{len(unique)} sources exceed the {MAX_SOURCES}-tile limit for one read; "
            "select the tiles covering the region first, or use the dataset's VRT mosaic"
        )
    max_cells = max(10_000, min(int(max_cells), 50_000_000))
    opened: list[Any] = []
    skipped: list[dict[str, str]] = []
    with rasterio.Env(**GDAL_ENV):
        try:
            for path in unique:
                try:
                    opened.append(rasterio.open(path))
                except rasterio.errors.RasterioIOError as exc:
                    skipped.append(
                        {
                            "source": path.removeprefix("/vsicurl/"),
                            "reason": str(exc)[:200],
                        }
                    )
            if not opened:
                raise GeoServiceError(
                    "No source could be opened: "
                    + "; ".join(f"{s['source']}: {s['reason']}" for s in skipped[:3])
                )
            if any(ds.crs is None for ds in opened):
                raise GeoServiceError(
                    "A source declares no CRS; it cannot be placed on the map"
                )
            target = (
                rasterio.crs.CRS.from_user_input(dst_crs) if dst_crs else opened[0].crs
            )
            want = transform_bounds(
                "EPSG:4326", target, *region.as_list(), densify_pts=21
            )
            used, outside = [], 0
            for ds in opened:
                b = transform_bounds(ds.crs, target, *ds.bounds, densify_pts=21)
                if (
                    b[0] < want[2]
                    and want[0] < b[2]
                    and b[1] < want[3]
                    and want[1] < b[3]
                ):
                    used.append(ds)
                else:
                    outside += 1
            if not used:
                raise GeoServiceError(
                    f"None of the {len(opened)} opened sources overlaps the region "
                    f"{region.as_list()}; the data does not cover it"
                )
            first = used[0]
            if first.crs == target:
                native_res = min(abs(ds.res[0]) for ds in used if ds.crs == target)
            else:  # approximate the native cell size in the target CRS
                b = transform_bounds(first.crs, target, *first.bounds)
                native_res = (b[2] - b[0]) / first.width
            _check_read_budget(
                used,
                want,
                target,
                native_res,
                band,
                max_cells,
                max_read_bytes or MAX_READ_BYTES,
            )
            width = (want[2] - want[0]) / native_res
            height = (want[3] - want[1]) / native_res
            factor = max(1.0, math.sqrt(width * height / max_cells))
            res = native_res * factor
            cols = max(1, round((want[2] - want[0]) / res))
            rows = max(1, round((want[3] - want[1]) / res))
            transform = from_origin(want[0], want[3], res, res)
            source_dtype = np.dtype(first.dtypes[band - 1])
            floating = np.issubdtype(source_dtype, np.floating)
            dtype = np.float32 if floating or factor > 1 else source_dtype
            nodata = (
                float("nan")
                if np.issubdtype(dtype, np.floating)
                else (first.nodata if first.nodata is not None else 0)
            )
            mosaic = np.full((rows, cols), nodata, dtype=dtype)
            for ds in used:
                layer = np.full((rows, cols), nodata, dtype=dtype)
                if ds.crs == target and not ds.transform.b and not ds.transform.d:
                    _read_aligned(
                        ds, band, layer, transform, res, nodata, floating, factor
                    )
                else:
                    _warp(ds, band, layer, transform, target, nodata, floating, factor)
                fill = (
                    ~np.isnan(layer)
                    if np.issubdtype(dtype, np.floating)
                    else layer != nodata
                )
                empty = (
                    np.isnan(mosaic)
                    if np.issubdtype(dtype, np.floating)
                    else mosaic == nodata
                )
                mosaic[fill & empty] = layer[fill & empty]
            not_covered = _not_covered(used, target, transform, rows, cols, res)
            units = (first.units or [None])[band - 1]
            description = (first.descriptions or [None])[band - 1]
            data = (
                np.ma.masked_invalid(mosaic)
                if np.issubdtype(dtype, np.floating)
                else np.ma.masked_equal(mosaic, nodata)
            )
            profile = {
                "driver": "GTiff",
                "height": rows,
                "width": cols,
                "count": 1,
                "dtype": dtype,
                "crs": target,
                "transform": transform,
                "nodata": nodata,
                "compress": "deflate",
            }
            with rasterio.open(out, "w", **profile) as dst:
                dst.write(mosaic, 1)
                if units:
                    dst.update_tags(1, units=units)
        finally:
            for ds in opened:
                ds.close()
    valid = int(data.count())
    stats = (
        {
            "min": float(data.min()),
            "max": float(data.max()),
            "mean": round(float(data.mean()), 4),
            "std": round(float(data.std()), 4),
        }
        if valid
        else None
    )
    return {
        "output_path": str(out),
        "crs": target.to_string(),
        "shape": [rows, cols],
        "resolution": round(res, 6),
        "native_resolution": round(native_res, 6),
        "decimation": round(factor, 3),
        "bounds_native": [round(v, 3) for v in want],
        "sources_used": len(used),
        "sources_outside_region": outside,
        "sources_unreadable": skipped,
        "valid_fraction": round(valid / data.size, 4) if data.size else 0.0,
        "region_not_covered": not_covered,
        "nodata": None if isinstance(nodata, float) and math.isnan(nodata) else nodata,
        "units": units,
        "band_description": description,
        "statistics": stats,
    }


def _not_covered(used, target, transform, rows, cols, res) -> dict[str, Any] | None:
    """The part of the window that no source's footprint reaches, or ``None``.

    Footprints are the sources' own bounds, so this separates "no tile here" from
    nodata cells inside a tile. A missing neighbour tile shows up here instead of
    as a silently empty strip of the output.
    """
    from rasterio.warp import transform_bounds

    xs = transform.c + (np.arange(cols) + 0.5) * res
    ys = transform.f - (np.arange(rows) + 0.5) * res
    covered = np.zeros((rows, cols), dtype=bool)
    for ds in used:
        b = transform_bounds(ds.crs, target, *ds.bounds, densify_pts=21)
        inside_x = (xs >= b[0]) & (xs <= b[2])
        inside_y = (ys >= b[1]) & (ys <= b[3])
        covered |= inside_y[:, None] & inside_x[None, :]
    missing = ~covered
    if not missing.any():
        return None
    rows_idx = np.flatnonzero(missing.any(axis=1))
    cols_idx = np.flatnonzero(missing.any(axis=0))
    box = (
        xs[cols_idx[0]] - res / 2,
        ys[rows_idx[-1]] - res / 2,
        xs[cols_idx[-1]] + res / 2,
        ys[rows_idx[0]] + res / 2,
    )
    west, south, east, north = transform_bounds(target, "EPSG:4326", *box)
    return {
        "fraction": round(float(missing.mean()), 4),
        "bbox_wgs84": [round(v, 5) for v in (west, south, east, north)],
        "note": "No source covers this part of the region; its cells are nodata. "
        "Add the tiles that cover it, or report it as missing.",
    }


def _resampling(floating: bool, factor: float):
    from rasterio.enums import Resampling

    if not floating:
        return Resampling.nearest
    return Resampling.average if factor > 1.5 else Resampling.bilinear


def _read_aligned(ds, band, layer, transform, res, nodata, floating, factor) -> None:
    """Same CRS: read just the overlapping window, decimated by GDAL (uses overviews)."""
    from rasterio.windows import from_bounds

    rows, cols = layer.shape
    left, top = transform.c, transform.f
    right, bottom = left + cols * res, top - rows * res
    b = ds.bounds
    l, r = max(left, b.left), min(right, b.right)
    btm, tp = max(bottom, b.bottom), min(top, b.top)
    if l >= r or btm >= tp:
        return
    c0, c1 = round((l - left) / res), round((r - left) / res)
    r0, r1 = round((top - tp) / res), round((top - btm) / res)
    if c1 <= c0 or r1 <= r0:
        return
    window = from_bounds(l, btm, r, tp, ds.transform)
    data = ds.read(
        band,
        window=window,
        out_shape=(r1 - r0, c1 - c0),
        masked=True,
        resampling=_resampling(floating, factor),
    )
    # Convert before filling: an integer source decimated into a float output takes
    # NaN as nodata, which an integer array cannot hold.
    values = data.astype(layer.dtype, copy=False).filled(nodata)
    layer[r0:r1, c0:c1] = values


def _warp(ds, band, layer, transform, target, nodata, floating, factor) -> None:
    """Different CRS: GDAL warps only the source windows the destination needs."""
    import rasterio
    from rasterio.warp import reproject

    reproject(
        source=rasterio.band(ds, band),
        destination=layer,
        src_nodata=ds.nodata,
        dst_transform=transform,
        dst_crs=target,
        dst_nodata=nodata,
        resampling=_resampling(floating, factor),
    )


def _check_read_budget(
    used: list[Any],
    want: Any,
    target: Any,
    native_res: float,
    band: int,
    max_cells: int,
    budget: int,
) -> None:
    """Stop a read whose full-resolution fetch would exceed ``budget``, saying how to go on."""
    from rasterio.warp import transform_bounds

    width = (want[2] - want[0]) / native_res
    height = (want[3] - want[1]) / native_res
    total = 0.0
    smallest_factor = math.inf
    for ds in used:
        left, bottom, right, top = transform_bounds(
            target, ds.crs, *want, densify_pts=21
        )
        b = ds.bounds
        w = max(0.0, min(right, b.right) - max(left, b.left)) / abs(ds.res[0])
        h = max(0.0, min(top, b.top) - max(bottom, b.bottom)) / abs(ds.res[1])
        cells = w * h
        # GDAL reads the coarsest overview that is still at least as fine as the output.
        factor = max(1.0, math.sqrt(width * height / max_cells))
        usable = [o for o in ds.overviews(band) if o <= factor] or [1]
        read_factor = max(usable)
        smallest_factor = min(smallest_factor, read_factor)
        total += cells / (read_factor**2) * np.dtype(ds.dtypes[band - 1]).itemsize
    if total > budget:
        area_km2 = (
            (want[2] - want[0]) * (want[3] - want[1]) / 1e6
            if target.is_projected
            else None
        )
        fits = budget / max(total, 1) * (area_km2 or 0)
        needed = math.ceil(total / 2**20) * 2**20
        raise GeoServiceError(
            f"This window would fetch about {total / 1e9:.1f} GB of full-resolution data "
            f"(the source has {'no overviews' if smallest_factor == 1 else 'only coarse overviews'} "
            "at this scale, so every native block is read even for a coarse result), more "
            f"than this call's budget of {budget / 1e9:.1f} GB. To read it anyway, call again "
            f"with max_read_bytes={needed}; in a conversation with a person, ask them first "
            "when that is a large transfer. Or read a "
            + (
                f"smaller region (about {fits:.0f} km\u00b2 fits) "
                if area_km2
                else "smaller region "
            )
            + "or use a coarser product for the whole area, such as a 10 or 30 m DEM or a WCS "
            "coverage."
        )
