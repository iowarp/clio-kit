"""TempestExtremes DetectBlobs, StitchBlobs and BlobStats, run as subprocesses.

The binaries come from a TempestExtremes build on this machine
(``TEMPESTEXTREMES_BIN`` names its ``bin`` directory, otherwise PATH).
Arguments follow the TempestExtremes documentation: ``--thresholdcmd
"var,op,value,distance"``, ``--minsize``, ``--mintime``, ``--out_list``.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from .common import WeatherToolError


def binary(name: str) -> str:
    folder = os.environ.get("TEMPESTEXTREMES_BIN")
    if folder:
        path = Path(folder).expanduser() / name
        if path.is_file():
            return str(path)
        raise WeatherToolError(
            "backend_unavailable", f"{name} is not in TEMPESTEXTREMES_BIN={folder}"
        )
    found = shutil.which(name)
    if not found:
        raise WeatherToolError(
            "backend_unavailable",
            f"TempestExtremes {name} is not installed; build it from https://github.com/ClimateGlobalChange/tempestextremes "
            "(cmake + NetCDF) and set TEMPESTEXTREMES_BIN to its bin directory",
        )
    return found


def _run(args: list[str], work: Path, timeout: float) -> str:
    done = subprocess.run(
        args, cwd=work, capture_output=True, text=True, timeout=timeout, check=False
    )
    if done.returncode != 0:
        tail = (done.stderr or done.stdout).strip().splitlines()[-8:]
        raise WeatherToolError(
            "worker_failed", f"{Path(args[0]).name} failed: " + " | ".join(tail)
        )
    return done.stdout


def detect_and_stitch(
    infile: Path,
    work: Path,
    *,
    threshold: str,
    min_size: int,
    min_time: str,
    min_overlap: float | None,
    lon: str,
    lat: str,
    extra_detect: list[str],
    regional: bool,
    timeout: float,
) -> tuple[Path, Path, str]:
    """DetectBlobs then StitchBlobs; returns (tagged, stitched) NetCDF and the command lines."""
    tagged = work / "blobs_tagged.nc"
    stitched = work / "blobs_stitched.nc"
    detect = [
        binary("DetectBlobs"),
        "--in_data",
        str(infile),
        "--out",
        str(tagged),
        "--thresholdcmd",
        threshold,
        "--minabslat",
        "0",
        "--geofiltercmd",
        f"area,>=,{min_size}km2",
        "--latname",
        lat,
        "--lonname",
        lon,
        *extra_detect,
    ]
    if regional:
        detect.append("--regional")
    _run(detect, work, timeout)
    var = "binary_tag"
    stitch = [
        binary("StitchBlobs"),
        "--in",
        str(tagged),
        "--out",
        str(stitched),
        "--var",
        var,
        "--mintime",
        min_time,
        "--latname",
        lat,
        "--lonname",
        lon,
    ]
    if min_overlap is not None:
        stitch += ["--min_overlap_prev", str(min_overlap)]
    if regional:
        stitch.append("--regional")
    _run(stitch, work, timeout)
    return tagged, stitched, " ".join(detect) + "\n" + " ".join(stitch)


def blob_table(
    stitched: Path,
    *,
    lon: str,
    lat: str,
    values: xr.DataArray | None,
    extra: dict[str, xr.DataArray],
) -> pd.DataFrame:
    """Per object and time: area, centroid, extent and (optionally) the mean of fields under it.

    Computed from the stitched object IDs directly (equivalent to BlobStats'
    centlon, centlat, area and minlat/maxlat, plus field sums), so the table
    is a plain frame the views can read.
    """
    with xr.open_dataset(stitched) as ds:
        name = next(v for v in ds.data_vars if "object" in str(v).lower())
        ids = ds[name].load()
    tname = next(d for d in ids.dims if d not in (lon, lat))
    latv = ids[lat].to_numpy()
    lonv = ids[lon].to_numpy()
    dlat = abs(float(np.median(np.diff(latv))))
    dlon = abs(float(np.median(np.diff(lonv))))
    cell_area = (
        (6371.0**2)
        * np.radians(dlat)
        * np.radians(dlon)
        * np.cos(np.radians(latv))[:, None]
    )
    rows = []
    for t_index in range(ids.sizes[tname]):
        frame = ids.isel({tname: t_index}).transpose(lat, lon).to_numpy()
        present = np.unique(frame[frame > 0])
        when = pd.Timestamp(ids[tname].to_numpy()[t_index])
        for obj in present:
            mask = frame == obj
            area = cell_area * mask
            yy, xx = np.nonzero(mask)
            centre = float(
                np.degrees(
                    np.angle((np.exp(1j * np.radians(lonv))[None, :] * area).sum())
                )
            )
            # Extents relative to the centroid, so an object across 0° (or 180°) has
            # its real west and east edges rather than the grid's first and last.
            offsets = (lonv[xx] - centre + 180.0) % 360.0 - 180.0
            record: dict[str, Any] = {
                "object": int(obj),
                "time": when,
                "area_km2": float(area.sum()),
                "centroid_lat": float((latv[:, None] * area).sum() / area.sum()),
                "centroid_lon": centre,
                "lat_min": float(latv[yy].min()),
                "lat_max": float(latv[yy].max()),
                "lon_min": _wrap(centre + float(offsets.min())),
                "lon_max": _wrap(centre + float(offsets.max())),
                "cells": int(mask.sum()),
            }
            fields = dict(extra)
            if values is not None:
                fields["value"] = values
            for label, field in fields.items():
                f = (
                    field.isel({tname: t_index}).transpose(lat, lon).to_numpy()
                    if tname in field.dims
                    else field.transpose(lat, lon).to_numpy()
                )
                record[f"{label}_mean"] = float(np.nanmean(f[mask]))
                record[f"{label}_max"] = float(np.nanmax(f[mask]))
            rows.append(record)
    return pd.DataFrame(rows)


def object_summary(
    table: pd.DataFrame, region_mask: xr.DataArray | None = None
) -> pd.DataFrame:
    """One row per object: start, end, duration, peak area, mean centroid."""
    if table.empty:
        return pd.DataFrame(
            columns=["object", "start", "end", "duration_hours", "area_max_km2"]
        )
    out = (
        table.groupby("object")
        .agg(
            start=("time", "min"),
            end=("time", "max"),
            steps=("time", "size"),
            area_max_km2=("area_km2", "max"),
            centroid_lat_mean=("centroid_lat", "mean"),
            centroid_lon_mean=("centroid_lon", _circular_mean),
        )
        .reset_index()
    )
    out["duration_hours"] = (out["end"] - out["start"]).dt.total_seconds() / 3600
    return out


def _wrap(lon: float) -> float:
    """A longitude in [-180, 180)."""
    return (lon + 180.0) % 360.0 - 180.0


def _circular_mean(lons: pd.Series) -> float:
    """The mean direction of longitudes, so a mean across 0° or 180° is not halfway round."""
    return float(np.degrees(np.angle(np.exp(1j * np.radians(lons.to_numpy())).mean())))
