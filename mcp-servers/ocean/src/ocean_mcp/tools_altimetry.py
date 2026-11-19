"""Tools on altimetry and tracks: eddies, track crossings and SWOT swaths."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from .app import FETCHES, WRITES, mcp, ocean_tool, open_grid
from .implementation import fetch, views
from .implementation.common import (
    Output,
    Source,
    local_file_source,
    media_type,
    output_path,
    result,
    write_provenance,
)
from .implementation.workers import run_worker
from .tools_circulation import _grid_source
from .tools_profiles import _save_table


@mcp.tool(
    name="eddy_track",
    title="Track Eddies",
    description=(
        "Identify mesoscale eddies in daily gridded sea level (SLA or ADT) with py-eddy-tracker "
        "(closed contours after a Bessel high-pass at a stated wavelength) and link them into tracks; "
        "returns a track table (time, position, amplitude, speed radius, polarity). Runs in a separate "
        "GPL worker environment; the first call builds it."
    ),
    annotations=WRITES,
    tags={"ocean", "eddies", "altimetry", "tracking"},
)
@ocean_tool("eddy_track")
def eddy_track(
    grid: Annotated[
        str, Field(description="Daily sea level NetCDF on a regular lon-lat grid.")
    ],
    variable: Annotated[
        str, Field(description="Height variable (sla or adt), in metres.")
    ] = "sla",
    longitude: Annotated[
        str, Field(description="Longitude coordinate name.")
    ] = "longitude",
    latitude: Annotated[
        str, Field(description="Latitude coordinate name.")
    ] = "latitude",
    time: Annotated[str, Field(description="Time coordinate name.")] = "time",
    start: Annotated[str | None, Field(description="First day.")] = None,
    end: Annotated[str | None, Field(description="Last day.")] = None,
    wavelength_km: Annotated[
        float | None,
        Field(
            description="Bessel high-pass wavelength, km (e.g. 700 for SLA); none to skip."
        ),
    ] = 700.0,
    step_m: Annotated[float, Field(description="Contour step, metres.")] = 0.002,
    shape_error: Annotated[
        float, Field(description="Maximum shape error, percent.")
    ] = 55.0,
    virtual: Annotated[
        int, Field(description="Days an eddy may be missed and still tracked.")
    ] = 4,
    min_lifetime_days: Annotated[
        int, Field(description="Shortest track kept, days.")
    ] = 10,
    product: Annotated[
        str | None, Field(description="Product name/DOI for provenance.")
    ] = None,
    licence: Annotated[
        str | None, Field(description="Data licence of the product.")
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "eddies",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    target = output_path(name, ".parquet", output_dir)
    request = {
        "input": str(Path(grid).expanduser().resolve()),
        "variable": variable,
        "x": longitude,
        "y": latitude,
        "time": time,
        "start": start,
        "end": end,
        "wavelength_km": wavelength_km,
        "step_m": step_m,
        "shape_error": shape_error,
        "virtual": virtual,
        "output_dir": str(target.parent),
        "stem": name,
    }
    reply = run_worker("eddy", request, timeout_s=3 * 3600)
    tracks = pd.read_csv(reply["tracks_path"], parse_dates=["time"])
    if len(tracks):
        life = tracks.groupby("track")["time"].agg(
            lambda t: (t.max() - t.min()).days + 1
        )
        tracks = tracks[tracks["track"].map(life) >= min_lifetime_days]
    Path(reply["tracks_path"]).unlink(missing_ok=True)
    tracks.to_parquet(target, index=False)
    method = {
        k: v for k, v in request.items() if k not in ("input", "output_dir", "stem")
    } | {"min_lifetime_days": min_lifetime_days, "worker": reply["worker"]}
    sidecar = write_provenance(
        target,
        tool="eddy_track",
        sources=[_grid_source(grid, product, licence)],
        method=method,
        packages=("pandas",),
        notes=[
            f"py-eddy-tracker {reply.get('py_eddy_tracker_version')} ran in a separate GPL-3.0 process"
        ],
    )
    polarity = (
        tracks.groupby("polarity")["track"].nunique().to_dict() if len(tracks) else {}
    )
    return result(
        f"Identified {reply['identified']['anticyclonic']} anticyclonic and {reply['identified']['cyclonic']} cyclonic eddy observations; "
        f"{tracks['track'].nunique() if len(tracks) else 0} tracks live at least {min_lifetime_days} days {polarity}.",
        [
            Output(
                target,
                "tracks",
                media_type(target),
                sidecar,
                rows=len(tracks),
                view=views.table_view("tracks", target, tracks),
            )
        ],
        identified=reply["identified"],
        tracks=int(tracks["track"].nunique()) if len(tracks) else 0,
        tracks_by_polarity={str(k): int(v) for k, v in polarity.items()},
        track_paths=track_paths(tracks),
    )


#: Most tracks whose start, end and drift a result lists (longest first).
MAX_TRACK_PATHS = 40


def track_paths(tracks: pd.DataFrame) -> list[dict[str, Any]]:
    """Where each track began and ended, how long it lived and how far it drifted.

    A track's origin is what ties it to a known eddy source, so it is read here
    rather than left in the table: longest tracks first, at most MAX_TRACK_PATHS.
    """

    if not len(tracks):
        return []
    rows = []
    for track, group in tracks.sort_values("time").groupby("track", sort=False):
        first, last = group.iloc[0], group.iloc[-1]
        lat = np.radians((first["latitude"] + last["latitude"]) / 2)
        east_km = (last["longitude"] - first["longitude"]) * 111.32 * np.cos(lat)
        north_km = (last["latitude"] - first["latitude"]) * 110.57
        rows.append(
            {
                "track": str(track),
                "polarity": str(first.get("polarity", "")),
                "start": first["time"].strftime("%Y-%m-%d"),
                "end": last["time"].strftime("%Y-%m-%d"),
                "days": int((last["time"] - first["time"]).days) + 1,
                "start_position": [
                    round(float(first["longitude"]), 2),
                    round(float(first["latitude"]), 2),
                ],
                "end_position": [
                    round(float(last["longitude"]), 2),
                    round(float(last["latitude"]), 2),
                ],
                "drift_east_km": round(float(east_km)),
                "drift_north_km": round(float(north_km)),
            }
        )
    rows.sort(key=lambda row: -row["days"])
    return rows[:MAX_TRACK_PATHS]


@mcp.tool(
    name="track_crossings",
    title="Track Crossings",
    description=(
        "Which tracks (eddies, particles, storms) cross a contour of a field, such as an isobath of a "
        "bathymetry grid: a track crosses when any of its points lies on the shallow (or low) side. "
        "Returns the count and the crossing tracks with the first crossing time."
    ),
    annotations=WRITES,
    tags={"tracks", "isobath"},
)
@ocean_tool("track_crossings")
def track_crossings(
    tracks: Annotated[
        str, Field(description="Track table with track, time, longitude, latitude.")
    ],
    field: Annotated[str, Field(description="Gridded field (e.g. bathymetry) NetCDF.")],
    variable: Annotated[
        str, Field(description="Field variable (e.g. elevation or depth).")
    ],
    level: Annotated[
        float,
        Field(
            description="Contour level (e.g. -200 for the 200 m isobath of elevation)."
        ),
    ],
    side: Annotated[
        Literal["above", "below"], Field(description="Side that counts as crossed.")
    ] = "above",
    name: Annotated[str, Field(description="Name for the output.")] = "crossings",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table = (
        pd.read_parquet(tracks)
        if tracks.endswith(".parquet")
        else pd.read_csv(tracks, parse_dates=["time"])
    )
    ds = open_grid(field, [variable])
    lon_name = next(n for n in ("longitude", "lon", "x") if n in ds.coords)
    lat_name = next(n for n in ("latitude", "lat", "y") if n in ds.coords)
    values = (
        ds[variable]
        .interp(
            {
                lon_name: xr.DataArray(table["longitude"].to_numpy(), dims="p"),
                lat_name: xr.DataArray(table["latitude"].to_numpy(), dims="p"),
            }
        )
        .to_numpy()
    )
    table["field"] = values
    table["crossed"] = values > level if side == "above" else values < level
    first = (
        table[table["crossed"]]
        .sort_values("time")
        .groupby("track")
        .first()
        .reset_index()
    )
    out = _save_table(
        first,
        name,
        "tracks",
        output_dir,
        tool="track_crossings",
        sources=[local_file_source(Path(tracks)), local_file_source(Path(field))],
        method={"variable": variable, "level": level, "side": side},
        packages=("xarray",),
    )
    total = int(table["track"].nunique())
    return result(
        f"{len(first)} of {total} tracks reach the {side} side of {variable} = {level:g}.",
        [out],
        crossing=len(first),
        total=total,
        tracks_crossing=first["track"].tolist()[:500],
    )


@mcp.tool(
    name="swath_fetch",
    title="Fetch SWOT Swaths",
    description=(
        "Search SWOT L2 LR sea surface height granules over a box and period (no login needed), or "
        "download them with download=true (needs NASA Earthdata Login in the connector)."
    ),
    annotations=FETCHES,
    tags={"ocean", "swot", "altimetry", "fetch"},
)
@ocean_tool("swath_fetch")
def swath_fetch(
    box: Annotated[
        list[float], Field(description="[lon_min, lon_max, lat_min, lat_max].")
    ],
    start: Annotated[str, Field(description="Start date.")],
    end: Annotated[str, Field(description="End date.")],
    short_name: Annotated[
        str, Field(description="CMR short name of the SWOT collection.")
    ] = fetch.SWOT_SHORT_NAME,
    download: Annotated[
        bool, Field(description="Download the granules (needs Earthdata Login).")
    ] = False,
    output_dir: Annotated[
        str | None, Field(description="Directory for downloads.")
    ] = None,
) -> dict[str, Any]:
    if not download:
        found = fetch.swath_search(box, start, end, short_name)
        return result(
            f"{len(found)} {short_name} granules over the box from {start} to {end}.",
            [],
            granules=found,
        )
    folder = output_path("swot", "", output_dir)
    paths = fetch.swath_download(box, start, end, short_name, folder)
    outputs = []
    for p in paths:
        sidecar = write_provenance(
            p,
            tool="swath_fetch",
            sources=[
                Source(
                    name=short_name,
                    kind="service",
                    reference="NASA PO.DAAC via earthaccess",
                    subset={"box": box, "start": start, "end": end},
                    licence="NASA data: no restrictions on use; cite the dataset DOI",
                )
            ],
            method={},
            packages=("earthaccess",),
        )
        outputs.append(Output(p, "swath", media_type(p), sidecar))
    return result(f"Downloaded {len(paths)} SWOT granules.", outputs)


@mcp.tool(
    name="swath_compare",
    title="Compare Swath",
    description=(
        "Compare SWOT swath sea surface height with a gridded product at each pass time inside a box "
        "(each pass's mean removed from both): a pixel table with both values and the difference."
    ),
    annotations=WRITES,
    tags={"ocean", "swot", "altimetry"},
)
@ocean_tool("swath_compare")
def swath_compare(
    swaths: Annotated[list[str], Field(description="SWOT granule NetCDF files.")],
    grid: Annotated[str, Field(description="Gridded sea level NetCDF.")],
    box: Annotated[
        list[float], Field(description="[lon_min, lon_max, lat_min, lat_max].")
    ],
    swath_variable: Annotated[
        str, Field(description="Swath variable (e.g. ssha_karin_2).")
    ] = "ssha_karin_2",
    grid_variable: Annotated[
        str, Field(description="Gridded variable (e.g. sla).")
    ] = "sla",
    name: Annotated[str, Field(description="Name for the output.")] = "swath-compare",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table, numbers = fetch.swath_compare(
        [Path(s) for s in swaths],
        open_grid(grid, [grid_variable]),
        swath_variable=swath_variable,
        grid_variable=grid_variable,
        box=box,
    )
    out = _save_table(
        table,
        name,
        "points",
        output_dir,
        tool="swath_compare",
        sources=[local_file_source(Path(s)) for s in swaths]
        + [local_file_source(Path(grid))],
        method={"box": box, **numbers},
        packages=("xarray",),
    )
    return result(
        f"{numbers['passes']} passes, {numbers['pixels']} pixels; RMS difference {numbers['rms_difference']:.3f}.",
        [out],
        **numbers,
    )
