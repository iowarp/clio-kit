"""Tools on AI models: Earth2Studio forecasts, cyclone tracks and CorrDiff downscaling."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .app import WRITES, mcp, weather_tool
from .implementation import blobs, views
from .implementation.common import (
    Output,
    Source,
    WeatherToolError,
    media_type,
    output_path,
    result,
    write_provenance,
)
from .implementation.grids import LAT, LON, open_grid, pick
from .implementation.workers import run_worker
from .saving import save_table
from .tools_forecast import _file_source

CHECKPOINT_LICENCES = {
    "fcn": "FourCastNet checkpoint (nvidia/fourcastnet1 on Hugging Face): Apache-2.0",
    "dlwp": "DLWP checkpoint (NVIDIA): Apache-2.0",
    "corrdiff": "CorrDiff inference package (NGC nvidia/modulus/corrdiff_inference_package): NVIDIA licence on NGC; check before redistribution",
}
SOURCE_NOTES = {
    "arco": "ARCO ERA5 initial conditions (Copernicus licence / CC-BY 4.0)",
    "ifs": "ECMWF IFS open data initial conditions (CC-BY 4.0)",
    "gfs": "NOAA GFS initial conditions (public domain)",
}


@mcp.tool(
    name="ai_forecast",
    title="AI Forecast",
    description=(
        "Run an AI weather model with Earth2Studio from initial conditions (ARCO ERA5 for past events, "
        "IFS open data or GFS for recent ones): FourCastNet ('fcn') or DLWP ('dlwp'), steps of the "
        "model's time step (6 h). Records the checkpoint and its licence, the initial-condition source "
        "and time and the device. Needs a CUDA GPU unless allow_cpu (slow); the first call builds a "
        "large worker environment."
    ),
    annotations=WRITES,
    tags={"weather", "ai", "forecast", "earth2studio"},
)
@weather_tool("ai_forecast")
def ai_forecast(
    model: Annotated[Literal["fcn", "dlwp"], Field(description="Prognostic model.")],
    time: Annotated[
        str, Field(description="Initial time (ISO), e.g. 2023-08-27T00:00.")
    ],
    steps: Annotated[int, Field(description="Model steps (6 h each).")],
    source: Annotated[
        Literal["arco", "ifs", "gfs"], Field(description="Initial-condition source.")
    ] = "arco",
    variables: Annotated[
        list[str] | None,
        Field(description="Variables to keep (e.g. ['msl', 'u10m', 'v10m', 'z500'])."),
    ] = None,
    allow_cpu: Annotated[
        bool, Field(description="Run on the CPU when no GPU is present.")
    ] = False,
    name: Annotated[str, Field(description="Name for the output.")] = "ai-forecast",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    target = output_path(f"{name}-{model}", ".nc", output_dir)
    reply = run_worker(
        "earth2",
        {
            "task": "forecast",
            "model": model,
            "time": time,
            "steps": steps,
            "source": source,
            "variables": variables,
            "allow_cpu": allow_cpu,
            "output": str(target.resolve()),
        },
        timeout_s=6 * 3600,
    )
    src = Source(
        name=f"initial conditions: {source}",
        kind="service",
        reference=source,
        version=time,
        licence=SOURCE_NOTES[source],
    )
    sidecar = write_provenance(
        target,
        tool="ai_forecast",
        sources=[src],
        method={
            "model": model,
            "citation": reply["citation"],
            "checkpoint_licence": CHECKPOINT_LICENCES[model],
            "steps": steps,
            "device": reply["device"],
            "earth2studio": "0.20.0",
        },
        notes=[CHECKPOINT_LICENCES[model]],
    )
    return result(
        f"{model.upper()} from {time} ({source}), {steps} steps on {reply['device']}.",
        [
            Output(
                target,
                "grid",
                media_type(target),
                sidecar,
                view=views.grid_file_view("grid", target),
            )
        ],
        model=model,
        device=reply["device"],
        checkpoint_licence=CHECKPOINT_LICENCES[model],
    )


def _valid_time_file(path: str, work: Path) -> tuple[Path, str, str]:
    """Earth2Studio output (time, lead_time, lat, lon) as a valid-time file for TempestExtremes."""
    ds = open_grid(path)
    lon, lat = str(pick(ds, LON)), str(pick(ds, LAT))
    if "lead_time" in ds.dims:
        if "time" in ds.dims and ds.sizes["time"] > 1:
            raise WeatherToolError("bad_input", "track one initialisation at a time")
        init = pd.Timestamp(np.atleast_1d(ds["time"].to_numpy())[0])
        ds = ds.isel(time=0, drop=True) if "time" in ds.dims else ds
        valid = [init + pd.Timedelta(x) for x in ds["lead_time"].to_numpy()]
        ds = ds.assign_coords(lead_time=valid).rename({"lead_time": "time"})
    ds = ds.transpose("time", lat, lon, ...)
    out = work / "valid.nc"
    ds.to_netcdf(out)
    return out, lon, lat


@mcp.tool(
    name="tc_track",
    title="Track Cyclones",
    description=(
        "Track tropical cyclones in a forecast (or reanalysis) field. backend 'tempestextremes' (CPU): "
        "DetectNodes on sea-level pressure minima with a closed-contour criterion and StitchNodes into "
        "tracks (Zarzycki & Ullrich 2017 style); 'earth2studio' (TCTrackerWuDuan, needs CUDA). Optionally "
        "keep the track nearest a start position. Writes a track table with lead time, position, MSLP and wind."
    ),
    annotations=WRITES,
    tags={"weather", "cyclone", "tracking"},
)
@weather_tool("tc_track")
def tc_track(
    forecast: Annotated[
        str,
        Field(description="Forecast NetCDF with sea-level pressure (and 10 m winds)."),
    ],
    model: Annotated[str, Field(description="Label for the forecast (model name).")],
    backend: Annotated[
        Literal["tempestextremes", "earth2studio"], Field(description="Tracker.")
    ] = "tempestextremes",
    mslp: Annotated[
        str, Field(description="Sea-level pressure variable (Pa).")
    ] = "msl",
    u10: Annotated[str | None, Field(description="10 m eastward wind.")] = "u10m",
    v10: Annotated[str | None, Field(description="10 m northward wind.")] = "v10m",
    closed_contour_pa: Annotated[
        float, Field(description="MSLP rise required within the contour distance, Pa.")
    ] = 200.0,
    contour_distance_degrees: Annotated[
        float, Field(description="Closed-contour distance, degrees.")
    ] = 5.5,
    min_time: Annotated[str, Field(description="Shortest track (e.g. '24h').")] = "24h",
    start: Annotated[
        list[float] | None,
        Field(description="[lon, lat] to keep the nearest track at the first time."),
    ] = None,
    name: Annotated[str, Field(description="Name for the output.")] = "tc-track",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if backend == "earth2studio":
        raise WeatherToolError(
            "backend_unavailable",
            "Earth2Studio's TCTrackerWuDuan/TCTrackerVitart need CUDA (cupy and cuCIM); this machine has no GPU. "
            "Use backend 'tempestextremes' (CPU) or run on a GPU machine.",
        )
    with tempfile.TemporaryDirectory(prefix="tc-") as tmp:
        work = Path(tmp)
        infile, lon, lat = _valid_time_file(forecast, work)
        nodes = work / "nodes.txt"
        out_cmd = f"{mslp},min,0" + (
            f";_VECMAG({u10},{v10}),max,2" if u10 and v10 else ""
        )
        detect = [
            blobs.binary("DetectNodes"),
            "--in_data",
            str(infile),
            "--out",
            str(nodes),
            "--searchbymin",
            mslp,
            "--closedcontourcmd",
            f"{mslp},{closed_contour_pa},{contour_distance_degrees},0",
            "--mergedist",
            "6.0",
            "--outputcmd",
            out_cmd,
            "--latname",
            lat,
            "--lonname",
            lon,
            "--regional",
        ]
        blobs._run(detect, work, 3600)
        tracks_txt = work / "tracks.txt"
        stitch = [
            blobs.binary("StitchNodes"),
            "--in",
            str(nodes),
            "--out",
            str(tracks_txt),
            "--in_fmt",
            "lon,lat,msl" + (",wind" if u10 and v10 else ""),
            "--range",
            "8.0",
            "--mintime",
            min_time,
            "--maxgap",
            "1",
            "--out_file_format",
            "csv",
        ]
        blobs._run(stitch, work, 3600)
        table = pd.read_csv(tracks_txt, skipinitialspace=True)
    table.columns = [c.strip() for c in table.columns]
    if table.empty:
        raise WeatherToolError("no_data", "no cyclone track met the criteria")
    times = pd.to_datetime(
        {
            "year": table["year"],
            "month": table["month"],
            "day": table["day"],
            "hour": table["hour"],
        },
        utc=True,
    )
    frame = pd.DataFrame(
        {
            "model": model,
            "track": table["track_id"],
            "time": times,
            "longitude": ((table["lon"] + 180) % 360) - 180,
            "latitude": table["lat"],
            "mslp_hpa": table["msl"] / 100.0,
        }
    )
    if "wind" in table:
        frame["wind_kt"] = table["wind"] * 1.943844
    frame["lead_hours"] = (
        frame["time"] - frame["time"].min()
    ).dt.total_seconds() / 3600
    if start:
        first = frame.sort_values("time").groupby("track").first()
        dist = np.hypot(
            (first["longitude"] - start[0]) * np.cos(np.radians(start[1])),
            first["latitude"] - start[1],
        )
        frame = frame[frame["track"] == dist.idxmin()]
    out = save_table(
        frame,
        f"{name}-{model}",
        "tracks",
        output_dir,
        tool="tc_track",
        sources=[_file_source(forecast, model)],
        method={
            "backend": backend,
            "detect": " ".join(detect[3:]),
            "stitch": " ".join(stitch[3:]),
            "start": start,
            "citation": "Ullrich et al. (2021) TempestExtremes v2.1; Zarzycki & Ullrich (2017)",
        },
        packages=("pandas",),
    )
    return result(
        f"{frame['track'].nunique()} track(s) in {model}; lowest MSLP {frame['mslp_hpa'].min():.0f} hPa.",
        [out],
        tracks=int(frame["track"].nunique()),
        model=model,
    )


@mcp.tool(
    name="downscale",
    title="Downscale",
    description=(
        "Downscale coarse fields to about 2 km with NVIDIA CorrDiff where a trained model exists: "
        "Earth2Studio's CorrDiffTaiwan (its fixed grid around Taiwan, from ERA5 or GFS inputs). Other "
        "regions need a trained model and are refused. Needs a CUDA GPU unless allow_cpu (very slow)."
    ),
    annotations=WRITES,
    tags={"weather", "downscaling", "ai"},
)
@weather_tool("downscale")
def downscale(
    time: Annotated[str, Field(description="Valid time (ISO).")],
    region: Annotated[
        Literal["taiwan"], Field(description="Region with a trained model.")
    ] = "taiwan",
    source: Annotated[
        Literal["arco", "gfs"], Field(description="Coarse input source.")
    ] = "arco",
    samples: Annotated[int, Field(description="Diffusion samples.")] = 1,
    allow_cpu: Annotated[
        bool, Field(description="Run on the CPU when no GPU is present.")
    ] = False,
    name: Annotated[str, Field(description="Name for the output.")] = "corrdiff",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    target = output_path(name, ".nc", output_dir)
    reply = run_worker(
        "earth2",
        {
            "task": "downscale",
            "time": time,
            "source": source,
            "samples": samples,
            "allow_cpu": allow_cpu,
            "output": str(target.resolve()),
        },
        timeout_s=6 * 3600,
    )
    sidecar = write_provenance(
        target,
        tool="downscale",
        sources=[
            Source(
                name=f"coarse input: {source}",
                kind="service",
                reference=source,
                version=time,
                licence=SOURCE_NOTES[source],
            )
        ],
        method={
            "model": "CorrDiffTaiwan",
            "region": region,
            "samples": samples,
            "device": reply["device"],
            "citation": reply["citation"],
            "checkpoint_licence": CHECKPOINT_LICENCES["corrdiff"],
        },
        notes=[CHECKPOINT_LICENCES["corrdiff"]],
    )
    return result(
        f"CorrDiffTaiwan at {time} ({samples} sample(s)) on {reply['device']}.",
        [
            Output(
                target,
                "grid",
                media_type(target),
                sidecar,
                view=views.grid_file_view("grid", target),
            )
        ],
        device=reply["device"],
        region=region,
    )
