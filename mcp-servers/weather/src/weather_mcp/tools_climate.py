"""Tools on objects and climate: TempestExtremes blobs, indices, composites, runs, CMIP6."""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .app import FETCHES, WRITES, mcp, weather_tool
from .implementation import blobs, climate, views
from .implementation.common import (
    Output,
    Source,
    WeatherToolError,
    media_type,
    output_path,
    result,
    table_records,
    write_provenance,
)
from .implementation.grids import LAT, LON, open_grid, pick
from .saving import save_grid, save_table
from .tools_forecast import _file_source

HISTORICAL = [1995, 2014]
FUTURE = [2081, 2100]
TE_CITATION = "Ullrich et al. (2021), TempestExtremes v2.1, GMD 14, 5023-5048"


def _te_input(
    grid: str, variable: str, derive_magnitude: list[str] | None, work: Path
) -> tuple[Path, str, str, str]:
    """A NetCDF TempestExtremes can read (lon/lat names, time first), optionally |(a, b)|."""
    ds = open_grid(grid)
    lon, lat = pick(ds, LON), pick(ds, LAT)
    if derive_magnitude:
        a, b = derive_magnitude
        ds = ds.assign({variable: np.hypot(ds[a], ds[b])})
    if variable not in ds:
        raise WeatherToolError("bad_input", f"{Path(grid).name} lacks {variable}")
    da = ds[variable]
    time = next(d for d in da.dims if d not in (lon, lat))
    out = da.transpose(time, lat, lon).to_dataset(name=variable)
    path = work / "te_input.nc"
    out.to_netcdf(path)
    return path, str(lon), str(lat), str(time)


@mcp.tool(
    name="detect_blobs",
    title="Detect Blobs",
    description=(
        "TempestExtremes DetectBlobs then StitchBlobs on a gridded field: tag cells passing a threshold "
        "('var,op,value,distance', e.g. 'ivt,>=,250,0' for atmospheric rivers or a Z500 anomaly for "
        "blocks), keep objects of at least min_area_km2, and link them in time with a minimum lifetime "
        "and overlap. derive_magnitude computes the field as |(a, b)| first (IVT from the two flux "
        "components). Writes the stitched object field (NetCDF). The server finds its TempestExtremes "
        "build itself (TEMPESTEXTREMES_BIN or PATH), which a shell may not see; when none is installed "
        "the call returns backend_unavailable."
    ),
    annotations=WRITES,
    tags={"weather", "tracking", "atmospheric river", "blocking"},
)
@weather_tool("detect_blobs")
def detect_blobs(
    grid: Annotated[
        str, Field(description="Gridded NetCDF with the field (or its components).")
    ],
    variable: Annotated[
        str, Field(description="Field to threshold (name to give a derived magnitude).")
    ],
    threshold: Annotated[
        str, Field(description="TempestExtremes thresholdcmd, e.g. 'ivt,>=,250,0'.")
    ],
    min_area_km2: Annotated[
        float, Field(description="Smallest object kept, km².")
    ] = 0.0,
    min_time: Annotated[
        str,
        Field(description="Shortest lifetime, steps or with units (e.g. '5d', '24h')."),
    ] = "1",
    min_overlap_percent: Annotated[
        float | None,
        Field(description="Overlap with the previous step to continue an object, %."),
    ] = None,
    derive_magnitude: Annotated[
        list[str] | None, Field(description="[a, b] to compute the field as |(a, b)|.")
    ] = None,
    regional: Annotated[
        bool, Field(description="The grid is regional, not global.")
    ] = True,
    name: Annotated[str, Field(description="Name for the output.")] = "blobs",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    target = output_path(name, ".nc", output_dir)
    with tempfile.TemporaryDirectory(prefix="te-") as tmp:
        work = Path(tmp)
        infile, lon, lat, _ = _te_input(grid, variable, derive_magnitude, work)
        _, stitched, commands = blobs.detect_and_stitch(
            infile,
            work,
            threshold=threshold,
            min_size=int(min_area_km2),
            min_time=min_time,
            min_overlap=min_overlap_percent,
            lon=lon,
            lat=lat,
            extra_detect=[],
            regional=regional,
            timeout=3600,
        )
        target.write_bytes(stitched.read_bytes())
    table = blobs.blob_table(target, lon=lon, lat=lat, values=None, extra={})
    summary = blobs.object_summary(table)
    method = {
        "threshold": threshold,
        "min_area_km2": min_area_km2,
        "min_time": min_time,
        "min_overlap_percent": min_overlap_percent,
        "derive_magnitude": derive_magnitude,
        "regional": regional,
        "commands": commands,
        "citation": TE_CITATION,
    }
    sidecar = write_provenance(
        target,
        tool="detect_blobs",
        sources=[_file_source(grid)],
        method=method,
        packages=("xarray",),
        notes=[
            "TempestExtremes binaries built from github.com/ClimateGlobalChange/tempestextremes (BSD-2-Clause)"
        ],
    )
    return result(
        f"{summary['object'].nunique() if len(summary) else 0} objects tracked ({threshold}, lifetime >= {min_time}).",
        [
            Output(
                target,
                "grid",
                media_type(target),
                sidecar,
                view=views.grid_file_view("grid", target),
            )
        ],
        objects=int(summary["object"].nunique()) if len(summary) else 0,
        objects_table=table_records(summary),
        method=method,
    )


@mcp.tool(
    name="stitch_blobs",
    title="Stitch Blobs",
    description=(
        "Re-run TempestExtremes StitchBlobs on an already tagged blob file with a different minimum "
        "lifetime or overlap, without detecting again."
    ),
    annotations=WRITES,
    tags={"weather", "tracking"},
)
@weather_tool("stitch_blobs")
def stitch_blobs(
    tagged: Annotated[
        str,
        Field(description="NetCDF with a binary_tag variable (DetectBlobs output)."),
    ],
    min_time: Annotated[str, Field(description="Shortest lifetime.")] = "1",
    min_overlap_percent: Annotated[
        float | None, Field(description="Overlap with the previous step, %.")
    ] = None,
    regional: Annotated[bool, Field(description="The grid is regional.")] = True,
    name: Annotated[str, Field(description="Name for the output.")] = "stitched",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(tagged)
    lon, lat = pick(ds, LON), pick(ds, LAT)
    target = output_path(name, ".nc", output_dir)
    args = [
        blobs.binary("StitchBlobs"),
        "--in",
        str(Path(tagged).resolve()),
        "--out",
        str(target),
        "--var",
        "binary_tag",
        "--mintime",
        min_time,
        "--latname",
        str(lat),
        "--lonname",
        str(lon),
    ]
    if min_overlap_percent is not None:
        args += ["--min_overlap_prev", str(min_overlap_percent)]
    if regional:
        args.append("--regional")
    blobs._run(args, target.parent, 3600)
    summary = blobs.object_summary(
        blobs.blob_table(target, lon=str(lon), lat=str(lat), values=None, extra={})
    )
    sidecar = write_provenance(
        target,
        tool="stitch_blobs",
        sources=[_file_source(tagged)],
        method={"command": " ".join(args), "citation": TE_CITATION},
    )
    return result(
        f"{len(summary)} objects after stitching (lifetime >= {min_time}).",
        [
            Output(
                target,
                "grid",
                media_type(target),
                sidecar,
                view=views.grid_file_view("grid", target),
            )
        ],
        objects=len(summary),
    )


@mcp.tool(
    name="blob_stats",
    title="Blob Statistics",
    description=(
        "Per object and time: area, centroid, extent and the mean and maximum of chosen fields under "
        "the object (e.g. precipitation); per object: life, peak area and, given a region mask "
        "(e.g. coastal land), the first time it reaches the region (landfall) and the field summed "
        "over the region while it is there."
    ),
    annotations=WRITES,
    tags={"weather", "tracking", "statistics"},
)
@weather_tool("blob_stats")
def blob_stats(
    objects: Annotated[str, Field(description="Stitched object NetCDF (object_id).")],
    fields: Annotated[
        str | None,
        Field(description="NetCDF holding fields to summarise under objects."),
    ] = None,
    variables: Annotated[
        list[str], Field(description="Field variables to summarise.")
    ] = [],  # noqa: B006
    region: Annotated[
        str | None, Field(description="NetCDF with a 0/1 region mask on the same grid.")
    ] = None,
    region_variable: Annotated[
        str | None, Field(description="Mask variable (1 inside).")
    ] = None,
    region_box: Annotated[
        list[float] | None, Field(description="Restrict the mask to this box.")
    ] = None,
    accumulate: Annotated[
        str | None,
        Field(description="Variable summed over the region while the object is there."),
    ] = None,
    scale: Annotated[
        float,
        Field(
            description="Unit factor for the accumulated variable (1000 for m to mm)."
        ),
    ] = 1.0,
    name: Annotated[str, Field(description="Name for the outputs.")] = "blob-stats",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(objects)
    lon, lat = str(pick(ds, LON)), str(pick(ds, LAT))
    field_ds = (
        open_grid(fields, variables + ([accumulate] if accumulate else []))
        if fields
        else None
    )
    extra = (
        {
            v: field_ds[v].interp(
                {pick(field_ds, LON): ds[lon], pick(field_ds, LAT): ds[lat]}
            )
            for v in variables
        }
        if field_ds is not None
        else {}
    )
    table = blobs.blob_table(Path(objects), lon=lon, lat=lat, values=None, extra=extra)
    summary = blobs.object_summary(table)
    if region and region_variable and len(summary):
        mask_ds = open_grid(region, [region_variable])
        mask = mask_ds[region_variable]
        mask = mask.isel(
            {d: 0 for d in mask.dims if d not in (pick(mask, LON), pick(mask, LAT))}
        )
        mask = (
            mask.interp(
                {pick(mask, LON): ds[lon], pick(mask, LAT): ds[lat]}, method="nearest"
            )
            > 0.5
        )
        if region_box:
            inside = (
                (ds[lon] >= region_box[0])
                & (ds[lon] <= region_box[1])
                & (ds[lat] >= region_box[2])
                & (ds[lat] <= region_box[3])
            )
            mask = mask & inside
        ids = ds[next(v for v in ds.data_vars if "object" in str(v).lower())]
        tname = next(d for d in ids.dims if d not in (lon, lat))
        acc_field = (
            field_ds[accumulate].interp(
                {pick(field_ds, LON): ds[lon], pick(field_ds, LAT): ds[lat]}
            )
            * scale
            if accumulate and field_ds is not None
            else None
        )
        landfall, totals = [], []
        for obj in summary["object"]:
            hit = ((ids == obj) & mask).any([lon, lat])
            times = ids[tname].to_numpy()[hit.to_numpy()]
            landfall.append(pd.Timestamp(times[0]) if len(times) else pd.NaT)
            if acc_field is not None and len(times):
                footprint = (ids == obj) & mask
                totals.append(
                    float(
                        (
                            acc_field.sel({tname: times})
                            * footprint.sel({tname: times})
                        ).sum()
                        / max(int(mask.sum()), 1)
                    )
                )
            else:
                totals.append(np.nan)
        summary["landfall"] = landfall
        if accumulate:
            summary[f"{accumulate}_region_mean_total"] = totals
    prov = {
        "tool": "blob_stats",
        "sources": [_file_source(objects)]
        + ([_file_source(fields)] if fields else [])
        + ([_file_source(region)] if region else []),
        "method": {
            "variables": variables,
            "region_variable": region_variable,
            "region_box": region_box,
            "accumulate": accumulate,
            "scale": scale,
            "accumulation": "field summed over steps while the object covers the region, averaged over the region's cells",
        },
        "packages": ("xarray",),
    }
    outputs = [
        save_table(table, f"{name}-steps", "tracks", output_dir, **prov),
        save_table(summary, f"{name}-objects", "events", output_dir, **prov),
    ]
    landfalls = (
        int(summary["landfall"].notna().sum()) if "landfall" in summary else None
    )
    return result(
        f"{len(summary)} objects"
        + (f"; {landfalls} reached the region" if landfalls is not None else "")
        + ".",
        outputs,
        objects=len(summary),
        landfalls=landfalls,
        objects_table=table_records(summary, limit=200),
    )


@mcp.tool(
    name="climate_index",
    title="Climate Index",
    description=(
        "ENSO (Niño 3.4 SST anomaly from a stated base period, ONI, and El Niño / La Niña phases by the "
        "±0.5 °C five-season rule) or NAO (leading EOF of seasonal SLP anomalies over 20-80N, 90W-40E, "
        "positive with low pressure over Iceland), with the pattern for NAO."
    ),
    annotations=WRITES,
    tags={"weather", "climate", "ENSO", "NAO"},
)
@weather_tool("climate_index")
def climate_index(
    index: Annotated[Literal["enso", "nao"], Field(description="Which index.")],
    grid: Annotated[
        str, Field(description="SST (ENSO) or SLP (NAO) NetCDF, monthly or finer.")
    ],
    variable: Annotated[str, Field(description="Variable name.")],
    base_period: Annotated[
        list[int],
        Field(description="[first_year, last_year] for the anomalies (ENSO)."),
    ] = [1991, 2020],  # noqa: B006
    season: Annotated[
        str, Field(description="Season for the NAO (DJF, JFM, NDJFM, ...).")
    ] = "DJF",
    name: Annotated[str, Field(description="Name for the outputs.")] = "index",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    da = open_grid(grid, [variable])[variable]
    prov: dict[str, Any] = {
        "tool": "climate_index",
        "sources": [_file_source(grid)],
        "packages": ("numpy", "xarray"),
    }
    if index == "enso":
        table = climate.nino34(da, (base_period[0], base_period[1]))
        prov["method"] = {
            "index": "Nino 3.4 / ONI",
            "box": climate.NINO34,
            "base_period": base_period,
            "phase_rule": "ONI beyond +-0.5 C for >= 5 consecutive overlapping seasons",
        }
        outputs = [save_table(table, name, "series", output_dir, **prov)]
        counts = table["phase"].value_counts().to_dict()
        return result(
            f"Niño 3.4 from {str(table['time'].iloc[0])[:7]} to {str(table['time'].iloc[-1])[:7]}: months by phase {counts}.",
            outputs,
            phase_months={str(k): int(v) for k, v in counts.items()},
            method=prov["method"],
        )
    table, pattern, explained = climate.nao_eof(da, season)
    prov["method"] = {
        "index": "NAO, leading EOF",
        "box": climate.NAO_BOX,
        "season": season,
        "weighting": "sqrt(cos lat)",
        "sign": "positive = low pressure over Iceland",
        "variance_explained": explained,
    }
    outputs = [
        save_table(table, name, "series", output_dir, **prov),
        save_grid(pattern.to_dataset(), f"{name}-pattern", "grid", output_dir, **prov),
    ]
    return result(
        f"NAO ({season}) for {len(table)} seasons; the leading EOF explains {explained:.0%} of the variance.",
        outputs,
        variance_explained=explained,
        method=prov["method"],
    )


@mcp.tool(
    name="composite",
    title="Composite By Phase",
    description=(
        "Seasonal-mean anomaly composites of a field for the years in each phase of an index table "
        "(from climate_index), with Welch t-test p-values against the other years."
    ),
    annotations=WRITES,
    tags={"weather", "climate", "composite"},
)
@weather_tool("composite")
def composite(
    grid: Annotated[str, Field(description="Field NetCDF (e.g. precipitation).")],
    variable: Annotated[str, Field(description="Field variable.")],
    index_table: Annotated[str, Field(description="Index table from climate_index.")],
    kind: Annotated[Literal["enso", "nao"], Field(description="Index kind.")],
    season: Annotated[str, Field(description="Season (DJF, JJA, ...).")] = "DJF",
    nao_threshold: Annotated[
        float, Field(description="NAO phase threshold, standard deviations.")
    ] = 1.0,
    name: Annotated[str, Field(description="Name for the output.")] = "composite",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    idx = (
        pd.read_parquet(index_table)
        if index_table.endswith(".parquet")
        else pd.read_csv(index_table)
    )
    phases = climate.phase_years(idx, season=season, kind=kind, threshold=nao_threshold)
    out, numbers = climate.composite(
        open_grid(grid, [variable])[variable],
        phases,
        season=season,
        phase_column="phase",
    )
    saved = save_grid(
        out,
        name,
        "grid",
        output_dir,
        tool="composite",
        sources=[_file_source(grid), _file_source(index_table)],
        method={
            "kind": kind,
            "season": season,
            "nao_threshold": nao_threshold,
            **numbers,
        },
        packages=("scipy", "xarray"),
    )
    sizes = {k: len(v) for k, v in numbers["years_by_phase"].items()}
    return result(
        f"{season} composites by {kind} phase: years per phase {sizes}.",
        [saved],
        **numbers,
    )


@mcp.tool(
    name="evaluate_run",
    title="Evaluate Run",
    description=(
        "Evaluate a climate or weather model run against a reference (ERA5) on a common grid: time-mean "
        "bias map, zonal-mean bias, monthly seasonal cycles per named region and their RMSE."
    ),
    annotations=WRITES,
    tags={"weather", "climate", "evaluation", "model"},
)
@weather_tool("evaluate_run")
def evaluate_run(
    run: Annotated[str, Field(description="Model output NetCDF/Zarr (or glob).")],
    reference: Annotated[str, Field(description="Reference NetCDF/Zarr.")],
    variable: Annotated[str, Field(description="Run variable.")],
    reference_variable: Annotated[
        str | None, Field(description="Reference variable if named differently.")
    ] = None,
    regions: Annotated[
        dict[str, list[float]],
        Field(description="{name: [lon_min, lon_max, lat_min, lat_max]}."),
    ] = {"global": [-180, 180, -90, 90]},  # noqa: B006
    grid_step: Annotated[
        float | None,
        Field(description="Common grid step, degrees (default: the coarser grid)."),
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "evaluation",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    out, cycles, numbers = climate.evaluate_run(
        open_grid(run, [variable])[variable],
        open_grid(reference, [reference_variable or variable])[
            reference_variable or variable
        ],
        regions=regions,
        grid_step=grid_step,
    )
    prov = {
        "tool": "evaluate_run",
        "sources": [_file_source(run), _file_source(reference)],
        "method": {"regions": regions, **numbers},
        "packages": ("xarray",),
    }
    outputs = [
        save_grid(out, f"{name}-bias", "grid", output_dir, **prov),
        save_table(cycles, f"{name}-seasonal-cycle", "series", output_dir, **prov),
    ]
    return result(
        f"Run minus reference over {numbers['months']} months: global mean bias {numbers['global_mean_bias']:+.3g}, RMSE {numbers['global_rmse']:.3g}.",
        outputs,
        **numbers,
    )


@mcp.tool(
    name="cmip_change",
    title="CMIP6 Change",
    description=(
        "Regional change of a CMIP6 variable per model from the Pangeo cloud catalog (no key): future "
        "period minus historical period for one scenario, optionally for one season; per-model change, "
        "regional series and maps of change, multi-model mean and spread."
    ),
    annotations=FETCHES,
    tags={"weather", "climate", "CMIP6", "projections"},
)
@weather_tool("cmip_change")
def cmip_change(
    models: Annotated[list[str], Field(description="CMIP6 source_ids.")],
    experiment: Annotated[str, Field(description="Scenario (ssp245, ssp585, ...).")],
    box: Annotated[
        list[float], Field(description="[lon_min, lon_max, lat_min, lat_max].")
    ],
    variable: Annotated[
        str, Field(description="CMIP6 variable_id (tas, tasmax, pr, ...).")
    ] = "tas",
    table: Annotated[str, Field(description="table_id.")] = "Amon",
    member: Annotated[str, Field(description="member_id.")] = "r1i1p1f1",
    historical: Annotated[
        list[int], Field(description="[first, last] historical years.")
    ] = HISTORICAL,
    future: Annotated[
        list[int], Field(description="[first, last] future years.")
    ] = FUTURE,
    season: Annotated[
        str | None, Field(description="Season (JJA for summer), or the annual mean.")
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "cmip-change",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    changes, series, maps, sources = climate.cmip_change(
        models=models,
        experiment=experiment,
        variable=variable,
        table=table,
        member=member,
        box=box,
        historical=(historical[0], historical[1]),
        future=(future[0], future[1]),
        season=season,
    )
    srcs = [
        Source(
            name=f"CMIP6 {s['model']} {s['experiment']}",
            kind="product",
            reference=s["zstore"],
            version=s["version"],
            licence="CMIP6 data: CC BY 4.0 for most models; check each model's terms of use",
        )
        for s in sources
    ]
    prov = {
        "tool": "cmip_change",
        "sources": srcs,
        "method": {
            "box": box,
            "historical": historical,
            "future": future,
            "season": season,
            "member": member,
            "catalog": climate.PANGEO_CMIP6,
        },
        "packages": ("intake-esm", "xarray", "gcsfs"),
    }
    outputs = [
        save_table(changes, name, "table", output_dir, **prov),
        save_table(series, f"{name}-series", "series", output_dir, **prov),
        save_grid(maps, f"{name}-maps", "grid", output_dir, **prov),
    ]
    spread = float(changes["change"].std()) if len(changes) > 1 else 0.0
    return result(
        f"{experiment} {variable} change {future[0]}-{future[1]} vs {historical[0]}-{historical[1]}: multi-model mean {changes['change'].mean():+.3g}, spread (sd) {spread:.3g} across {len(changes)} models.",
        outputs,
        changes=changes.to_dict("records"),
        multi_model_mean=float(changes["change"].mean()),
        spread_sd=spread,
    )
