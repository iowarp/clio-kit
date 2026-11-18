"""Tools on profiles and sections: casts, Argo, gliders, TEOS-10, water masses."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .app import FETCHES, WRITES, expand_paths, mcp, ocean_tool, open_grid
from .implementation import fetch, hydrography, sections, views
from .implementation.common import (
    OceanToolError,
    Output,
    Source,
    local_file_source,
    media_type,
    output_path,
    result,
    write_provenance,
)
from .implementation.profiles_io import (
    profile_summary,
    read_casts,
    read_profiles,
    write_profiles,
)

TEOS = ("gsw", "pandas", "pyarrow")


def _save_table(
    table: pd.DataFrame, name: str, role: str, output_dir: str | None, **prov: Any
) -> Output:
    path = output_path(name, ".parquet", output_dir)
    write_profiles(table, path) if "pressure" in table else table.to_parquet(
        path, index=False
    )
    sidecar = write_provenance(path, **prov)
    view = views.table_view(role, path, table)
    return Output(path, role, media_type(path), sidecar, rows=len(table), view=view)


def _save_grid(
    ds: Any, name: str, role: str, output_dir: str | None, **prov: Any
) -> Output:
    path = output_path(name, ".nc", output_dir)
    ds.to_netcdf(path)
    sidecar = write_provenance(path, **prov)
    return Output(
        path, role, media_type(path), sidecar, view=views.grid_view(role, path, ds)
    )


@mcp.tool(
    name="ctd_read",
    title="Read CTD Casts",
    description=(
        "Read CTD cast files (Sea-Bird .cnv, CCHDO WHP-exchange _ct1.csv or a zip of them, CF NetCDF, "
        "or CSV) into one profile table (Parquet) with profile, time, latitude, longitude, pressure and "
        "the measured variables. Reports profiles, depth range, positions and the formats read."
    ),
    annotations=WRITES,
    tags={"ocean", "ctd", "profiles"},
)
@ocean_tool("ctd_read")
def ctd_read(
    paths: Annotated[
        list[str], Field(description="Cast files or zips; globs allowed.")
    ],
    name: Annotated[str, Field(description="Name for the output table.")] = "casts",
    columns: Annotated[
        dict[str, str] | None,
        Field(
            description="For CSV only: table column -> CSV column, e.g. {'pressure': 'PRES'}."
        ),
    ] = None,
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    files = expand_paths(paths)
    if not files:
        raise OceanToolError("bad_input", f"no files match {paths}")
    table, formats = read_casts(files, columns)
    out = _save_table(
        table,
        name,
        "profiles",
        output_dir,
        tool="ctd_read",
        sources=[local_file_source(f) for f in files[:200]],
        method={
            "formats": formats,
            "qc": "WHP-exchange WOCE flags 2 and 6 kept; others set missing",
        },
        packages=TEOS,
    )
    stats = profile_summary(table)
    return result(
        f"Read {stats['profiles']} casts ({stats['samples']} samples, to {stats['pressure_max_dbar']:.0f} dbar) from {len(files)} file(s) [{', '.join(formats)}].",
        [out],
        **stats,
        formats=formats,
    )


@mcp.tool(
    name="teos10",
    title="TEOS-10 Properties",
    description=(
        "Add TEOS-10 properties to a profile table with gsw: Absolute Salinity (SA), Conservative "
        "Temperature (CT), potential temperature (pt0), sigma0 and depth; optionally buoyancy "
        "frequency N² per profile as a second table. Needs in-situ temperature and practical salinity."
    ),
    annotations=WRITES,
    tags={"ocean", "teos-10", "profiles"},
)
@ocean_tool("teos10")
def teos10(
    profiles: Annotated[str, Field(description="Profile table (.parquet or .csv).")],
    buoyancy_frequency: Annotated[
        bool, Field(description="Also write N² per profile.")
    ] = True,
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table = hydrography.add_teos10(read_profiles(profiles))
    stem = Path(profiles).stem + "-teos10"
    prov = {
        "tool": "teos10",
        "sources": [local_file_source(Path(profiles))],
        "method": {
            "functions": [
                "gsw.SA_from_SP",
                "gsw.CT_from_t",
                "gsw.pt0_from_t",
                "gsw.sigma0",
                "gsw.Nsquared",
            ]
        },
        "packages": TEOS,
    }
    outputs = [_save_table(table, stem, "profiles", output_dir, **prov)]
    numbers: dict[str, Any] = {
        "CT_range": [float(table["CT"].min()), float(table["CT"].max())],
        "SA_range": [float(table["SA"].min()), float(table["SA"].max())],
        "sigma0_range": [float(table["sigma0"].min()), float(table["sigma0"].max())],
    }
    if buoyancy_frequency:
        n2 = hydrography.buoyancy_frequency(table)
        outputs.append(
            _save_table(n2, Path(profiles).stem + "-n2", "profiles", output_dir, **prov)
        )
        numbers["N2_max"] = float(n2["N2"].max())
    return result(
        f"TEOS-10 for {table['profile'].nunique()} profiles: CT {numbers['CT_range'][0]:.2f} to {numbers['CT_range'][1]:.2f} °C, "
        f"sigma0 {numbers['sigma0_range'][0]:.2f} to {numbers['sigma0_range'][1]:.2f} kg m-3.",
        outputs,
        **numbers,
    )


@mcp.tool(
    name="water_masses",
    title="Water Masses",
    description=(
        "Assign each sample of a profile table to a water mass by its potential temperature, "
        "salinity and pressure against a stated regional definition: a built-in one "
        "(atlantic_emery_meincke_1986) or your own {name, citation, masses:[{name, theta:[lo,hi], "
        "salinity:[lo,hi], pressure:[lo,hi]}]}. Returns the definition used, counts and where each "
        "mass sits (pressure, latitude, mean θ-S)."
    ),
    annotations=WRITES,
    tags={"ocean", "water masses", "profiles"},
)
@ocean_tool("water_masses")
def water_masses(
    profiles: Annotated[str, Field(description="Profile table (.parquet or .csv).")],
    definition: Annotated[
        str | dict[str, Any],
        Field(description="Built-in definition name, or an inline definition object."),
    ] = "atlantic_emery_meincke_1986",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    name, masses, citation = hydrography.parse_definition(definition)
    table, counts = hydrography.assign_water_masses(read_profiles(profiles), masses)
    cores = hydrography.water_mass_cores(table)
    out = _save_table(
        table,
        Path(profiles).stem + "-water-masses",
        "profiles",
        output_dir,
        tool="water_masses",
        sources=[local_file_source(Path(profiles))],
        method={
            "definition": name,
            "citation": citation,
            "masses": [m.__dict__ for m in masses],
            "rule": "inside the θ-S-pressure box; ties to the nearest box centre",
        },
        packages=TEOS,
    )
    named = [c["water_mass"] for c in cores]
    return result(
        f"Assigned water masses by '{name}': {', '.join(named) or 'none'}; "
        f"{counts.get('unassigned', 0)} samples outside every stated range.",
        [out],
        definition={
            "name": name,
            "citation": citation,
            "masses": [m.__dict__ for m in masses],
        },
        counts=counts,
        cores=cores,
    )


@mcp.tool(
    name="argo_fetch",
    title="Fetch Argo",
    description=(
        "Fetch Argo profiles with argopy into a profile table: a region box "
        "[lon_min, lon_max, lat_min, lat_max, p_min, p_max, 'YYYY-MM-DD', 'YYYY-MM-DD'] or float "
        "WMO numbers; core (phy) or BGC (oxygen, chlorophyll, nitrate). 'standard' mode keeps only "
        "good adjusted data. Large boxes over many years take minutes; fetch by year if needed."
    ),
    annotations=FETCHES,
    tags={"ocean", "argo", "profiles", "fetch"},
)
@ocean_tool("argo_fetch")
def argo_fetch(
    box: Annotated[
        list[Any] | None, Field(description="Region box with dates (8 values).")
    ] = None,
    floats: Annotated[
        list[int] | None, Field(description="Float WMO numbers instead of a box.")
    ] = None,
    dataset: Annotated[
        Literal["phy", "bgc"], Field(description="Core (phy) or BGC parameters.")
    ] = "phy",
    mode: Annotated[
        Literal["standard", "research", "expert"],
        Field(description="argopy user mode (QC)."),
    ] = "standard",
    source: Annotated[
        Literal["erddap", "gdac", "argovis"], Field(description="argopy data source.")
    ] = "erddap",
    params: Annotated[
        list[str] | None, Field(description="BGC parameters, e.g. ['DOXY'].")
    ] = None,
    name: Annotated[str, Field(description="Name for the output table.")] = "argo",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table, src = fetch.argo_fetch(
        box=box, floats=floats, dataset=dataset, mode=mode, source=source, params=params
    )
    out = _save_table(
        table,
        name,
        "profiles",
        output_dir,
        tool="argo_fetch",
        sources=[src],
        method={"mode": mode, "dataset": dataset},
        packages=("argopy",),
    )
    stats: dict[str, Any] = profile_summary(table)
    return result(
        f"Fetched {stats['profiles']} Argo profiles ({dataset}, {mode} mode) from {stats['time_range'][0]} to {stats['time_range'][1]}.",
        [out],
        **stats,
        citation="Argo (2000). Argo float data and metadata from Global Data Assembly Centre (Argo GDAC). SEANOE. doi:10.17882/42182",
    )


def _save_positions(
    table: pd.DataFrame, dataset_id: str, output_dir: str | None, src: Source
) -> Output:
    """One row per profile (where and when it was taken, and its transect), on a map."""

    columns = ["profile", "time", "latitude", "longitude", "transect"]
    positions = (
        table[[c for c in columns if c in table]]
        .groupby("profile", as_index=False, sort=False)
        .first()
    )
    if "transect" in positions:
        positions["leg"] = [
            f"transect {int(t)}" if t >= 0 else "between transects"
            for t in positions["transect"]
        ]
    path = output_path(f"{dataset_id}_positions", ".parquet", output_dir)
    positions.to_parquet(path, index=False)
    sidecar = write_provenance(
        path,
        tool="glider_fetch",
        sources=[src],
        method={"positions": "first sample of each profile"},
        packages=("erddapy",),
    )
    view = views.positions_view(
        path, positions, dataset_id, title=f"{dataset_id} track"
    )
    return Output(
        path, "positions", media_type(path), sidecar, rows=len(positions), view=view
    )


@mcp.tool(
    name="glider_fetch",
    title="Fetch Glider",
    description=(
        "Find glider deployments in the IOOS Glider DAC (give box and dates), or fetch one "
        "deployment (give dataset_id): its profiles as a table split into transects by heading "
        "reversals, with suspect samples flagged (provider QC, spikes, density inversions)."
    ),
    annotations=FETCHES,
    tags={"ocean", "glider", "profiles", "fetch"},
)
@ocean_tool("glider_fetch")
def glider_fetch(
    dataset_id: Annotated[
        str | None, Field(description="Glider DAC dataset id to fetch.")
    ] = None,
    box: Annotated[
        list[float] | None,
        Field(description="[lon_min, lon_max, lat_min, lat_max] to search."),
    ] = None,
    start: Annotated[str | None, Field(description="Search start date.")] = None,
    end: Annotated[str | None, Field(description="Search end date.")] = None,
    turn_degrees: Annotated[
        float, Field(description="Heading change that starts a new transect.")
    ] = 120.0,
    min_profiles: Annotated[
        int, Field(description="Shortest transect kept, in profiles.")
    ] = 10,
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if dataset_id is None:
        if not (box and start and end):
            raise OceanToolError(
                "bad_input", "give dataset_id, or box with start and end to search"
            )
        found = fetch.glider_search(box, start, end)
        return result(
            f"Found {len(found)} glider deployments in the box.", [], deployments=found
        )
    table, src = fetch.glider_fetch(dataset_id)
    if "temperature" in table and "salinity" in table:
        table = hydrography.add_teos10(table)
    table = fetch.split_transects(table, turn_degrees, min_profiles)
    table, flagged = fetch.flag_suspect(table)
    out = _save_table(
        table,
        dataset_id,
        "profiles",
        output_dir,
        tool="glider_fetch",
        sources=[src],
        method={
            "turn_degrees": turn_degrees,
            "min_profiles": min_profiles,
            "flags": "provider QC not 1/2; |x - rolling median| > 6 MAD; sigma0 decrease > 0.03 kg m-3",
        },
        packages=("erddapy", "gsw"),
    )
    transects = sorted(int(t) for t in table["transect"].unique() if t >= 0)
    track = _save_positions(table, dataset_id, output_dir, src)
    if out.view is not None and track.view is not None:
        # A deployment is a track first: its profiles and where they were taken are one
        # selection, so picking a profile on either view picks it on the other.
        out.view = {**out.view, **views.profile_selection(dataset_id)}
    return result(
        f"Deployment {dataset_id}: {table['profile'].nunique()} profiles in {len(transects)} transects; "
        f"{int((table['suspect'] != '').sum())} samples flagged.",
        [out, track],
        transects=transects,
        flagged=flagged[:200],
        **profile_summary(table),
    )


@mcp.tool(
    name="profile_metrics",
    title="Profile Metrics",
    description=(
        "Per-profile mixed layer depth by a stated criterion (density or temperature threshold from a "
        "reference pressure, e.g. 0.03 kg m-3 from 10 dbar), mean N² over a pressure band, and "
        "optionally anomaly profiles on standard levels against a reference table with their vertical "
        "type. Returns one row per profile and, with a reference, the anomaly table."
    ),
    annotations=WRITES,
    tags={"ocean", "mixed layer", "stratification", "profiles"},
)
@ocean_tool("profile_metrics")
def profile_metrics(
    profiles: Annotated[str, Field(description="Profile table (.parquet or .csv).")],
    criterion: Annotated[
        Literal["density_threshold", "temperature_threshold"],
        Field(description="MLD criterion."),
    ] = "density_threshold",
    threshold: Annotated[float, Field(description="Threshold (kg m-3 or °C).")] = 0.03,
    reference_pressure: Annotated[
        float, Field(description="Reference pressure, dbar.")
    ] = 10.0,
    stratification_band: Annotated[
        list[float], Field(description="[p_lo, p_hi] for mean N², dbar.")
    ] = [20.0, 200.0],  # noqa: B006
    anomaly_reference: Annotated[
        str | None, Field(description="Reference profile table for anomalies.")
    ] = None,
    anomaly_variable: Annotated[
        str, Field(description="Variable for anomalies.")
    ] = "CT",
    levels: Annotated[
        list[float] | None, Field(description="Standard pressures for anomalies.")
    ] = None,
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table = hydrography.add_teos10(read_profiles(profiles))
    metrics = hydrography.profile_metrics(
        table,
        criterion=criterion,
        threshold=threshold,
        reference_pressure=reference_pressure,
        stratification_range=(stratification_band[0], stratification_band[1]),
    )
    method = {
        "mld": {
            "criterion": criterion,
            "threshold": threshold,
            "reference_pressure_dbar": reference_pressure,
        },
        "n2_band_dbar": stratification_band,
    }
    sources = [local_file_source(Path(profiles))]
    outputs = [
        _save_table(
            metrics,
            Path(profiles).stem + "-metrics",
            "profile-metrics",
            output_dir,
            tool="profile_metrics",
            sources=sources,
            method=method,
            packages=TEOS,
        )
    ]
    numbers: dict[str, Any] = {
        "profiles": len(metrics),
        "mld_median_dbar": float(metrics["mld_dbar"].median()),
        "mld_range_dbar": [
            float(metrics["mld_dbar"].min()),
            float(metrics["mld_dbar"].max()),
        ],
        "mld_undetermined": int(metrics["mld_dbar"].isna().sum()),
    }
    if anomaly_reference:
        ref = hydrography.add_teos10(read_profiles(anomaly_reference))
        grid = np.asarray(
            levels
            or [
                5,
                10,
                20,
                30,
                50,
                75,
                100,
                150,
                200,
                300,
                400,
                500,
                600,
                800,
                1000,
                1200,
                1500,
                2000,
            ],
            float,
        )
        anomaly = hydrography.anomaly_profiles(table, ref, [anomaly_variable], grid)
        kinds = hydrography.classify_vertical_structure(anomaly, anomaly_variable)
        anomaly = anomaly.merge(
            kinds[["profile", "vertical_type"]], on="profile", how="left"
        )
        method["anomaly"] = {
            "variable": anomaly_variable,
            "reference": anomaly_reference,
            "levels_dbar": grid.tolist(),
            "vertical_type_rule": "column-wide if >80% of levels warm; else surface-intensified if peak <= 50 dbar; else subsurface",
        }
        sources.append(local_file_source(Path(anomaly_reference)))
        outputs.append(
            _save_table(
                anomaly,
                Path(profiles).stem + "-anomaly",
                "profiles",
                output_dir,
                tool="profile_metrics",
                sources=sources,
                method=method,
                packages=TEOS,
            )
        )
        numbers["vertical_types"] = kinds["vertical_type"].value_counts().to_dict()
        col = f"{anomaly_variable}_anomaly"
        layer = anomaly.groupby("pressure")[col].mean()
        numbers["mean_anomaly_by_level"] = {
            str(int(k)): float(v) for k, v in layer.items() if np.isfinite(v)
        }
    return result(
        f"Profile metrics for {numbers['profiles']} profiles: median MLD {numbers['mld_median_dbar']:.0f} dbar ({criterion}, {threshold} from {reference_pressure:g} dbar).",
        outputs,
        **numbers,
        method=method,
    )


@mcp.tool(
    name="section",
    title="Build Section",
    description=(
        "Build a section (field along a path against depth or pressure) as NetCDF: from a profile table "
        "(casts ordered by time, latitude, longitude or a given order, on pressure levels) or from a "
        "gridded file sampled along a path of [lon, lat] vertices. The path is stored with the section "
        "so a map can draw it and a redrawn path can recompute it."
    ),
    annotations=WRITES,
    tags={"ocean", "section"},
)
@ocean_tool("section")
def section(
    variables: Annotated[
        list[str], Field(description="Variables to put on the section.")
    ],
    profiles: Annotated[
        str | None, Field(description="Profile table to section.")
    ] = None,
    grid: Annotated[
        str | None, Field(description="Gridded NetCDF/Zarr (or glob) to sample.")
    ] = None,
    path: Annotated[
        list[list[float]] | None,
        Field(description="[[lon, lat], ...] for a gridded section."),
    ] = None,
    order: Annotated[
        Literal["time", "latitude", "longitude", "given"],
        Field(description="Cast order."),
    ] = "time",
    profile_order: Annotated[
        list[str] | None, Field(description="Cast ids, with order 'given'.")
    ] = None,
    profile_filter: Annotated[
        dict[str, Any] | None,
        Field(description="Column -> value to keep, e.g. {'transect': 3}."),
    ] = None,
    pressure_step: Annotated[
        float, Field(description="Vertical step for profile sections, dbar.")
    ] = 10.0,
    horizontal_step_km: Annotated[
        float | None, Field(description="Regular spacing between casts, km.")
    ] = None,
    step_km: Annotated[
        float, Field(description="Sampling step along a gridded path, km.")
    ] = 10.0,
    time: Annotated[
        str | None, Field(description="Time to take from a gridded field (nearest).")
    ] = None,
    name: Annotated[str, Field(description="Name for the output.")] = "section",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if profiles:
        table = read_profiles(profiles)
        for column, value in (profile_filter or {}).items():
            table = table[table[column] == value]
        if (
            any(v in ("SA", "CT", "sigma0", "pt0") for v in variables)
            and "SA" not in table
        ):
            table = hydrography.add_teos10(table)
        ds = sections.section_from_profiles(
            table,
            variables,
            order=order,
            profile_order=profile_order,
            pressure_step=pressure_step,
            horizontal_step_km=horizontal_step_km,
        )
        src = [local_file_source(Path(profiles))]
        method: dict[str, Any] = {
            "source": "profiles",
            "order": order,
            "pressure_step_dbar": pressure_step,
            "horizontal_step_km": horizontal_step_km,
            "filter": profile_filter,
            "interpolation": "linear in pressure, no extrapolation",
        }
    elif grid and path:
        ds = sections.section_from_grid(
            open_grid(grid, variables), variables, path, step_km=step_km, time=time
        )
        src = [Source(name=Path(grid).name, kind="file", reference=grid)]
        method = {
            "source": "grid",
            "path": path,
            "step_km": step_km,
            "time": time,
            "interpolation": "bilinear in lon-lat",
        }
    else:
        raise OceanToolError("bad_input", "give profiles, or grid with a path")
    out = _save_grid(
        ds,
        name,
        "section",
        output_dir,
        tool="section",
        sources=src,
        method=method,
        packages=("xarray", "gsw"),
    )
    summary = sections.section_summary(ds)
    return result(
        f"Section '{name}': {summary['columns']} columns over {summary['length_km']:.0f} km, variables {', '.join(summary['variables'])}.",
        [out],
        **summary,
        path=json.loads(ds.attrs["path"]),
    )
