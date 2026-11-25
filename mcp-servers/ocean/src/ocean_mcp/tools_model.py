"""Tools on model runs: spin-up drift and model against observations."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import Field

from .app import WRITES, mcp, ocean_tool, open_grid
from .implementation import model_checks
from .implementation.common import Source, local_file_source, result
from .implementation.profiles_io import read_profiles
from .tools_profiles import _save_table


@mcp.tool(
    name="drift",
    title="Model Drift",
    description=(
        "Spin-up check of a model run: area-weighted domain means of temperature and salinity and mean "
        "kinetic energy per level and time, and per level whether the drift over the last window is "
        "below a stated tolerance (equilibrated) or not."
    ),
    annotations=WRITES,
    tags={"ocean", "model", "spin-up"},
)
@ocean_tool("drift")
def drift(
    grid: Annotated[
        str,
        Field(description="Model output NetCDF/Zarr (or glob of files over the run)."),
    ],
    temperature: Annotated[str, Field(description="Temperature variable.")] = "thetao",
    salinity: Annotated[str | None, Field(description="Salinity variable.")] = "so",
    u: Annotated[
        str | None, Field(description="Eastward velocity, for kinetic energy.")
    ] = None,
    v: Annotated[
        str | None, Field(description="Northward velocity, for kinetic energy.")
    ] = None,
    window: Annotated[
        int, Field(description="Time steps at the end of the run tested for drift.")
    ] = 12,
    tolerance: Annotated[
        dict[str, float],
        Field(
            description="Allowed change over the window per quantity, e.g. {'thetao': 0.01, 'so': 0.002, 'kinetic_energy': 1e-4}."
        ),
    ] = {"default": 0.01},  # noqa: B006
    name: Annotated[str, Field(description="Name for the outputs.")] = "drift",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(grid, [temperature] + [x for x in (salinity, u, v) if x])
    series, verdicts = model_checks.drift(
        ds,
        temperature=temperature,
        salinity=salinity,
        u=u,
        v=v,
        window=window,
        tolerance=tolerance,
    )
    prov = {
        "tool": "drift",
        "sources": [Source(name=Path(grid).name, kind="file", reference=grid)],
        "method": {
            "window_steps": window,
            "tolerance": tolerance,
            "rule": "|linear slope| x window length < tolerance",
        },
        "packages": ("xarray",),
    }
    outputs = [
        _save_table(series, f"{name}-series", "series", output_dir, **prov),
        _save_table(verdicts, f"{name}-verdicts", "table", output_dir, **prov),
    ]
    drifting = verdicts[~verdicts["equilibrated"]] if len(verdicts) else verdicts
    return result(
        f"{len(verdicts) - len(drifting)} of {len(verdicts)} level-quantity pairs within tolerance over the last {window} steps"
        + (
            f"; still drifting: {', '.join(sorted({str(q) for q in drifting['quantity']}))}"
            if len(drifting)
            else "; the run looks equilibrated by this test"
        )
        + ".",
        outputs,
        equilibrated=bool(len(drifting) == 0),
        drifting=drifting.head(40).to_dict("records"),
    )


@mcp.tool(
    name="colocate",
    title="Model vs Observations",
    description=(
        "Sample a model at each observation's position, depth and time (profiles from Argo, CTD or "
        "moorings) and report bias and RMSE overall, by depth band and by region cell; writes the matched "
        "pairs (for a profile stack of model against observation) and the statistics."
    ),
    annotations=WRITES,
    tags={"ocean", "model", "evaluation"},
)
@ocean_tool("colocate")
def colocate(
    grid: Annotated[str, Field(description="Model output NetCDF/Zarr (or glob).")],
    observations: Annotated[str, Field(description="Profile table of observations.")],
    model_variable: Annotated[str, Field(description="Model variable (e.g. thetao).")],
    observed_variable: Annotated[
        str, Field(description="Observed column (e.g. temperature).")
    ],
    depth_bands: Annotated[
        list[float] | None, Field(description="Band edges, m.")
    ] = None,
    region_step_degrees: Annotated[
        float, Field(description="Region cell size, degrees.")
    ] = 5.0,
    max_time_offset_days: Annotated[
        float, Field(description="Largest allowed obs-model time gap, days.")
    ] = 15.0,
    name: Annotated[str, Field(description="Name for the outputs.")] = "colocate",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if depth_bands is None:
        depth_bands = [0, 50, 200, 500, 1000, 2000]
    ds = open_grid(grid, [model_variable])
    obs = read_profiles(observations)
    pairs, stats, numbers = model_checks.colocate(
        ds,
        obs,
        model_variable=model_variable,
        observed_variable=observed_variable,
        depth_bands=depth_bands,
        region_step_degrees=region_step_degrees,
        max_time_offset_days=max_time_offset_days,
    )
    prov = {
        "tool": "colocate",
        "sources": [
            Source(name=Path(grid).name, kind="file", reference=grid),
            local_file_source(Path(observations)),
        ],
        "method": {
            "interpolation": "linear in lon, lat, depth, time",
            "depth_bands_m": depth_bands,
            "region_step_degrees": region_step_degrees,
            "max_time_offset_days": max_time_offset_days,
        },
        "packages": ("xarray",),
    }
    outputs = [
        _save_table(pairs, f"{name}-pairs", "profiles", output_dir, **prov),
        _save_table(stats, f"{name}-stats", "table", output_dir, **prov),
    ]
    return result(
        f"{numbers['matched']} observations matched: bias {numbers['overall_bias']:+.3f}, RMSE {numbers['overall_rmse']:.3f} ({numbers['dropped_outside_time']} outside the run's time).",
        outputs,
        **numbers,
    )


@mcp.tool(
    name="mom6_preflight",
    title="MOM6 Pre-flight",
    description=(
        "Check a regional MOM6 run directory before running: open-boundary segments in MOM_input match "
        "the segment files; boundary and initial files cover the run dates in input.nml; the land mask "
        "has no isolated wet cells or one-cell channels (except cells the scientist kept); MINIMUM_DEPTH "
        "agrees with the bathymetry; LAYOUT fits the mask table. Each check passes, fails or says why it "
        "could not run."
    ),
    annotations=WRITES,
    tags={"ocean", "model", "MOM6"},
)
@ocean_tool("mom6_preflight")
def mom6_preflight(
    run_dir: Annotated[
        str,
        Field(description="Run directory holding MOM_input, MOM_override, input.nml."),
    ],
    input_dir: Annotated[
        str | None, Field(description="INPUT directory if not run_dir/INPUT.")
    ] = None,
    kept_cells: Annotated[
        list[list[int]],
        Field(description="[[j, i], ...] narrow cells the scientist chose to keep."),
    ] = [],  # noqa: B006
    name: Annotated[str, Field(description="Name for the report.")] = "mom6-preflight",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    import pandas as pd

    from .implementation import mom6

    checks, facts = mom6.preflight(
        Path(run_dir),
        input_dir=Path(input_dir) if input_dir else None,
        kept_cells=kept_cells,
    )
    rows = [c.as_dict() for c in checks]
    problems = [
        {"check": c.name, **(i if isinstance(i, dict) else {"issue": i})}
        for c in checks
        for i in c.items
    ]
    table = pd.DataFrame(
        problems or [{"check": "none", "issue": "no problems"}]
    ).astype(str)
    out = _save_table(
        table,
        name,
        "table",
        output_dir,
        tool="mom6_preflight",
        sources=[Source(name=Path(run_dir).name, kind="file", reference=run_dir)],
        method={
            "parameters": "MOM6 names confirmed in mom-ocean/MOM6 0e9a561",
            "kept_cells": kept_cells,
        },
        packages=("xarray",),
    )
    failed = [c.name for c in checks if c.status == "failed"]
    skipped = [c.name for c in checks if c.status == "not_checked"]
    summary = (
        f"Pre-flight: {len(checks) - len(failed) - len(skipped)} passed, {len(failed)} failed"
        + (f" ({', '.join(failed)})" if failed else "")
        + (f", {len(skipped)} not checked ({', '.join(skipped)})" if skipped else "")
        + "."
    )
    payload = result(summary, [out], checks=rows, **facts)
    if failed:
        payload["status"] = "partial"
        payload["outcome"] = "checks_failed"
    return payload
