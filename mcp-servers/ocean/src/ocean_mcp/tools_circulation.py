"""Tools on circulation: transport, overturning, tides, spectra, eddies, particles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
from pydantic import Field

from .app import WRITES, mcp, ocean_tool, open_grid
from .implementation import (
    particles as particles_impl,
)
from .implementation import (
    timeseries,
)
from .implementation import (
    transport as transport_impl,
)
from .implementation.common import (
    OceanToolError,
    Source,
    local_file_source,
    result,
)
from .tools_profiles import _save_grid, _save_table


def _grid_source(path: str, product: str | None, licence: str | None) -> Source:
    return Source(
        name=product or Path(path).name,
        kind="file",
        reference=path,
        licence=licence or "",
    )


@mcp.tool(
    name="transport",
    title="Section Transport",
    description=(
        "Volume (Sv) and heat (PW, relative to a stated reference temperature) transport normal to a "
        "path through gridded u, v (and temperature) on a regular lon-lat product, per time step. "
        "Positive is to the right of the path's direction. The result states that a drawn path on a "
        "regular grid gives an approximate transport."
    ),
    annotations=WRITES,
    tags={"ocean", "transport", "circulation"},
)
@ocean_tool("transport")
def transport(
    grid: Annotated[
        str, Field(description="Gridded NetCDF/Zarr (or glob) with u, v, depth.")
    ],
    path: Annotated[
        list[list[float]], Field(description="[[lon, lat], ...] across the passage.")
    ],
    u: Annotated[str, Field(description="Eastward velocity variable.")] = "uo",
    v: Annotated[str, Field(description="Northward velocity variable.")] = "vo",
    temperature: Annotated[
        str | None, Field(description="Temperature variable for heat transport.")
    ] = None,
    reference_temperature: Annotated[
        float, Field(description="Reference temperature for heat transport, °C.")
    ] = 0.0,
    step_km: Annotated[
        float, Field(description="Sampling step along the path, km.")
    ] = 5.0,
    product: Annotated[
        str | None, Field(description="Product name/DOI for provenance.")
    ] = None,
    licence: Annotated[
        str | None, Field(description="Data licence of the product.")
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "transport",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(grid, [u, v] + ([temperature] if temperature else []))
    out, numbers = transport_impl.section_transport(
        ds,
        path,
        u=u,
        v=v,
        temperature=temperature,
        step_km=step_km,
        reference_temperature=reference_temperature,
    )
    out.attrs["path"] = json.dumps({"type": "LineString", "coordinates": path})
    velocity = out.get("normal_velocity")
    vertical = next(
        (
            d
            for d in ("depth", "lev", "level", "z", "pressure")
            if velocity is not None and d in velocity.dims
        ),
        None,
    )
    if velocity is not None and vertical and velocity.ndim > 2:
        # The mean section, for the section view (it draws distance against depth).
        others = [d for d in velocity.dims if d not in ("distance", vertical)]
        out["normal_velocity_mean"] = velocity.mean(others, keep_attrs=True)
    saved = _save_grid(
        out,
        name,
        "transport",
        output_dir,
        tool="transport",
        sources=[_grid_source(grid, product, licence)],
        method={
            "path": path,
            "step_km": step_km,
            "sign": "positive to the right of the path direction",
            "reference_temperature_degC": reference_temperature,
            "rho0": 1025.0,
            "cp": 3992.0,
            "approximation": numbers["approximation"],
        },
        packages=("xarray",),
        notes=[numbers["approximation"]],
    )
    return result(
        f"Mean volume transport {numbers['mean_volume_transport_sv']:.1f} Sv through the path"
        + (f", heat {numbers['mean_heat_transport_pw']:.3f} PW" if temperature else "")
        + " (approximate: drawn path on a regular grid).",
        [saved],
        **numbers,
    )


@mcp.tool(
    name="overturning",
    title="Overturning",
    description=(
        "Meridional overturning streamfunction (Sv) in depth or density space, on a grid whose rows "
        "are latitude lines (regular or MOM6-style), from a per-cell meridional transport (vmo kg/s "
        "or vh m3/s) or a velocity v, within a basin_mask or the whole file; plus the time series of its maximum "
        "at a chosen latitude (e.g. 26.5 N). A curvilinear grid is refused with the reason."
    ),
    annotations=WRITES,
    tags={"ocean", "overturning", "AMOC", "model"},
)
@ocean_tool("overturning")
def overturning(
    grid: Annotated[str, Field(description="Model output NetCDF/Zarr (or glob).")],
    transport_variable: Annotated[
        str | None, Field(description="Per-cell meridional transport (vmo/vh).")
    ] = None,
    velocity: Annotated[
        str | None, Field(description="Meridional velocity, if no transport variable.")
    ] = None,
    space: Annotated[
        Literal["depth", "density"], Field(description="Vertical coordinate of psi.")
    ] = "depth",
    density: Annotated[
        str | None,
        Field(description="Density variable for density space (e.g. sigma2)."),
    ] = None,
    density_bins: Annotated[
        list[float] | None, Field(description="Density class edges.")
    ] = None,
    latitude: Annotated[
        float, Field(description="Latitude for the time series.")
    ] = 26.5,
    basin_mask: Annotated[
        str | None,
        Field(description="Variable (1 inside basin) to restrict the zonal sum."),
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "overturning",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(grid)
    if basin_mask:
        mask = ds[basin_mask]
        ds = ds.where(mask > 0)
    psi = transport_impl.overturning(
        ds,
        transport=transport_variable,
        velocity=velocity,
        space=space,
        density=density,
        density_bins=np.asarray(density_bins, float) if density_bins else None,
    )
    series, numbers = transport_impl.overturning_at(psi, latitude)
    psi["psi_max_at_latitude"] = series
    saved = _save_grid(
        psi,
        name,
        "section",
        output_dir,
        tool="overturning",
        sources=[
            local_file_source(Path(grid))
            if Path(grid).is_file()
            else Source(name=grid, kind="file", reference=grid)
        ],
        method={
            "space": space,
            "variable": transport_variable or velocity,
            "integration": "zonal sum, then cumulative from the surface (depth) or lightest class (density)",
            "basin_mask": basin_mask,
            "latitude": latitude,
        },
        packages=("xarray",),
    )
    return result(
        f"Overturning in {space} space: maximum at {numbers['latitude_used']:.2f}° averages {numbers['mean_max_sv']:.1f} Sv.",
        [saved],
        **numbers,
    )


@mcp.tool(
    name="harmonics",
    title="Tidal Harmonics",
    description=(
        "Tidal harmonic analysis with UTide of a sea level (height) or current (u, v) series: "
        "constituent amplitudes and phases (or ellipse parameters for currents) with confidence and "
        "SNR, the predicted and residual series, and optionally a comparison against another "
        "constituent table (a tide model at the same point)."
    ),
    annotations=WRITES,
    tags={"ocean", "tides", "harmonics", "time series"},
)
@ocean_tool("harmonics")
def harmonics(
    series: Annotated[
        str, Field(description="Series file (.csv, .parquet, .nc) with a time column.")
    ],
    latitude: Annotated[float, Field(description="Latitude of the gauge or mooring.")],
    height: Annotated[str | None, Field(description="Sea level column.")] = None,
    u: Annotated[str | None, Field(description="Eastward current column.")] = None,
    v: Annotated[str | None, Field(description="Northward current column.")] = None,
    constituents: Annotated[
        list[str] | str,
        Field(description="'auto' or a list such as ['M2','S2','K1','O1']."),
    ] = "auto",
    method: Annotated[
        Literal["ols", "robust"], Field(description="UTide fit method.")
    ] = "ols",
    compare_with: Annotated[
        str | None,
        Field(description="Constituent table (.csv/.parquet) to compare against."),
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "tides",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    columns = [c for c in (height, u, v) if c]
    data = timeseries.read_series(series, columns)
    table, recon, numbers = timeseries.harmonic_analysis(
        data,
        height=height,
        u=u,
        v=v,
        latitude=latitude,
        constituents=constituents,
        method=method,
    )
    prov = {
        "tool": "harmonics",
        "sources": [local_file_source(Path(series))],
        "method": {
            "latitude": latitude,
            "constituents": constituents,
            "method": method,
            "trend": False,
        },
        "packages": ("utide",),
    }
    outputs = [
        _save_table(table, f"{name}-constituents", "table", output_dir, **prov),
        _save_table(recon, f"{name}-series", "series", output_dir, **prov),
    ]
    if compare_with:
        other = (
            pd.read_parquet(compare_with)
            if compare_with.endswith(".parquet")
            else pd.read_csv(compare_with)
        )
        diff = timeseries.compare_constituents(
            table, other, list(table["constituent"].head(8))
        )
        outputs.append(
            _save_table(diff, f"{name}-comparison", "table", output_dir, **prov)
        )
        numbers["comparison"] = diff.to_dict("records")
    return result(
        f"Harmonic analysis: {numbers['significant']} significant constituents (SNR >= 2), dominant "
        f"{', '.join(numbers['dominant'][:4])}; tides explain {numbers['variance_explained']:.0%} of the variance.",
        outputs,
        **numbers,
        constituents=table.head(12).to_dict("records"),
    )


@mcp.tool(
    name="spectra",
    title="Spectra",
    description=(
        "Power spectrum (Welch) of a series, or rotary spectra (clockwise and anticlockwise) of a "
        "current u + iv; frequencies in cycles per day, with the inertial frequency for a latitude."
    ),
    annotations=WRITES,
    tags={"ocean", "spectra", "time series"},
)
@ocean_tool("spectra")
def spectra(
    series: Annotated[str, Field(description="Series file with a time column.")],
    u: Annotated[str, Field(description="Column (or eastward current).")],
    v: Annotated[
        str | None, Field(description="Northward current for rotary spectra.")
    ] = None,
    segment_days: Annotated[
        float, Field(description="Welch segment length, days.")
    ] = 16.0,
    latitude: Annotated[
        float | None, Field(description="Latitude, for the inertial frequency.")
    ] = None,
    name: Annotated[str, Field(description="Name for the output.")] = "spectra",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    data = timeseries.read_series(series, [c for c in (u, v) if c])
    table, numbers = timeseries.spectra(
        data, u=u, v=v, segment_days=segment_days, latitude=latitude
    )
    out = _save_table(
        table,
        name,
        "series",
        output_dir,
        tool="spectra",
        sources=[local_file_source(Path(series))],
        method={
            "welch_segment_days": segment_days,
            "detrend": "linear",
            "rotary": v is not None,
        },
        packages=("scipy",),
    )
    return result(
        f"Spectra of {len(data)} samples; peaks at {numbers['peak_frequency_cpd']} cycles/day.",
        [out],
        **numbers,
    )


@mcp.tool(
    name="sea_level_decompose",
    title="Sea Level Trends",
    description=(
        "Trends of a tide gauge record and the nearest altimetry series over their common period "
        "(annual cycle removed, autocorrelation-corrected errors) and of their difference, the local "
        "part, which includes vertical land motion."
    ),
    annotations=WRITES,
    tags={"ocean", "sea level", "trend"},
)
@ocean_tool("sea_level_decompose")
def sea_level_decompose(
    gauge: Annotated[str, Field(description="Gauge series file (time + column).")],
    gauge_column: Annotated[str, Field(description="Gauge sea level column.")],
    altimetry: Annotated[
        str, Field(description="Altimetry series file (time + column).")
    ],
    altimetry_column: Annotated[str, Field(description="Altimetry sea level column.")],
    units_scale: Annotated[
        float,
        Field(
            description="Multiply altimetry by this to match gauge units (e.g. 1000 for m to mm)."
        ),
    ] = 1.0,
    name: Annotated[str, Field(description="Name for the output.")] = "sea-level",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    g = timeseries.read_series(gauge, [gauge_column])[gauge_column]
    a = (
        timeseries.read_series(altimetry, [altimetry_column])[altimetry_column]
        * units_scale
    )
    table, numbers = timeseries.sea_level_decompose(g, a)
    out = _save_table(
        table,
        name,
        "series",
        output_dir,
        tool="sea_level_decompose",
        sources=[local_file_source(Path(gauge)), local_file_source(Path(altimetry))],
        method={
            "resample": "monthly means",
            "trend": "OLS with annual harmonic, lag-1 inflated errors",
            "units_scale": units_scale,
        },
        packages=("scipy",),
    )
    lt = numbers["local_trend"]
    return result(
        f"Gauge {numbers['gauge_trend']['trend_per_year']:.2f}/yr, altimetry {numbers['altimetry_trend']['trend_per_year']:.2f}/yr; "
        f"local part {lt['trend_per_year']:.2f} ± {lt['stderr_per_year']:.2f}/yr over {numbers['common_period'][0][:7]} to {numbers['common_period'][1][:7]}.",
        [out],
        **numbers,
    )


@mcp.tool(
    name="particles",
    title="Particle Tracking",
    description=(
        "Lagrangian particles with Parcels 3 (RK4) in a gridded velocity field, seeded on a disc around "
        "a station and run forward (where water goes) or backward (where it came from). Writes the "
        "trajectories and a density grid of end points (destinations or origins)."
    ),
    annotations=WRITES,
    tags={"ocean", "lagrangian", "particles"},
)
@ocean_tool("particles")
def particles(
    grid: Annotated[
        str, Field(description="Velocity NetCDF/Zarr (or glob) with u, v, time.")
    ],
    station: Annotated[list[float], Field(description="[lon, lat] release point.")],
    days: Annotated[float, Field(description="Integration length, days.")],
    direction: Annotated[
        Literal["forward", "backward"],
        Field(description="Forward or backward in time."),
    ] = "backward",
    u: Annotated[str, Field(description="Eastward velocity variable.")] = "uo",
    v: Annotated[str, Field(description="Northward velocity variable.")] = "vo",
    depth: Annotated[
        float | None,
        Field(description="Depth level to use (nearest); default the shallowest."),
    ] = None,
    count: Annotated[int, Field(description="Particles to release.")] = 500,
    radius_km: Annotated[float, Field(description="Release disc radius, km.")] = 10.0,
    start: Annotated[
        str | None,
        Field(
            description="Release time (default end for backward, start for forward)."
        ),
    ] = None,
    dt_minutes: Annotated[
        float, Field(description="Integration step, minutes.")
    ] = 30.0,
    output_hours: Annotated[float, Field(description="Output interval, hours.")] = 24.0,
    diffusivity_m2s: Annotated[
        float,
        Field(
            description=(
                "Horizontal diffusivity for motion the grid does not resolve, m2/s "
                "(0: resolved flow only, and the spread is not an uncertainty)."
            ),
            ge=0,
        ),
    ] = 0.0,
    density_step_degrees: Annotated[
        float, Field(description="Cell size of the end-point density grid.")
    ] = 0.25,
    seed: Annotated[
        int, Field(description="Random seed for the release positions.")
    ] = 0,
    product: Annotated[
        str | None, Field(description="Product name/DOI for provenance.")
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "particles",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if count > 20000:
        raise OceanToolError("bad_input", "at most 20000 particles per run")
    ds = open_grid(grid, [u, v])
    table, numbers = particles_impl.run_particles(
        ds,
        u=u,
        v=v,
        depth=depth,
        station=(station[0], station[1]),
        radius_km=radius_km,
        count=count,
        days=days,
        direction=direction,
        start=start,
        dt_minutes=dt_minutes,
        output_hours=output_hours,
        seed=seed,
        diffusivity_m2s=diffusivity_m2s,
    )
    density = particles_impl.end_density(table, direction, density_step_degrees)
    prov = {
        "tool": "particles",
        "sources": [_grid_source(grid, product, None)],
        "method": {
            **numbers,
            "radius_km": radius_km,
            "dt_minutes": dt_minutes,
            "seed": seed,
            "depth": depth,
        },
        "packages": ("parcels",),
    }
    outputs = [
        _save_table(table, f"{name}-trajectories", "tracks", output_dir, **prov),
        _save_grid(
            density, f"{name}-{density.attrs['meaning']}", "grid", output_dir, **prov
        ),
    ]
    return result(
        f"{numbers['kept_to_end']} of {count} particles ran {days:g} days {direction}; "
        f"median displacement {numbers['median_displacement_km']:.0f} km; end cloud "
        f"{numbers['end_spread_km']:.0f} km across (median distance from its centre), "
        f"from {numbers['spread_meaning']}.",
        outputs,
        **numbers,
    )
