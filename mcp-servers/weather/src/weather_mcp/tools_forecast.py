"""Tools on fields and forecasts: fetch, exceedance, threshold events, sections, soundings."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from pydantic import Field

from .app import FETCHES, WRITES, mcp, weather_tool
from .implementation import analysis, atmosphere, fetch
from .implementation.common import Source, WeatherToolError, local_file_source, result
from .implementation.grids import area_mean, at_point, describe, open_grid, select_box
from .saving import save_grid, save_table


def _file_source(
    path: str, product: str | None = None, licence: str | None = None
) -> Source:
    p = Path(path).expanduser()
    if p.is_file():
        src = local_file_source(p, product)
        src.licence = licence or ""
        return src
    return Source(
        name=product or path, kind="file", reference=path, licence=licence or ""
    )


@mcp.tool(
    name="era5_fetch",
    title="Fetch ERA5",
    description=(
        "Subset ERA5 hourly fields (single and pressure levels) from the analysis-ready ARCO-ERA5 copy "
        "on Google Cloud, no key needed: variables by their long names (2m_temperature, "
        "mean_sea_level_pressure, geopotential, temperature, specific_humidity, u_component_of_wind, "
        "vertical_integral_of_eastward_water_vapour_flux, total_precipitation, ...), a box, a period, "
        "optional pressure levels and hours. Refuses subsets larger than max_values."
    ),
    annotations=FETCHES,
    tags={"weather", "era5", "reanalysis", "fetch"},
)
@weather_tool("era5_fetch")
def era5_fetch(
    variables: Annotated[list[str], Field(description="ERA5 long variable names.")],
    box: Annotated[
        list[float], Field(description="[lon_min, lon_max, lat_min, lat_max].")
    ],
    start: Annotated[str, Field(description="Start time (ISO).")],
    end: Annotated[str, Field(description="End time (ISO).")],
    levels: Annotated[
        list[int] | None, Field(description="Pressure levels, hPa.")
    ] = None,
    hours: Annotated[
        list[int] | None, Field(description="UTC hours to keep, e.g. [0, 6, 12, 18].")
    ] = None,
    max_values: Annotated[
        float, Field(description="Largest subset allowed, values.")
    ] = 2e8,
    name: Annotated[str, Field(description="Name for the output.")] = "era5",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds, src = fetch.era5_arco(
        variables=variables,
        box=box,
        start=start,
        end=end,
        levels=levels,
        hours=hours,
        max_values=max_values,
    )
    out = save_grid(
        ds,
        name,
        "grid",
        output_dir,
        tool="era5_fetch",
        sources=[src],
        method={"store": fetch.ARCO_ERA5},
        packages=("xarray", "gcsfs", "zarr"),
    )
    info = describe(ds)
    return result(
        f"ERA5 {', '.join(variables)} over {box} from {info['time_range'][0]} to {info['time_range'][1]} ({info['dims']}).",
        [out],
        **info,
        licence=src.licence,
    )


@mcp.tool(
    name="ensemble_fetch",
    title="Fetch Ensemble",
    description=(
        "ECMWF ensemble (control + 50 perturbed members) from ECMWF open data, no key: one parameter "
        "(tp, 2t, 10u, 10v, msl, ...) at the given forecast steps (hours), cut to a point or a box, "
        "from the latest run or a given date and run hour. Open data keeps only recent runs."
    ),
    annotations=FETCHES,
    tags={"weather", "ensemble", "ecmwf", "fetch"},
)
@weather_tool("ensemble_fetch")
def ensemble_fetch(
    param: Annotated[
        str, Field(description="Parameter short name (tp, 2t, msl, ...).")
    ],
    steps: Annotated[
        list[int], Field(description="Forecast steps, hours (e.g. 0 to 72 every 6).")
    ],
    point: Annotated[
        list[float] | None, Field(description="[lon, lat] (nearest grid point).")
    ] = None,
    box: Annotated[
        list[float] | None, Field(description="[lon_min, lon_max, lat_min, lat_max].")
    ] = None,
    date: Annotated[
        str | None, Field(description="Run date YYYYMMDD (default latest).")
    ] = None,
    time: Annotated[int | None, Field(description="Run hour (0, 6, 12, 18).")] = None,
    model: Annotated[
        Literal["ifs", "aifs-ens"], Field(description="IFS ENS or AIFS ENS.")
    ] = "ifs",
    name: Annotated[str, Field(description="Name for the output.")] = "ensemble",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if point is None and box is None:
        raise WeatherToolError("bad_input", "give a point or a box")
    ds, src = fetch.ensemble_open_data(
        param=param,
        steps=steps,
        box=box,
        point=point,
        date=date,
        time=time,
        model=model,
    )
    out = save_grid(
        ds,
        name,
        "grid",
        output_dir,
        tool="ensemble_fetch",
        sources=[src],
        method={"stream": "enfo", "types": ["cf", "pf"]},
        packages=("ecmwf-opendata", "cfgrib"),
    )
    return result(
        f"{src.name}, {src.version}: {param} at {len(steps)} steps.",
        [out],
        members=int(ds.sizes.get("number", 0)),
        run=src.version,
        licence=src.licence,
    )


@mcp.tool(
    name="exceedance",
    title="Exceedance Odds",
    description=(
        "Probabilities that an ensemble exceeds thresholds at a point, per window (e.g. daily 24 h "
        "precipitation from an accumulated field): member amounts, percentiles and the fraction of "
        "members above each threshold, plus the probability for the whole period. Writes the table "
        "and the per-member plume."
    ),
    annotations=WRITES,
    tags={"weather", "ensemble", "probability"},
)
@weather_tool("exceedance")
def exceedance(
    ensemble: Annotated[
        str, Field(description="Ensemble NetCDF at a point (member and step dims).")
    ],
    variable: Annotated[str, Field(description="Variable (e.g. tp).")],
    thresholds: Annotated[
        list[float], Field(description="Thresholds in output units (e.g. [10, 30] mm).")
    ],
    accumulated: Annotated[
        bool, Field(description="Variable is accumulated since the run start.")
    ] = True,
    scale: Annotated[
        float | None,
        Field(
            description="Unit factor to the thresholds' units; omit to use the field's own "
            "units (m to mm, mm and kg m-2 as they are)."
        ),
    ] = None,
    window_hours: Annotated[int, Field(description="Window length, hours.")] = 24,
    name: Annotated[str, Field(description="Name for the outputs.")] = "exceedance",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(ensemble, [variable])
    table, plume, numbers = analysis.exceedance(
        ds,
        variable=variable,
        thresholds=thresholds,
        accumulated=accumulated,
        scale=scale,
        window_hours=window_hours,
    )
    prov = {
        "tool": "exceedance",
        "sources": [_file_source(ensemble)],
        "method": {
            "thresholds": thresholds,
            "accumulated": accumulated,
            "scale": numbers["scale"],
            "units_in": numbers["units_in"],
            "window_hours": window_hours,
            "probability": "fraction of members above the threshold",
        },
        "packages": ("xarray",),
    }
    outputs = [
        save_table(table, name, "series", output_dir, **prov),
        save_table(plume, f"{name}-plume", "series", output_dir, **prov),
    ]
    head = ", ".join(
        f"P(>{t:g}) up to {numbers[f'max_probability_gt_{t:g}']:.0%}"
        for t in thresholds
    )
    return result(
        f"{numbers['members']} members, {numbers['windows']} windows of {window_hours} h: {head}.",
        outputs,
        **numbers,
        windows_table=table.astype({"window_start": str, "window_end": str}).to_dict(
            "records"
        ),
    )


@mcp.tool(
    name="threshold_events",
    title="Threshold Events",
    description=(
        "Events above a calendar-day percentile threshold in a series (for example heat waves: daily "
        "maximum 2 m temperature above the 90th percentile of a 1991-2020 baseline for at least 3 "
        "days): climatology, threshold and events with duration, peak and anomalies. The series can be "
        "a file or a gridded field at a point or averaged over a box; hourly data are reduced to daily."
    ),
    annotations=WRITES,
    tags={"weather", "climate", "extremes", "heat wave"},
)
@weather_tool("threshold_events")
def threshold_events(
    baseline: Annotated[
        list[int], Field(description="[first_year, last_year] of the baseline.")
    ],
    series: Annotated[
        str | None, Field(description="Series file (time + column).")
    ] = None,
    column: Annotated[str | None, Field(description="Value column.")] = None,
    grid: Annotated[
        str | None, Field(description="Gridded file instead of a series.")
    ] = None,
    variable: Annotated[str | None, Field(description="Gridded variable.")] = None,
    point: Annotated[
        list[float] | None, Field(description="[lon, lat] in the grid.")
    ] = None,
    box: Annotated[
        list[float] | None, Field(description="Box to average instead of a point.")
    ] = None,
    daily_statistic: Annotated[
        Literal["mean", "max", "min", "sum"],
        Field(description="How hours become days."),
    ] = "max",
    percentile: Annotated[float, Field(description="Threshold percentile.")] = 90.0,
    window_days: Annotated[
        int, Field(description="Calendar window for the percentile, days.")
    ] = 15,
    min_duration: Annotated[int, Field(description="Shortest event, days.")] = 3,
    max_gap: Annotated[int, Field(description="Longest gap joined, days.")] = 0,
    offset: Annotated[
        float, Field(description="Added to values (e.g. -273.15 for K to °C).")
    ] = 0.0,
    period: Annotated[
        list[str] | None, Field(description="[start, end] to report events in.")
    ] = None,
    name: Annotated[
        str, Field(description="Name for the outputs.")
    ] = "threshold-events",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if series and column:
        frame = (
            pd.read_parquet(series)
            if series.endswith(".parquet")
            else pd.read_csv(series)
        )
        time_col = next(
            c
            for c in frame.columns
            if c.lower() in ("time", "date", "valid_time", "datetime")
        )
        s = pd.Series(
            frame[column].to_numpy(), index=pd.to_datetime(frame[time_col], utc=True)
        )
        src = [_file_source(series)]
    elif grid and variable and (point or box):
        da = open_grid(grid, [variable])[variable]
        da = at_point(da, point) if point else area_mean(select_box(da, box or []))
        s = da.to_series()
        s.index = pd.to_datetime(s.index, utc=True)
        src = [_file_source(grid)]
    else:
        raise WeatherToolError(
            "bad_input", "give series and column, or grid, variable and a point or box"
        )
    daily = analysis.daily_series(s + offset, daily_statistic)
    clim = analysis.threshold_climatology(
        daily,
        baseline=(baseline[0], baseline[1]),
        percentile=percentile,
        window_days=window_days,
    )
    found = analysis.threshold_events(clim, min_duration=min_duration, max_gap=max_gap)
    if period and len(found):
        found = found[
            (found["end"] >= pd.Timestamp(period[0]))
            & (found["start"] <= pd.Timestamp(period[1]))
        ]
    method = {
        "daily_statistic": daily_statistic,
        "baseline": baseline,
        "percentile": percentile,
        "window_days": window_days,
        "min_duration_days": min_duration,
        "max_gap_days": max_gap,
        "offset": offset,
        "point": point,
        "box": box,
    }
    prov = {
        "tool": "threshold_events",
        "sources": src,
        "method": method,
        "packages": ("pandas",),
    }
    outputs = [
        save_table(clim.reset_index(), f"{name}-series", "series", output_dir, **prov),
        save_table(found, f"{name}-events", "events", output_dir, **prov),
    ]
    per_year = (
        found.groupby(found["start"].dt.year).size().to_dict() if len(found) else {}
    )
    longest = (
        found.sort_values("duration_days", ascending=False).head(5)
        if len(found)
        else found
    )
    return result(
        f"{len(found)} events above the {percentile:g}th percentile of daily {daily_statistic} for at least {min_duration} days (baseline {baseline[0]}-{baseline[1]}).",
        outputs,
        events=len(found),
        events_per_year={int(k): int(v) for k, v in per_year.items()},
        longest=longest.astype({"start": str, "end": str, "peak_day": str}).to_dict(
            "records"
        )
        if len(longest)
        else [],
        method=method,
    )


@mcp.tool(
    name="cross_section",
    title="Cross Section",
    description=(
        "Vertical cross-section of pressure-level fields along a path of [lon, lat] vertices (geodesic "
        "legs, MetPy) at one time, with potential and equivalent potential temperature and the wind "
        "along and across the path; NetCDF with the path stored for the map."
    ),
    annotations=WRITES,
    tags={"weather", "section", "front"},
)
@weather_tool("cross_section")
def cross_section(
    grid: Annotated[
        str,
        Field(
            description="Pressure-level NetCDF/GRIB with temperature (and humidity, winds)."
        ),
    ],
    path: Annotated[
        list[list[float]], Field(description="[[lon, lat], ...] vertices.")
    ],
    temperature: Annotated[
        str, Field(description="Temperature variable, K.")
    ] = "temperature",
    humidity: Annotated[
        str | None, Field(description="Specific humidity variable.")
    ] = "specific_humidity",
    u: Annotated[
        str | None, Field(description="Eastward wind variable.")
    ] = "u_component_of_wind",
    v: Annotated[
        str | None, Field(description="Northward wind variable.")
    ] = "v_component_of_wind",
    others: Annotated[list[str], Field(description="Further variables to carry.")] = [],  # noqa: B006
    time: Annotated[str | None, Field(description="Time to use (nearest).")] = None,
    steps_per_leg: Annotated[int, Field(description="Sample points per leg.")] = 100,
    name: Annotated[str, Field(description="Name for the output.")] = "cross-section",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(grid, [x for x in (temperature, humidity, u, v, *others) if x])
    out = atmosphere.cross_section(
        ds,
        path,
        temperature=temperature,
        humidity=humidity,
        u=u,
        v=v,
        others=others,
        steps_per_leg=steps_per_leg,
        time=time,
    )
    saved = save_grid(
        out,
        name,
        "section",
        output_dir,
        tool="cross_section",
        sources=[_file_source(grid)],
        method={
            "path": path,
            "time": time,
            "steps_per_leg": steps_per_leg,
            "interpolation": "MetPy cross_section, linear",
        },
        packages=("metpy", "xarray"),
    )
    theta = out["potential_temperature"]
    return result(
        f"Section along {len(path)} vertices, {float(out['distance'].max()):.0f} km, {out.sizes[out.attrs['vertical']]} levels; theta {float(theta.min()):.0f} to {float(theta.max()):.0f} K.",
        [saved],
        length_km=float(out["distance"].max()),
        variables=[str(v) for v in out.data_vars],
        path=json.loads(out.attrs["path"]),
    )


@mcp.tool(
    name="soundings",
    title="Soundings",
    description=(
        "Vertical profiles at points from pressure-level fields at one time, with surface-based, "
        "mixed-layer (100 hPa) and most-unstable CAPE and CIN, lifted index and LCL (MetPy); writes the "
        "profiles with the parcel path (for a skew-T) and the index table."
    ),
    annotations=WRITES,
    tags={"weather", "sounding", "instability", "CAPE"},
)
@weather_tool("soundings")
def soundings(
    grid: Annotated[
        str,
        Field(description="Pressure-level NetCDF/GRIB with temperature and humidity."),
    ],
    points: Annotated[
        list[dict[str, Any]], Field(description="[{name, longitude, latitude}, ...].")
    ],
    temperature: Annotated[
        str, Field(description="Temperature variable, K.")
    ] = "temperature",
    humidity: Annotated[
        str, Field(description="Specific humidity variable.")
    ] = "specific_humidity",
    u: Annotated[
        str | None, Field(description="Eastward wind variable.")
    ] = "u_component_of_wind",
    v: Annotated[
        str | None, Field(description="Northward wind variable.")
    ] = "v_component_of_wind",
    time: Annotated[str | None, Field(description="Time to use (nearest).")] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "soundings",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    ds = open_grid(grid, [x for x in (temperature, humidity, u, v) if x])
    profiles, indices = atmosphere.soundings(
        ds, points, temperature=temperature, humidity=humidity, u=u, v=v, time=time
    )
    prov = {
        "tool": "soundings",
        "sources": [_file_source(grid)],
        "method": {
            "time": time,
            "parcels": "surface-based; mixed-layer 100 hPa; most unstable",
            "dewpoint": "from specific humidity",
        },
        "packages": ("metpy",),
    }
    outputs = [
        save_table(profiles, f"{name}-profiles", "profiles", output_dir, **prov),
        save_table(indices, f"{name}-indices", "table", output_dir, **prov),
    ]
    top = indices.sort_values("mucape", ascending=False).iloc[0]
    return result(
        f"{len(indices)} soundings; highest MUCAPE {top['mucape']:.0f} J/kg at {top['point']} (lifted index {top['lifted_index']:.1f}).",
        outputs,
        indices=indices.to_dict("records"),
    )
