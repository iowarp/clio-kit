"""Tools on events and plans: marine heatwaves, blooms, heat budgets, cruises."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal

import numpy as np
import pandas as pd
import xarray as xr
from pydantic import Field

from .app import WRITES, mcp, ocean_tool, open_grid
from .implementation import cruise, events, timeseries
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
from .implementation.workers import run_worker
from .tools_profiles import _save_table

HOBDAY = "Hobday et al. (2016) definition; categories after Hobday et al. (2018)"


def _point_series(grid: str, variable: str, point: list[float]) -> pd.Series:
    ds = open_grid(grid, [variable])
    lon = next(n for n in ("longitude", "lon", "x") if n in ds.coords)
    lat = next(n for n in ("latitude", "lat", "y") if n in ds.coords)
    x = point[0] + 360 if float(ds[lon].max()) > 180 and point[0] < 0 else point[0]
    da = ds[variable].sel({lon: x, lat: point[1]}, method="nearest")
    extra = [d for d in da.dims if d not in ("time",)]
    if extra:
        da = da.isel({d: 0 for d in extra})
    s = da.to_series()
    s.index = pd.to_datetime(s.index, utc=True)
    return s


@mcp.tool(
    name="mhw_point",
    title="Heatwaves At A Point",
    description=(
        "Marine heatwaves (or any threshold events) in a daily series at a point, by the Hobday "
        "definition: day-of-year climatology and percentile threshold over a stated baseline, events of "
        "at least min_duration days, gaps up to max_gap joined; per event duration, max/mean/cumulative "
        "intensity, category and onset/decline rates. Input is a series file or a gridded file and a point."
    ),
    annotations=WRITES,
    tags={"ocean", "marine heatwave", "extremes", "time series"},
)
@ocean_tool("mhw_point")
def mhw_point(
    baseline: Annotated[
        list[int], Field(description="[first_year, last_year] of the baseline.")
    ],
    series: Annotated[
        str | None, Field(description="Series file (time + column).")
    ] = None,
    column: Annotated[
        str | None, Field(description="Value column in the series file.")
    ] = None,
    grid: Annotated[
        str | None, Field(description="Gridded daily file, instead of a series.")
    ] = None,
    variable: Annotated[str | None, Field(description="Gridded variable.")] = None,
    point: Annotated[
        list[float] | None, Field(description="[lon, lat] in the grid (nearest cell).")
    ] = None,
    percentile: Annotated[float, Field(description="Threshold percentile.")] = 90.0,
    min_duration: Annotated[int, Field(description="Shortest event, days.")] = 5,
    max_gap: Annotated[int, Field(description="Longest gap joined, days.")] = 2,
    window_days: Annotated[int, Field(description="Climatology window, days.")] = 11,
    smooth_days: Annotated[int, Field(description="Climatology smoothing, days.")] = 31,
    tail: Annotated[
        Literal["upper", "lower"],
        Field(description="Upper (heat) or lower (cold) events."),
    ] = "upper",
    period: Annotated[
        list[str] | None, Field(description="[start, end] to report events in.")
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "mhw-point",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if series and column:
        s = timeseries.read_series(series, [column])[column]
        src = [local_file_source(Path(series))]
    elif grid and variable and point:
        s = _point_series(grid, variable, point)
        src = [
            Source(
                name=Path(grid).name,
                kind="file",
                reference=grid,
                subset={"point": point},
            )
        ]
    else:
        raise OceanToolError(
            "bad_input", "give series and column, or grid, variable and point"
        )
    pct = percentile if tail == "upper" else 100 - percentile
    frame = events.climatology(
        s,
        baseline=(baseline[0], baseline[1]),
        percentile=pct,
        window_days=window_days,
        smooth_days=smooth_days,
    )
    found = events.detect_events(
        frame, min_duration=min_duration, max_gap=max_gap, tail=tail
    )
    if period and len(found):
        found = found[
            (found["end"] >= pd.Timestamp(period[0]))
            & (found["start"] <= pd.Timestamp(period[1]))
        ]
    method = {
        "definition": HOBDAY,
        "baseline": baseline,
        "percentile": percentile,
        "tail": tail,
        "min_duration_days": min_duration,
        "max_gap_days": max_gap,
        "window_days": window_days,
        "smooth_days": smooth_days,
        "point": point,
    }
    prov = {
        "tool": "mhw_point",
        "sources": src,
        "method": method,
        "packages": ("numpy",),
    }
    outputs = [
        _save_table(
            frame.reset_index(), f"{name}-series", "series", output_dir, **prov
        ),
        _save_table(found, f"{name}-events", "events", output_dir, **prov),
    ]
    top = (
        found.sort_values("intensity_cumulative", ascending=False).head(10)
        if len(found)
        else found
    )
    return result(
        f"{len(found)} events above the {percentile:g}th percentile ({baseline[0]}-{baseline[1]} baseline)"
        + (
            f"; strongest peaked at {top.iloc[0]['intensity_max']:.2f} ({top.iloc[0]['category']}), {int(top.iloc[0]['duration_days'])} days"
            if len(top)
            else ""
        )
        + ".",
        outputs,
        events=len(found),
        top_events=top.astype({"start": str, "end": str, "peak": str}).to_dict(
            "records"
        )
        if len(top)
        else [],
        method=method,
    )


@mcp.tool(
    name="mhw_track",
    title="Track Heatwaves",
    description=(
        "Detect and track marine heatwaves over a region with marEx (anomalies from a stated baseline, "
        "seasonal percentile threshold, connected-object tracking with merging/splitting), keeping events "
        "of at least min_duration days. Writes the event ID field (NetCDF) and per-event daily series; "
        "run mhw_events next for Hobday metrics and categories. Runs in a separate AGPL worker environment."
    ),
    annotations=WRITES,
    tags={"ocean", "marine heatwave", "extremes", "tracking"},
)
@ocean_tool("mhw_track")
def mhw_track(
    grid: Annotated[
        str,
        Field(
            description="Daily SST NetCDF on a regular lon-lat grid (a long record)."
        ),
    ],
    baseline: Annotated[
        list[int], Field(description="[first_year, last_year] reference period.")
    ],
    variable: Annotated[str, Field(description="SST variable.")] = "sst",
    longitude: Annotated[str, Field(description="Longitude coordinate.")] = "lon",
    latitude: Annotated[str, Field(description="Latitude coordinate.")] = "lat",
    time: Annotated[str, Field(description="Time coordinate.")] = "time",
    box: Annotated[
        list[float] | None,
        Field(description="[lon_min, lon_max, lat_min, lat_max] subset."),
    ] = None,
    percentile: Annotated[float, Field(description="Threshold percentile.")] = 90.0,
    method_anomaly: Annotated[
        Literal[
            "fixed_baseline",
            "shifting_baseline",
            "detrend_harmonic",
            "detrend_fixed_baseline",
        ],
        Field(description="marEx anomaly method."),
    ] = "fixed_baseline",
    min_duration: Annotated[int, Field(description="Shortest event kept, days.")] = 5,
    r_fill: Annotated[
        int, Field(description="marEx R_fill: hole filling radius, cells.")
    ] = 2,
    area_filter_quartile: Annotated[
        float, Field(description="Drop objects below this area quantile.")
    ] = 0.5,
    product: Annotated[
        str | None, Field(description="Product name/DOI for provenance.")
    ] = None,
    licence: Annotated[
        str | None, Field(description="Data licence of the product.")
    ] = None,
    name: Annotated[str, Field(description="Name for the outputs.")] = "mhw",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    target = output_path(name, ".events.nc", output_dir)
    request = {
        "input": str(Path(grid).expanduser().resolve()),
        "variable": variable,
        "dimensions": {"time": time, "x": longitude, "y": latitude},
        "box": box,
        "baseline": baseline,
        "percentile": percentile,
        "method_anomaly": method_anomaly,
        "method_extreme": "seasonal_percentile",
        "R_fill": r_fill,
        "area_filter_quartile": area_filter_quartile,
        "min_duration_days": min_duration,
        "output_dir": str(target.parent),
        "stem": name,
    }
    reply = run_worker("marex", request, timeout_s=4 * 3600)
    series_csv = Path(reply["series_path"])
    series_parquet = series_csv.with_suffix(".parquet")
    pd.read_csv(series_csv, parse_dates=["time"]).to_parquet(
        series_parquet, index=False
    )
    series_csv.unlink(missing_ok=True)
    events_nc = Path(reply["events_path"])
    method = {
        k: v for k, v in request.items() if k not in ("input", "output_dir", "stem")
    } | {
        "marEx": reply.get("preprocess_attrs"),
        "tracker": reply.get("tracker_attrs"),
        "worker": reply["worker"],
    }
    src = [
        Source(
            name=product or Path(grid).name,
            kind="file",
            reference=grid,
            licence=licence or "",
        )
    ]
    note = [
        f"marEx {reply.get('marex_version')} ran in a separate AGPL-3.0-or-later process"
    ]
    outputs = [
        Output(
            events_nc,
            "grid",
            media_type(events_nc),
            write_provenance(
                events_nc, tool="mhw_track", sources=src, method=method, notes=note
            ),
        ),
        Output(
            series_parquet,
            "events",
            media_type(series_parquet),
            write_provenance(
                series_parquet, tool="mhw_track", sources=src, method=method, notes=note
            ),
        ),
    ]
    return result(
        f"marEx tracked {reply['events_tracked']} objects; {reply['events_kept']} last at least {min_duration} days.",
        outputs,
        events_tracked=reply["events_tracked"],
        events_kept=reply["events_kept"],
    )


@mcp.tool(
    name="mhw_events",
    title="Heatwave Metrics",
    description=(
        "Hobday metrics and category for every tracked marine heatwave from mhw_track's event series: "
        "start, end, duration, area-mean max/mean/cumulative intensity, intensity ratio to the threshold, "
        "category (moderate/strong/severe/extreme), peak area and centroid path. Optionally only events "
        "overlapping a period or a box."
    ),
    annotations=WRITES,
    tags={"ocean", "marine heatwave", "extremes"},
)
@ocean_tool("mhw_events")
def mhw_events(
    event_series: Annotated[
        str, Field(description="Per-event daily series from mhw_track (.parquet).")
    ],
    period: Annotated[
        list[str] | None, Field(description="[start, end] to keep events overlapping.")
    ] = None,
    name: Annotated[str, Field(description="Name for the output.")] = "mhw-events",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    s = pd.read_parquet(event_series)
    rows = []
    for event_id, group in s.groupby("event"):
        group = group.sort_values("time")
        intensity = group["anomaly_mean"].to_numpy()
        spread = group["threshold_mean"].to_numpy()
        peak = int(np.argmax(intensity))
        ratio = (
            float(intensity[peak] / spread[peak]) if spread[peak] > 0 else float("nan")
        )
        category = (
            events.CATEGORIES[min(max(int(np.floor(ratio)) - 1, 0), 3)]
            if np.isfinite(ratio)
            else "unknown"
        )
        rows.append(
            {
                "event": int(event_id),
                "start": group["time"].iloc[0],
                "end": group["time"].iloc[-1],
                "peak": group["time"].iloc[peak],
                "duration_days": int(
                    (group["time"].iloc[-1] - group["time"].iloc[0]).days + 1
                ),
                "intensity_max": float(intensity[peak]),
                "intensity_mean": float(intensity.mean()),
                "intensity_cumulative": float(intensity.sum()),
                "intensity_ratio": ratio,
                "category": category,
                "peak_cell_anomaly": float(group["anomaly_max"].max()),
                "area_max_km2": float(group["area_km2"].max()),
                "centroid_lon_at_peak": float(group["centroid_lon"].iloc[peak]),
                "centroid_lat_at_peak": float(group["centroid_lat"].iloc[peak]),
            }
        )
    table = pd.DataFrame(rows)
    if period and len(table):
        table = table[
            (table["end"] >= pd.Timestamp(period[0]))
            & (table["start"] <= pd.Timestamp(period[1]))
        ]
    table = table.sort_values("intensity_cumulative", ascending=False)
    out = _save_table(
        table,
        name,
        "events",
        output_dir,
        tool="mhw_events",
        sources=[local_file_source(Path(event_series))],
        method={
            "definition": HOBDAY,
            "intensity": "area-weighted mean anomaly over the event's cells relative to marEx's climatology",
            "category_basis": "peak area-mean anomaly / area-mean threshold anomaly",
            "period": period,
        },
        packages=("pandas",),
    )
    head = table.head(10)
    return result(
        f"{len(table)} tracked events"
        + (
            f"; the largest by cumulative intensity ran {head.iloc[0]['start']:%Y-%m-%d} to {head.iloc[0]['end']:%Y-%m-%d} ({head.iloc[0]['category']}, {head.iloc[0]['duration_days']} days)"
            if len(head)
            else ""
        )
        + ".",
        [out],
        events=len(table),
        top_events=head.astype({"start": str, "end": str, "peak": str}).to_dict(
            "records"
        ),
        categories=table["category"].value_counts().to_dict() if len(table) else {},
    )


@mcp.tool(
    name="bloom_phenology",
    title="Bloom Start",
    description=(
        "Phytoplankton bloom initiation per year from a chlorophyll series (or a gridded product averaged "
        "over a box) by a stated method: median plus 5 % (Siegel et al. 2002) or the steepest rise of "
        "log-chlorophyll, both tied to the chosen months' chlorophyll peak. A year whose bloom is already "
        "under way when the months open gets no date, with the reason."
    ),
    annotations=WRITES,
    tags={"ocean", "chlorophyll", "phenology"},
)
@ocean_tool("bloom_phenology")
def bloom_phenology(
    method: Annotated[
        Literal["median_plus_5pct", "max_rate"],
        Field(description="Bloom start method."),
    ],
    series: Annotated[
        str | None, Field(description="Series file (time + column).")
    ] = None,
    column: Annotated[str | None, Field(description="Chlorophyll column.")] = None,
    grid: Annotated[
        str | None, Field(description="Gridded chlorophyll file instead.")
    ] = None,
    variable: Annotated[
        str | None, Field(description="Gridded variable (e.g. CHL).")
    ] = None,
    box: Annotated[
        list[float] | None,
        Field(description="[lon_min, lon_max, lat_min, lat_max] to average."),
    ] = None,
    season: Annotated[
        list[int], Field(description="[first_month, last_month] to search.")
    ] = [1, 7],  # noqa: B006
    smooth_days: Annotated[int, Field(description="Smoothing window, days.")] = 8,
    name: Annotated[str, Field(description="Name for the outputs.")] = "bloom",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    if series and column:
        s = timeseries.read_series(series, [column])[column]
        src = [local_file_source(Path(series))]
    elif grid and variable and box:
        ds = open_grid(grid, [variable])
        lon = next(n for n in ("longitude", "lon") if n in ds.coords)
        lat = next(n for n in ("latitude", "lat") if n in ds.coords)
        sub = ds[variable].sel({lon: slice(box[0], box[1])})
        sub = (
            sub.sel({lat: slice(box[2], box[3])})
            if sub[lat][0] < sub[lat][-1]
            else sub.sel({lat: slice(box[3], box[2])})
        )
        w = xr.DataArray(
            np.cos(np.deg2rad(sub[lat].to_numpy())), coords=sub[lat].coords
        )
        s = sub.weighted(w).mean([lon, lat]).to_series()
        s.index = pd.to_datetime(s.index, utc=True)
        src = [
            Source(
                name=Path(grid).name, kind="file", reference=grid, subset={"box": box}
            )
        ]
    else:
        raise OceanToolError(
            "bad_input", "give series and column, or grid, variable and box"
        )
    table, numbers = events.bloom_start(
        s, method=method, season=(season[0], season[1]), smooth_days=smooth_days
    )
    prov = {
        "tool": "bloom_phenology",
        "sources": src,
        "method": {
            "method": method,
            "season_months": season,
            "smooth_days": smooth_days,
            "box": box,
        },
        "packages": ("pandas",),
    }
    series_frame = (
        s.rename("chlorophyll")
        .reset_index()
        .rename(columns={s.index.name or "index": "time"})
    )
    outputs = [
        _save_table(table, f"{name}-starts", "events", output_dir, **prov),
        _save_table(series_frame, f"{name}-series", "series", output_dir, **prov),
    ]
    return result(
        f"Bloom start found in {numbers['years_with_start']} of {numbers['years']} years ({method}).",
        outputs,
        **numbers,
        starts=table.astype({"bloom_start": str}).to_dict("records"),
    )


@mcp.tool(
    name="mixed_layer_budget",
    title="Mixed Layer Budget",
    description=(
        "Mixed-layer heat budget over an event: temperature tendency against net surface heat flux "
        "divided by rho cp h (with optional shortwave penetration below h), the residual standing for "
        "advection, entrainment and mixing; daily and cumulative terms."
    ),
    annotations=WRITES,
    tags={"ocean", "heat budget", "mixed layer"},
)
@ocean_tool("mixed_layer_budget")
def mixed_layer_budget(
    series: Annotated[
        str,
        Field(
            description="Daily table (time + columns) with temperature, flux and MLD."
        ),
    ],
    temperature: Annotated[
        str, Field(description="Mixed-layer (or SST) temperature column, °C.")
    ],
    net_flux: Annotated[
        str, Field(description="Net downward surface heat flux column, W m-2.")
    ],
    mld: Annotated[str, Field(description="Mixed layer depth column, m.")],
    start: Annotated[str, Field(description="Event start.")],
    end: Annotated[str, Field(description="Event end.")],
    shortwave: Annotated[
        str | None, Field(description="Net shortwave column, W m-2, for penetration.")
    ] = None,
    name: Annotated[str, Field(description="Name for the output.")] = "budget",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    cols = [c for c in (temperature, net_flux, mld, shortwave) if c]
    frame = timeseries.read_series(series, cols)
    frame.index = frame.index.tz_convert(None)
    table, numbers = events.mixed_layer_budget(
        frame,
        temperature=temperature,
        net_flux=net_flux,
        shortwave=shortwave,
        mld=mld,
        start=start,
        end=end,
    )
    out = _save_table(
        table,
        name,
        "series",
        output_dir,
        tool="mixed_layer_budget",
        sources=[local_file_source(Path(series))],
        method={
            "equation": "dT/dt = (Q_net - Q_pen(h)) / (rho0 cp h) + residual",
            "rho0": 1025.0,
            "cp": 3992.0,
            **numbers,
        },
        packages=("pandas",),
    )
    return result(
        f"Over {numbers['window'][0]} to {numbers['window'][1]} the mixed layer warmed {numbers['temperature_change_K']:.2f} K; surface flux accounts for {numbers['surface_flux_contribution_K']:.2f} K, the residual {numbers['residual_contribution_K']:.2f} K.",
        [out],
        **numbers,
    )


@mcp.tool(
    name="cruise_plan",
    title="Cruise Plan",
    description=(
        "Plan stations to sample a moving feature (an eddy's forecast track) within ship time: steam from "
        "a port, a line of stations across the feature where it will be on arrival, and back; or re-time "
        "a plan whose stations the scientist moved (give stations). Reports whether it fits."
    ),
    annotations=WRITES,
    tags={"ocean", "fieldwork", "planning"},
)
@ocean_tool("cruise_plan")
def cruise_plan(
    port: Annotated[list[float], Field(description="[lon, lat] of the port.")],
    depart: Annotated[str, Field(description="Departure time (ISO).")],
    days_available: Annotated[float, Field(description="Ship time, days.")],
    track: Annotated[
        str | None,
        Field(description="Forecast track table (time, longitude, latitude)."),
    ] = None,
    track_id: Annotated[
        str | None, Field(description="Track to follow, if the table holds several.")
    ] = None,
    stations: Annotated[
        list[dict[str, Any]] | None,
        Field(
            description="Explicit stations [{name, longitude, latitude}] to re-time."
        ),
    ] = None,
    station_count: Annotated[
        int, Field(description="Stations across the feature.")
    ] = 9,
    line_length_km: Annotated[
        float, Field(description="Length of the station line, km.")
    ] = 160.0,
    bearing_degrees: Annotated[
        float, Field(description="Bearing of the line, degrees from north.")
    ] = 90.0,
    speed_knots: Annotated[float, Field(description="Ship speed, knots.")] = 10.0,
    station_hours: Annotated[
        float, Field(description="Time on each station, hours.")
    ] = 3.0,
    name: Annotated[str, Field(description="Name for the output.")] = "cruise",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    when = pd.Timestamp(depart)
    when = when.tz_localize("UTC") if when.tzinfo is None else when
    port_xy = (port[0], port[1])
    sources: list[Source] = []
    if stations:
        table, numbers = cruise.schedule(
            stations,
            port=port_xy,
            depart=when,
            speed_knots=speed_knots,
            station_hours=station_hours,
            days_available=days_available,
        )
        numbers["mode"] = "re-timed the given stations"
    elif track:
        t = (
            pd.read_parquet(track)
            if track.endswith(".parquet")
            else pd.read_csv(track, parse_dates=["time"])
        )
        if track_id is not None:
            t = t[t["track"].astype(str) == str(track_id)]
        if t.empty:
            raise OceanToolError(
                "no_data", "the track table holds no points for that track"
            )
        table, numbers = cruise.plan_across_feature(
            t,
            port=port_xy,
            depart=when,
            speed_knots=speed_knots,
            station_hours=station_hours,
            days_available=days_available,
            stations=station_count,
            line_length_km=line_length_km,
            bearing_degrees=bearing_degrees,
        )
        numbers["mode"] = "planned across the feature at its forecast arrival position"
        sources.append(local_file_source(Path(track)))
    else:
        raise OceanToolError(
            "bad_input", "give a forecast track, or stations to re-time"
        )
    out = _save_table(
        table,
        name,
        "stations",
        output_dir,
        tool="cruise_plan",
        sources=sources,
        method={
            "port": port,
            "speed_knots": speed_knots,
            "station_hours": station_hours,
            "days_available": days_available,
            "track_id": track_id,
            "bearing_degrees": bearing_degrees,
            "line_length_km": line_length_km,
        },
        packages=("gsw",),
    )
    verdict = "fits" if numbers["fits"] else "does NOT fit"
    return result(
        f"{numbers['stations']} stations, {numbers['steaming_km']:.0f} km steaming; plan {verdict} in {days_available:g} days ({numbers['days_used']:.1f} used).",
        [out],
        **numbers,
        schedule=table.astype({"arrive": str, "leave": str}).to_dict("records"),
    )
