"""Tools on verification: gridded scores, stations, best tracks."""

from __future__ import annotations

from typing import Annotated, Any

import pandas as pd
from pydantic import Field

from .app import FETCHES, WRITES, mcp, weather_tool
from .implementation import fetch, verification
from .implementation import stations as stations_mod
from .implementation.common import result
from .implementation.grids import open_grid
from .saving import save_grid, save_table
from .tools_forecast import _file_source


def _read_table(path: str) -> pd.DataFrame:
    return pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path)


@mcp.tool(
    name="score",
    title="Forecast Scores",
    description=(
        "Score forecasts against a reference (for example ERA5, or an IFS forecast) by lead time over a "
        "region: cos-latitude weighted RMSE and bias, ACC against a climatology if given, and for "
        "ensembles fair CRPS and spread; optionally the error map at one lead time."
    ),
    annotations=WRITES,
    tags={"weather", "verification", "scores"},
)
@weather_tool("score")
def score(
    forecast: Annotated[
        str,
        Field(
            description="Forecast NetCDF/Zarr with a lead dimension (and init time)."
        ),
    ],
    reference: Annotated[
        str, Field(description="Reference NetCDF/Zarr with valid times.")
    ],
    variable: Annotated[str, Field(description="Forecast variable.")],
    reference_variable: Annotated[
        str | None, Field(description="Reference variable if named differently.")
    ] = None,
    box: Annotated[
        list[float] | None,
        Field(description="[lon_min, lon_max, lat_min, lat_max] region."),
    ] = None,
    climatology: Annotated[
        str | None,
        Field(description="Climatology NetCDF for ACC (dayofyear or fixed)."),
    ] = None,
    climatology_variable: Annotated[
        str | None, Field(description="Variable in the climatology file.")
    ] = None,
    map_lead_hours: Annotated[
        float | None, Field(description="Lead time for the error map.")
    ] = None,
    model: Annotated[
        str, Field(description="Label for this forecast (model name).")
    ] = "forecast",
    name: Annotated[str, Field(description="Name for the outputs.")] = "scores",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    fc = open_grid(forecast, [variable])
    ref = open_grid(reference, [reference_variable or variable])
    clim = (
        open_grid(climatology)[climatology_variable or variable]
        if climatology
        else None
    )
    table, error_map, numbers = verification.score(
        fc,
        ref,
        variable=variable,
        reference_variable=reference_variable or variable,
        box=box,
        climatology=clim,
        map_lead_hours=map_lead_hours,
    )
    table.insert(0, "model", model)
    prov = {
        "tool": "score",
        "sources": [_file_source(forecast, model), _file_source(reference)],
        "method": {
            "box": box,
            "climatology": climatology,
            "crps": "fair (Ferro 2014)",
            "weights": "cos(latitude)",
            **{k: v for k, v in numbers.items() if k != "by_lead"},
        },
        "packages": ("xarray",),
    }
    outputs = [save_table(table, name, "series", output_dir, **prov)]
    if error_map is not None:
        outputs.append(
            save_grid(error_map, f"{name}-error-map", "grid", output_dir, **prov)
        )
    first, last = numbers["by_lead"][0], numbers["by_lead"][-1]
    return result(
        f"{model}: RMSE {first['rmse']:.3g} at {first['lead_hours']:g} h to {last['rmse']:.3g} at {last['lead_hours']:g} h over {numbers['initialisations']} initialisation(s).",
        outputs,
        model=model,
        **numbers,
    )


@mcp.tool(
    name="station_fetch",
    title="Fetch Stations",
    description=(
        "Hourly station observations (2 m temperature, dew point, sea-level pressure) from NOAA "
        "GHCN-hourly, for stations named by ICAO code (e.g. LPPT), WMO id or GHCNh id, over a "
        "period; values with a failed quality code are removed and counted; no key."
    ),
    annotations=FETCHES,
    tags={"weather", "observations", "stations", "fetch"},
)
@weather_tool("station_fetch")
def station_fetch(
    stations: Annotated[
        list[str],
        Field(description="GHCNh station ids, four-letter ICAO codes or WMO ids."),
    ],
    start: Annotated[str, Field(description="Start date.")],
    end: Annotated[str, Field(description="End date.")],
    name: Annotated[str, Field(description="Name for the output.")] = "stations",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table, src, done = stations_mod.ghcnh_stations(stations, start, end)
    out = save_table(
        table,
        name,
        "points",
        output_dir,
        tool="station_fetch",
        sources=[src],
        method={
            "variables": list(stations_mod.VARIABLES.values()),
            "quality": "values with a suspect or failed quality code removed",
            **done,
        },
        packages=("pandas",),
    )
    counts = table.groupby("station").size().to_dict()
    return result(
        f"{len(table)} observations from {len(counts)} stations.",
        [out],
        observations_by_station={str(k): int(v) for k, v in counts.items()},
        stations=done["stations"],
        removed_by_quality_code=done["removed_by_quality_code"],
        licence=src.licence,
    )


@mcp.tool(
    name="station_verify",
    title="Verify At Stations",
    description=(
        "Forecast at each station (bilinear) against the nearest observation in time: pairs, and bias, "
        "RMSE and MAE by station and lead time. 'offset' converts units (e.g. -273.15 for K to °C)."
    ),
    annotations=WRITES,
    tags={"weather", "verification", "stations"},
)
@weather_tool("station_verify")
def station_verify(
    forecast: Annotated[
        str, Field(description="Forecast NetCDF/GRIB with a lead dimension.")
    ],
    observations: Annotated[
        str,
        Field(description="Station table (station, time, latitude, longitude, value)."),
    ],
    variable: Annotated[str, Field(description="Forecast variable.")],
    observed: Annotated[str, Field(description="Observation column.")],
    offset: Annotated[
        float, Field(description="Added to the forecast before comparing.")
    ] = 0.0,
    tolerance_minutes: Annotated[
        int, Field(description="Largest obs-forecast time gap, minutes.")
    ] = 30,
    name: Annotated[str, Field(description="Name for the outputs.")] = "station-verify",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    pairs, stats, numbers = verification.station_verify(
        open_grid(forecast, [variable]),
        _read_table(observations),
        variable=variable,
        observed=observed,
        offset=offset,
        tolerance_minutes=tolerance_minutes,
    )
    prov = {
        "tool": "station_verify",
        "sources": [_file_source(forecast), _file_source(observations)],
        "method": {
            "offset": offset,
            "tolerance_minutes": tolerance_minutes,
            "interpolation": numbers["interpolation"],
        },
        "packages": ("xarray",),
    }
    outputs = [
        save_table(pairs, f"{name}-pairs", "points", output_dir, **prov),
        save_table(stats, f"{name}-stats", "table", output_dir, **prov),
    ]
    return result(
        f"{numbers['pairs']} forecast-observation pairs at {numbers['stations']} stations.",
        outputs,
        **numbers,
        worst=stats.sort_values("rmse", ascending=False).head(10).to_dict("records"),
    )


@mcp.tool(
    name="ibtracs_fetch",
    title="Fetch Best Track",
    description=(
        "Best track of a tropical cyclone from IBTrACS v04r01 (NOAA NCEI), by storm id (SID) or by "
        "name and season; subset 'last3years', 'ACTIVE' or a basin ('NA', 'WP', 'EP', 'NI', 'SI', 'SP', "
        "'SA') to keep the download small. Times, positions, pressure and wind: the WMO agency's values, "
        "or JTWC's where the WMO agency has none (provisional tracks), with the agency of each row and "
        "both agencies' columns (WMO winds are 10-minute means, JTWC's 1-minute)."
    ),
    annotations=FETCHES,
    tags={"weather", "cyclone", "best track", "fetch"},
)
@weather_tool("ibtracs_fetch")
def ibtracs_fetch(
    subset: Annotated[
        str, Field(description="IBTrACS file: last3years, ACTIVE, ALL or a basin code.")
    ] = "last3years",
    storm: Annotated[str | None, Field(description="IBTrACS SID.")] = None,
    name: Annotated[str | None, Field(description="Storm name.")] = None,
    season: Annotated[int | None, Field(description="Season year.")] = None,
    output_name: Annotated[
        str, Field(description="Name for the output.")
    ] = "best-track",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    table, src = fetch.ibtracs(subset, storm, season, name)
    out = save_table(
        table,
        output_name,
        "tracks",
        output_dir,
        tool="ibtracs_fetch",
        sources=[src],
        method={"subset": subset},
        packages=("pandas",),
    )
    storms = (
        table.groupby(["track", "name"])
        .agg(start=("time", "min"), end=("time", "max"), min_mslp=("mslp_hpa", "min"))
        .reset_index()
    )
    return result(
        f"{len(storms)} storm(s), {len(table)} best-track points.",
        [out],
        storms=storms.astype({"start": str, "end": str}).to_dict("records"),
        agencies={
            "pressure": table["mslp_agency"].value_counts().to_dict(),
            "wind": table["wind_agency"].value_counts().to_dict(),
        },
        licence=src.licence,
    )


@mcp.tool(
    name="track_score",
    title="Track Errors",
    description=(
        "Errors of forecast cyclone tracks (one or many models) against a best track: great-circle "
        "position error, central pressure and wind errors by model and lead time."
    ),
    annotations=WRITES,
    tags={"weather", "cyclone", "verification"},
)
@weather_tool("track_score")
def track_score(
    forecasts: Annotated[
        list[str],
        Field(
            description="Forecast track tables (model, time, lead_hours, latitude, longitude[, mslp_hpa, wind_kt])."
        ),
    ],
    best_track: Annotated[
        str, Field(description="Best track table from ibtracs_fetch.")
    ],
    keep_models: Annotated[
        list[str] | None,
        Field(description="Only these models (the scientist's kept runs)."),
    ] = None,
    name: Annotated[str, Field(description="Name for the output.")] = "track-errors",
    output_dir: Annotated[
        str | None, Field(description="Directory for outputs.")
    ] = None,
) -> dict[str, Any]:
    fc = pd.concat([_read_table(p) for p in forecasts], ignore_index=True)
    if keep_models:
        fc = fc[fc["model"].isin(keep_models)]
    table, numbers = verification.track_score(fc, _read_table(best_track))
    out = save_table(
        table,
        name,
        "table",
        output_dir,
        tool="track_score",
        sources=[_file_source(p) for p in forecasts] + [_file_source(best_track)],
        method={
            "interpolation": "best track linear in time",
            "distance": "great circle, R = 6371 km",
            "kept_models": keep_models,
        },
        packages=("pandas",),
    )
    means = ", ".join(
        f"{m} {e:.0f} km" for m, e in numbers["mean_position_error_km"].items()
    )
    return result(f"Mean position error by model: {means}.", [out], **numbers)
