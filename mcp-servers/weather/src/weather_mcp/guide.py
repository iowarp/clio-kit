"""Resources and prompts: what the server can do, where its data comes from, workflows.

The capabilities resource groups the tools by the job they do and says which data
source each fetch reads, whether it needs a key, and what needs a GPU or an external
program, so an agent can plan a chain of calls before making any.
"""

from __future__ import annotations

from typing import Any

from fastmcp.prompts import Message

from .app import mcp

#: Every tool, by the job it does. A test keeps this equal to the registered tools.
TOOL_GROUPS: dict[str, list[str]] = {
    "fetch": [
        "era5_fetch",
        "ensemble_fetch",
        "station_fetch",
        "ibtracs_fetch",
        "cmip_change",
    ],
    "forecast_analysis": [
        "exceedance",
        "threshold_events",
        "cross_section",
        "soundings",
    ],
    "objects": ["detect_blobs", "stitch_blobs", "blob_stats", "tc_track"],
    "verification": ["score", "station_verify", "track_score"],
    "climate": ["climate_index", "composite", "evaluate_run"],
    "ai": ["ai_forecast", "downscale"],
}

#: Where each fetching tool reads from; none of them needs a key.
SOURCES: dict[str, str] = {
    "era5_fetch": "ARCO-ERA5 analysis-ready copy on Google Cloud (hourly, single and pressure levels)",
    "ensemble_fetch": "ECMWF open data ensemble (control + 50 members; recent runs only)",
    "station_fetch": "NOAA NCEI GHCN-hourly station observations",
    "ibtracs_fetch": "IBTrACS v04r01 best tracks (NOAA NCEI)",
    "cmip_change": "CMIP6 from the Pangeo cloud catalog",
}


@mcp.resource(
    "weather://capabilities",
    name="capabilities",
    description=(
        "The tools grouped by job, the data source of each fetch, what needs a GPU or an "
        "external program, and what every output carries."
    ),
    mime_type="application/json",
    tags={"weather", "capabilities"},
)
def capabilities() -> dict[str, Any]:
    """What the weather server does, grouped by job, with its sources and requirements."""
    return {
        "tool_groups": TOOL_GROUPS,
        "sources": SOURCES,
        "keys_needed": "none; CDS/Polytope requests come from copernicus-mcp and the ECMWF skills",
        "requirements": {
            "detect_blobs, stitch_blobs, tc_track (tempestextremes backend)": (
                "a TempestExtremes build, found through TEMPESTEXTREMES_BIN or PATH"
            ),
            "ai_forecast, downscale, tc_track (earth2studio backend)": (
                "a CUDA GPU unless allow_cpu; the first call builds a large worker environment"
            ),
        },
        "outputs": (
            "files under WEATHER_MCP_OUTPUT_DIR (default ./weather-outputs), each with a "
            "<file>.provenance.json sidecar naming sources, licences, method and versions"
        ),
        "choices_left_to_you": "thresholds, baselines, regions and methods: state why",
    }


@mcp.prompt(tags={"weather", "ensemble", "workflow"})
def ensemble_risk(
    param: str, threshold: str, place: str, steps: str = "24, 48, 72"
) -> list[Message]:
    """Turn the latest ECMWF ensemble into a probability of exceeding a threshold at a place."""
    return [
        Message(
            f"Fetch the latest ECMWF ensemble of {param} with ensemble_fetch, cut to {place}, "
            f"at forecast steps {steps} hours. With exceedance, compute the probability that "
            f"{param} exceeds {threshold} at each step, saying how many of the members do. "
            "If station_fetch observes the quantity (2 m temperature, dew point or sea-level "
            "pressure), fetch the last few days at the nearest station for context. Report the "
            "probabilities per step with the run time, the threshold's units and why it was "
            "chosen, and cite the provenance sidecars of the outputs."
        )
    ]
