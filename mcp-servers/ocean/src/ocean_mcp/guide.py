"""Resources and prompts: what the server can do, its built-in definitions, workflows.

The capabilities resource groups the tools by the job they do, so an agent can see
which tools chain into which before calling any; the water-mass resource shows the
built-in definitions ``water_masses`` assigns against, with their citations, so the
choice of definition can be stated and checked.
"""

from __future__ import annotations

from typing import Any

from fastmcp.prompts import Message

from .app import mcp
from .implementation import hydrography

#: Every tool, by the job it does. A test keeps this equal to the registered tools.
TOOL_GROUPS: dict[str, list[str]] = {
    "profiles": [
        "ctd_read",
        "argo_fetch",
        "glider_fetch",
        "teos10",
        "water_masses",
        "profile_metrics",
        "section",
    ],
    "circulation": ["transport", "overturning", "particles"],
    "time_series": ["harmonics", "spectra", "sea_level_decompose"],
    "altimetry": ["eddy_track", "track_crossings", "swath_fetch", "swath_compare"],
    "events": [
        "mhw_point",
        "mhw_track",
        "mhw_events",
        "bloom_phenology",
        "mixed_layer_budget",
    ],
    "planning": ["cruise_plan"],
    "models": ["drift", "colocate", "mom6_preflight"],
}


@mcp.resource(
    "ocean://capabilities",
    name="capabilities",
    description="The tools grouped by job, the data they fetch or read, and what every output carries.",
    mime_type="application/json",
    tags={"ocean", "capabilities"},
)
def capabilities() -> dict[str, Any]:
    """What the ocean server does, grouped by job, with its inputs and outputs."""
    return {
        "tool_groups": TOOL_GROUPS,
        "fetches": {
            "argo_fetch": "Argo profiles through argopy (erddap, gdac or argovis)",
            "glider_fetch": "glider deployments and profiles from the IOOS Glider DAC",
            "swath_fetch": "SWOT L2 sea surface height granules (download needs an Earthdata Login)",
        },
        "reads": (
            "CTD casts (Sea-Bird .cnv, CCHDO exchange, CF NetCDF, CSV), profile tables "
            "(.parquet or .csv), and gridded NetCDF or "
            "Zarr files on disk (reanalysis, altimetry, model output), usually fetched "
            "with copernicus-mcp first"
        ),
        "outputs": (
            "files under OCEAN_MCP_OUTPUT_DIR (default ./ocean-outputs), each with a "
            "<file>.provenance.json sidecar naming the product, subset, method and code "
            "versions"
        ),
        "water_mass_definitions": sorted(hydrography.builtin_definitions()),
        "choices_left_to_you": "thresholds, baselines, regions and methods: state why",
    }


@mcp.resource(
    "ocean://water-masses",
    name="water_masses",
    description="The built-in water-mass definitions water_masses assigns against, with citations.",
    mime_type="application/json",
    tags={"ocean", "water masses"},
)
def water_mass_definitions() -> dict[str, Any]:
    """The water-mass definitions shipped with the server, keyed by name."""
    return hydrography.builtin_definitions()


@mcp.prompt(tags={"ocean", "profiles", "workflow"})
def water_mass_census(
    box: str, start: str, end: str, definition: str = "atlantic_emery_meincke_1986"
) -> list[Message]:
    """Fetch Argo profiles in a region and period, and say which water masses they hold."""
    return [
        Message(
            f"Fetch Argo profiles with argo_fetch for the region {box} (lon_min, lon_max, "
            f"lat_min, lat_max, p_min, p_max) from {start} to {end}, in standard mode. "
            "Convert them with teos10, then assign water masses with water_masses against "
            f"the '{definition}' definition (read ocean://water-masses to check that it "
            "fits this region, or give your own with its citation). Compute mixed layer "
            "depth and stratification with profile_metrics, and build a section with "
            "section across the main gradient. Report each mass's pressure range, mean "
            "theta-S and sample count, how many samples fall outside every range, and "
            "cite the provenance sidecars of the outputs."
        )
    ]
