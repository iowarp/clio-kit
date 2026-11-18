"""The FastMCP app and the wrapper every weather tool goes through.

Tools return a structured payload (``status``, ``summary``, numbers and the
``outputs`` written, each with its provenance sidecar) plus the summary as
the human text. A typed :class:`WeatherToolError` becomes a ``ToolError``
whose message starts with its reason code, so the agent and the transcript
both see why.
"""

from __future__ import annotations

import functools
import glob
import logging
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, ParamSpec

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from .implementation.common import WeatherToolError

logger = logging.getLogger("weather_mcp")

mcp: FastMCP = FastMCP(
    "weather",
    version="1.0.0",  # mcp-server-versions.toml; same as common.SERVER_VERSION
    instructions=(
        "Weather and climate analysis on gridded fields and series. Fetch ERA5 fields from the "
        "analysis-ready copy (era5_fetch, no key) or ECMWF ensemble open data (ensemble_fetch); "
        "ERA5 point series, CDS/Polytope requests and deterministic forecasts come from the ECMWF "
        "skills and copernicus-mcp. Exceedance probabilities from an ensemble (exceedance); "
        "threshold events such as heat waves on a series (threshold_events); cross-sections along a "
        "path and soundings with CAPE, CIN and lifted index (cross_section, soundings); object "
        "detection and tracking with TempestExtremes for atmospheric rivers, blocks and other blobs "
        "(detect_blobs, stitch_blobs, blob_stats); forecast scores against a reference (score), "
        "station observations and verification (station_fetch, station_verify), best tracks and "
        "track errors (ibtracs_fetch, track_score); climate indices, composites, run evaluation and "
        "CMIP6 regional change (climate_index, composite, evaluate_run, cmip_change); AI forecasts, "
        "cyclone tracking and downscaling with Earth2Studio (ai_forecast, tc_track, downscale). "
        "Every output file has a <file>.provenance.json sidecar with sources, licences, method and "
        "versions: cite it. You choose thresholds, baselines, regions and methods; state why."
    ),
)

P = ParamSpec("P")


def weather_tool(
    label: str,
) -> Callable[[Callable[P, dict[str, Any]]], Callable[P, Awaitable[ToolResult]]]:
    """Wrap a synchronous implementation: typed errors, summary text, payload."""

    def decorate(fn: Callable[P, dict[str, Any]]) -> Callable[P, Awaitable[ToolResult]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
            try:
                payload = fn(*args, **kwargs)
            except WeatherToolError as exc:
                raise ToolError(f"{exc.reason}: {exc.message}") from exc
            except ToolError:
                raise
            except Exception as exc:
                logger.exception("%s failed", label)
                raise ToolError(
                    f"internal_error: {label} failed: {type(exc).__name__}: {exc}"
                ) from exc
            return ToolResult(
                content=[
                    TextContent(type="text", text=str(payload.get("summary", label)))
                ],
                structured_content=payload,
            )

        return wrapper

    return decorate


def expand_paths(patterns: list[str]) -> list[Path]:
    """Files matching each path or glob, sorted and without duplicates."""
    found: set[Path] = set()
    for pattern in patterns:
        expanded = str(Path(pattern).expanduser())
        found.update(Path(m) for m in glob.glob(expanded))
    return sorted(found)


READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
WRITES = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": False,
}
FETCHES = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}
