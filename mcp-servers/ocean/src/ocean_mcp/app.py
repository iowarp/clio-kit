"""The FastMCP app and the wrapper every ocean tool goes through.

Tools return a structured payload (``status``, ``summary``, numbers and the
``outputs`` written, each with its provenance sidecar) plus the summary as
the human text. A typed :class:`OceanToolError` becomes a ``ToolError``
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

import xarray as xr
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent

from .implementation.common import OceanToolError

logger = logging.getLogger("ocean_mcp")

mcp: FastMCP = FastMCP(
    "ocean",
    version="1.0.0",  # mcp-server-versions.toml; same as common.SERVER_VERSION
    instructions=(
        "Ocean analysis on observations and model output. Read CTD casts (ctd_read) or fetch "
        "Argo (argo_fetch) and glider (glider_fetch) profiles into a profile table; convert to "
        "TEOS-10 (teos10); assign water masses against a stated definition (water_masses); compute "
        "mixed layer depth, stratification and anomaly profiles (profile_metrics); build sections "
        "from profiles or gridded fields (section); transports and overturning (transport, "
        "overturning); tides (harmonics), spectra (spectra), sea-level trends "
        "(sea_level_decompose), bloom starts (bloom_phenology); eddies from sea level (eddy_track, "
        "track_crossings); SWOT swaths (swath_fetch, swath_compare); particles (particles); marine "
        "heatwaves at a point (mhw_point) or tracked over a region (mhw_track, mhw_events); "
        "mixed-layer heat budgets (mixed_layer_budget); cruise plans (cruise_plan); model drift "
        "(drift) and model-observation colocation (colocate); regional MOM6 set-up checks "
        "(mom6_preflight). Gridded inputs (reanalysis, altimetry, model output) are files on "
        "disk, usually fetched with copernicus-mcp first. Every output file has a "
        "<file>.provenance.json sidecar with the product, subset, method and code versions: cite "
        "it. You choose thresholds, baselines, regions and methods; state why."
    ),
)

P = ParamSpec("P")


def ocean_tool(
    label: str,
) -> Callable[[Callable[P, dict[str, Any]]], Callable[P, Awaitable[ToolResult]]]:
    """Wrap a synchronous implementation: typed errors, summary text, payload."""

    def decorate(fn: Callable[P, dict[str, Any]]) -> Callable[P, Awaitable[ToolResult]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
            try:
                payload = fn(*args, **kwargs)
            except OceanToolError as exc:
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


def open_grid(path: str, variables: list[str] | None = None) -> xr.Dataset:
    """Open a gridded dataset: one NetCDF/Zarr path or a glob of NetCDF files."""
    expanded = str(Path(path).expanduser())
    matches = sorted(glob.glob(expanded))
    if not matches:
        raise OceanToolError("bad_input", f"no file matches {path}")
    try:
        if len(matches) > 1:
            ds = xr.open_mfdataset(matches, combine="by_coords")
        elif matches[0].endswith(".zarr") or Path(matches[0]).is_dir():
            ds = xr.open_zarr(matches[0])
        else:
            ds = xr.open_dataset(matches[0])
    except Exception as exc:
        raise OceanToolError("bad_input", f"could not open {path}: {exc}") from exc
    if variables:
        missing = [v for v in variables if v and v not in ds]
        if missing:
            raise OceanToolError(
                "bad_input",
                f"{Path(path).name} lacks {', '.join(missing)}; it has {', '.join(map(str, ds.data_vars))}",
            )
    return ds


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
