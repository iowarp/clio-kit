"""The grids MCP server: open, edit and replay hand edits to a gridded field.

``grid_open`` shows a 2-D field (bathymetry, land mask, topography) in an
interactive editor (MCP Apps, ``ui://grids/editor``). Each change the person
makes there is a ``grid_edit`` call: the whole ordered edit list applied to
the base grid, so the result depends only on the base and the list, undo is
the list without its last edit, and the log written beside the result
(``<output>.edits.json``) replays onto a regenerated grid with
``grid_replay``.
"""

from __future__ import annotations

import functools
import json
import logging
from collections.abc import Awaitable, Callable
from importlib import resources
from pathlib import Path
from typing import Annotated, Any, ParamSpec

import numpy as np
from fastmcp import FastMCP
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import editing
from .editing import GridError

logger = logging.getLogger("grids_mcp")

SERVER_VERSION = "1.0.0"
EDITOR_URI = "ui://grids/editor"

mcp: FastMCP = FastMCP(
    "grids",
    version="1.0.0",  # mcp-server-versions.toml; same as SERVER_VERSION
    instructions=(
        "Hand edits to 2-D gridded fields such as model bathymetry, land masks and "
        "topography. grid_open shows a field in an editor the person can draw on; each "
        "change there is a grid_edit call. grid_edit applies an ordered list of edits (set "
        "a value, clip to a range, fill wet cells cut off from the open ocean) located by "
        "longitude and latitude to a base grid and writes the result with an edit log "
        "carrying the base and result hashes. grid_replay applies a saved log to a "
        "regenerated grid. You choose the edits and their values; say why."
    ),
)

P = ParamSpec("P")

EDITS_DESCRIPTION = (
    "Ordered edits, each {op, region?, note?, ...}. op 'set' takes value; 'clip' takes min "
    "and/or max and optionally wet ('positive', 'negative' or 'nonzero') to clip only wet "
    "cells; 'fill_enclosed' takes wet and land_value and turns wet cells not connected to "
    "the largest wet area into land. region is {box: [west, south, east, north]} or "
    "{points: [[lon, lat], ...]} (nearest cell each); without it the edit covers the grid."
)


def grids_tool(
    label: str,
) -> Callable[[Callable[P, dict[str, Any]]], Callable[P, Awaitable[ToolResult]]]:
    """Typed errors as ``reason: message``; the summary as text; the payload as data."""

    def decorate(fn: Callable[P, dict[str, Any]]) -> Callable[P, Awaitable[ToolResult]]:
        @functools.wraps(fn)
        async def wrapper(*args: P.args, **kwargs: P.kwargs) -> ToolResult:
            try:
                payload = fn(*args, **kwargs)
            except GridError as exc:
                raise ToolError(f"{exc.reason}: {exc.message}") from exc
            except Exception as exc:
                logger.exception("%s failed", label)
                raise ToolError(
                    f"internal_error: {label} failed: {type(exc).__name__}: {exc}"
                ) from exc
            return ToolResult(
                content=[TextContent(type="text", text=str(payload["summary"]))],
                structured_content=payload,
            )

        return wrapper

    return decorate


def _describe(grid: editing.Grid) -> dict[str, Any]:
    values = grid.values
    finite = values[np.isfinite(values)]
    return {
        "path": str(grid.path),
        "variable": grid.variable,
        "shape": list(grid.shape),
        "dims": list(grid.dims),
        "units": str(grid.dataset[grid.variable].attrs.get("units", "")),
        "hash": editing.grid_hash(values, grid.lon, grid.lat),
        "lon_range": [float(grid.lon.min()), float(grid.lon.max())],
        "lat_range": [float(grid.lat.min()), float(grid.lat.max())],
        "value_range": [float(finite.min()), float(finite.max())]
        if finite.size
        else None,
    }


def _default_output(path: Path) -> Path:
    return path.with_name(f"{path.stem}.edited{path.suffix or '.nc'}")


@grids_tool("grid_open")
def grid_open(path: str, variable: str | None = None) -> dict[str, Any]:
    grid = editing.open_grid(path, variable)
    info = _describe(grid)
    rows, cols = grid.shape
    return {
        "status": "ok",
        "summary": (
            f"{grid.variable} in {grid.path.name}: {rows} x {cols} cells, lon "
            f"{info['lon_range'][0]:.2f} to {info['lon_range'][1]:.2f}, lat "
            f"{info['lat_range'][0]:.2f} to {info['lat_range'][1]:.2f}, values "
            f"{info['value_range']}, hash {info['hash']}. Shown in the grid editor."
        ),
        "grid": info,
        "output": str(_default_output(grid.path)),
    }


def _edit(
    path: str,
    edits: list[dict[str, Any]],
    output: str | None,
    variable: str | None,
    expect_base_hash: str | None,
) -> dict[str, Any]:
    grid = editing.open_grid(path, variable)
    base_hash = editing.grid_hash(grid.values, grid.lon, grid.lat)
    if expect_base_hash and expect_base_hash != base_hash:
        raise GridError(
            "bad_input",
            f"the base grid changed: expected hash {expect_base_hash}, {grid.path.name} is "
            f"{base_hash}. Replay the log with grid_replay to apply the edits by location.",
        )
    target = Path(output).expanduser() if output else _default_output(grid.path)
    if target.resolve() == grid.path.resolve():
        raise GridError(
            "bad_input", "output must differ from the base grid, which is kept"
        )
    values, applied = editing.apply_edits(grid, edits)
    result_hash = editing.grid_hash(values, grid.lon, grid.lat)
    log = {
        "schema": editing.LOG_SCHEMA,
        "tool": f"grids-mcp {SERVER_VERSION}",
        "base": {
            "path": str(grid.path),
            "variable": grid.variable,
            "hash": base_hash,
            "shape": list(grid.shape),
        },
        "edits": applied,
        "result": {"path": str(target), "hash": result_hash},
    }
    editing.write_result(grid, values, target, log)
    changed = sum(edit["cells_changed"] for edit in applied)
    return {
        "status": "ok",
        "summary": (
            f"{len(applied)} edits applied to {grid.path.name} ({changed} cell changes); "
            f"wrote {target.name} (hash {result_hash}) and its edit log."
        ),
        "base_hash": base_hash,
        "result_hash": result_hash,
        "edits": applied,
        "output": str(target),
        "log": str(editing.log_path(target)),
    }


@grids_tool("grid_edit")
def grid_edit(
    path: str,
    edits: list[dict[str, Any]],
    output: str | None = None,
    variable: str | None = None,
    expect_base_hash: str | None = None,
) -> dict[str, Any]:
    return _edit(path, edits, output, variable, expect_base_hash)


@grids_tool("grid_replay")
def grid_replay(log: str, path: str, output: str | None = None) -> dict[str, Any]:
    file = Path(log).expanduser()
    try:
        record = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise GridError(
            "bad_input", f"{file} is not a readable edit log: {exc}"
        ) from exc
    if record.get("schema") != editing.LOG_SCHEMA:
        raise GridError(
            "bad_input", f"{file.name} is not a {editing.LOG_SCHEMA} edit log"
        )
    logged = record["base"]
    edits = [
        {key: value for key, value in edit.items() if key != "cells_changed"}
        for edit in record["edits"]
    ]
    result = _edit(path, edits, output, logged.get("variable"), None)
    same_base = result["base_hash"] == logged["hash"]
    counts = [
        {"edit": i, "logged": old.get("cells_changed"), "now": new["cells_changed"]}
        for i, (old, new) in enumerate(
            zip(record["edits"], result["edits"], strict=True)
        )
    ]
    differing = [c for c in counts if c["logged"] != c["now"]]
    result["replay"] = {
        "same_base": same_base,
        "logged_base_hash": logged["hash"],
        "cell_counts": counts,
    }
    result["summary"] = (
        f"Replayed {len(edits)} edits from {file.name} onto {Path(path).name} "
        + (
            "(the same base grid). "
            if same_base
            else f"(a different base: {logged['hash']} logged, {result['base_hash']} now; "
            f"{len(differing)} edits changed a different number of cells). "
        )
        + f"Wrote {Path(result['output']).name} (hash {result['result_hash']})."
    )
    return result


@grids_tool("grid_window")
def grid_window(
    path: str, variable: str | None = None, max_cells: int = 384
) -> dict[str, Any]:
    grid = editing.open_grid(path, variable)
    view = editing.window(grid, grid.values, max_cells)
    return {
        "status": "ok",
        "summary": f"{view['shape'][0]} x {view['shape'][1]} cells of {grid.variable}",
        "grid": _describe(grid),
        "window": view,
    }


PATH = Annotated[str, Field(description="A NetCDF file holding the 2-D grid.")]
VARIABLE = Annotated[
    str | None,
    Field(description="The 2-D variable; optional when the file has only one."),
]


@mcp.tool(
    name="grid_open",
    title="Open Grid",
    description=(
        "Open a 2-D gridded field (model bathymetry, land mask, topography) in an editor the "
        "person can draw on. Returns its shape, extent, value range and content hash, and the "
        "default output path for edits."
    ),
    app=AppConfig(resource_uri=EDITOR_URI),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"grid", "bathymetry", "model", "editor"},
)
async def grid_open_tool(path: PATH, variable: VARIABLE = None) -> ToolResult:
    return await grid_open(path, variable)


@mcp.tool(
    name="grid_edit",
    title="Edit Grid",
    description=(
        "Apply an ordered list of edits to a base grid and write the result beside it with "
        "an edit log (<output>.edits.json) recording the base hash, each edit with the cells "
        "it changed, and the result hash. The base file is never changed. The grid editor "
        "calls this for every change; undo is the same list without its last edit."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"grid", "bathymetry", "model", "editor"},
)
async def grid_edit_tool(
    path: PATH,
    edits: Annotated[list[dict[str, Any]], Field(description=EDITS_DESCRIPTION)],
    output: Annotated[
        str | None,
        Field(description="Result path; default <name>.edited.nc beside the base."),
    ] = None,
    variable: VARIABLE = None,
    expect_base_hash: Annotated[
        str | None,
        Field(
            description="Refuse if the base grid's hash differs (it was regenerated)."
        ),
    ] = None,
) -> ToolResult:
    return await grid_edit(path, edits, output, variable, expect_base_hash)


@mcp.tool(
    name="grid_replay",
    title="Replay Grid Edits",
    description=(
        "Apply a saved edit log to a grid, usually one regenerated after the edits were "
        "made. Edits are located by longitude and latitude, so they land on the same places; "
        "the result says whether the base grid is the logged one and, edit by edit, whether "
        "it changed the same number of cells."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"grid", "bathymetry", "model"},
)
async def grid_replay_tool(
    log: Annotated[str, Field(description="An <output>.edits.json edit log.")],
    path: PATH,
    output: Annotated[str | None, Field(description="Result path.")] = None,
) -> ToolResult:
    return await grid_replay(log, path, output)


@mcp.tool(
    name="grid_window",
    title="Grid Window",
    description="Values of a grid strided for display. Called by the grid editor.",
    app=AppConfig(visibility=["app"]),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"grid"},
)
async def grid_window_tool(
    path: PATH,
    variable: VARIABLE = None,
    max_cells: Annotated[int, Field(description="Cells per axis, at most 512.")] = 384,
) -> ToolResult:
    return await grid_window(path, variable, max_cells)


@mcp.resource(
    EDITOR_URI,
    name="grids_editor",
    description="An editor for a 2-D grid: draw a box or pick cells, choose an edit, apply.",
    mime_type="text/html;profile=mcp-app",
    app=AppConfig(prefers_border=False),
)
def editor_resource() -> str:
    return resources.files("grids_mcp").joinpath("editor.html").read_text("utf-8")


@mcp.prompt(tags={"grid", "bathymetry", "workflow"})
def clean_bathymetry(path: str, variable: str | None = None) -> list[Message]:
    """Find and fix the usual problems in a model bathymetry, with a replayable edit log."""
    field = f" (variable {variable})" if variable else ""
    return [
        Message(
            f"Open the bathymetry at {path}{field} with grid_open and look at its shape, "
            "extent and value range. Look for what breaks an ocean model: wet cells cut "
            "off from the open ocean, cells shallower than the model's minimum depth, and "
            "spurious land or sea in straits and along the coast. Propose each fix as a "
            "grid_edit edit (fill_enclosed, clip with wet, or set over a box or points), "
            "with a note saying why, and apply them in one ordered list. Report the cells "
            "each edit changed, the output path and the edit log, and say that grid_replay "
            "reapplies the log to a regenerated grid."
        )
    ]


def main() -> None:
    """Run the server over stdio."""

    mcp.run()


if __name__ == "__main__":
    main()
