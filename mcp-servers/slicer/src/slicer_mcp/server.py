"""Slicer MCP server: slice 3D models with the slicer already on this machine.

The server runs PrusaSlicer (or OrcaSlicer) installed here, as a separate process, and
never downloads one (:mod:`.backends.locate`). It chooses presets from the slicer's own
vendor bundles, slices, reads the G-code once into per-layer typed toolpaths, and shows
them in an MCP Apps view (:mod:`.view`). It never talks to a printer.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import present, slicing
from .errors import SlicerError
from .view import register_view

PATHS_META = "ai.iowarp.clio/paths"

mcp = FastMCP(
    "slicer",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "Slice 3D models (STL, 3MF, STEP) for a 3D printer with the PrusaSlicer or "
        "OrcaSlicer installed on this machine, and show the toolpaths layer by layer. "
        "list_printers finds a printer preset and its qualities and materials; slice "
        "returns a slice id with time, filament, layers and warnings; show_slice opens "
        "the layer view; layer_stats breaks time and filament down by feature type; "
        "export writes the G-code or a per-layer CSV; compare_slices compares slices. "
        "No printer is controlled. If no slicer is installed, every tool says so "
        "(slicer_missing)."
    ),
)

T = TypeVar("T")


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except SlicerError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


@mcp.tool(
    name="list_printers",
    title="List Printers",
    description=(
        "List the printer presets the installed slicer knows. With no argument: its "
        "vendors. With vendor: that vendor's printer models and presets. With printer "
        "(words such as 'Prusa MK4' or an exact preset name): the matching preset and its "
        "quality (layer height) and material presets."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"3d-printing", "slicer"},
)
async def list_printers(
    vendor: Annotated[
        str | None, Field(description="A vendor name or part of one, e.g. 'Prusa'.")
    ] = None,
    printer: Annotated[
        str | None, Field(description="A printer in words or an exact preset name.")
    ] = None,
) -> ToolResult:
    data = _run(lambda: slicing.list_printers(vendor, printer))
    return _result(present.printers_text(data), data)


@mcp.tool(
    name="slice",
    title="Slice Model",
    description=(
        "Slice a 3D model (STL, 3MF, STEP, OBJ) for a printer with the slicer installed "
        "here. printer, material and quality are words ('Prusa MK4', 'PLA', '0.2 mm') or "
        "exact preset names; the presets chosen are reported. overrides sets slicer "
        "settings by their own names (support_material: true, fill_density: '20%', "
        "brim_width: 5); rotate turns the model in degrees about x, y, z before slicing. "
        "Returns a slice id, print time, filament in grams and metres, layer count, and "
        "the slicer's warnings in words. The G-code stays with the server; export writes "
        "it out."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    tags={"3d-printing", "slicer"},
    meta={PATHS_META: {"model": {"machine": "local", "access": "read"}}},
)
async def slice_tool(
    model: Annotated[str, Field(description="Path of the model file (STL, 3MF, STEP, OBJ).")],
    printer: Annotated[str, Field(description="The printer: words or an exact preset name.")],
    material: Annotated[
        str | None, Field(description="The material: a type such as PLA or a preset name.")
    ] = None,
    quality: Annotated[
        str | None,
        Field(description="Layer height ('0.2 mm'), a quality word, or a print preset name."),
    ] = None,
    overrides: Annotated[
        dict[str, Any] | None,
        Field(description="Slicer settings to change, by the slicer's own setting names."),
    ] = None,
    rotate: Annotated[
        dict[str, float] | None,
        Field(description="Degrees to turn the model before slicing: {x, y, z}."),
    ] = None,
) -> ToolResult:
    record = _run(lambda: slicing.slice_model(model, printer, material, quality, overrides, rotate))
    data = {k: v for k, v in record.items() if k not in {"layers", "view"}}
    data["layer_count"] = len(record["layers"])
    return _result(present.slice_text(record), data)


@mcp.tool(
    name="layer_stats",
    title="Layer Stats",
    description=(
        "Time and filament of a slice by feature type (outer wall, inner wall, infill, "
        "support, travel …) over a range of layers (all when omitted), the slowest layer, "
        "and one row per layer (layer, z_mm, time_s, filament_g, the types it prints and "
        "where on the bed each prints)."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"3d-printing", "slicer"},
)
async def layer_stats_tool(
    slice_id: Annotated[str, Field(description="The slice id slice returned.")],
    first_layer: Annotated[int | None, Field(description="First layer, from 0.")] = None,
    last_layer: Annotated[int | None, Field(description="Last layer, inclusive.")] = None,
) -> ToolResult:
    data = _run(lambda: slicing.layer_stats(slice_id, first_layer, last_layer))
    return _result(present.stats_text(data), data)


@mcp.tool(
    name="export",
    title="Export Slice",
    description=(
        "Write a slice out: format 'gcode' (the plain-text G-code for the printer) or "
        "'csv' (one row per layer: z_mm, height_mm, time_s, filament_g and time per "
        "feature type, for a chart). path defaults to the model's folder."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"3d-printing", "slicer"},
    meta={PATHS_META: {"path": {"machine": "local", "access": "write"}}},
)
async def export_tool(
    slice_id: Annotated[str, Field(description="The slice id slice returned.")],
    format: Annotated[str, Field(description="'gcode' or 'csv'.")] = "gcode",
    path: Annotated[
        str | None, Field(description="File or folder to write; default the model's folder.")
    ] = None,
    overwrite: Annotated[bool, Field(description="Replace an existing file.")] = False,
) -> ToolResult:
    data = _run(lambda: slicing.export(slice_id, format, path, overwrite))
    return _result(present.export_text(data), data)


@mcp.tool(
    name="compare_slices",
    title="Compare Slices",
    description=(
        "Compare two or more slices: print time, filament, layers, presets, overrides and "
        "where the time goes, each against the first."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"3d-printing", "slicer"},
)
async def compare_slices_tool(
    slice_ids: Annotated[list[str], Field(description="Two or more slice ids.")],
) -> ToolResult:
    data = _run(lambda: slicing.compare_slices(slice_ids))
    return _result(present.compare_text(data), data)


register_view(mcp)


@mcp.resource("slicer://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the slicer MCP server can do."""
    return {
        "tools": [
            "list_printers",
            "slice",
            "show_slice",
            "layer_stats",
            "export",
            "compare_slices",
        ],
        "slicers": ["PrusaSlicer (prusa-slicer)", "OrcaSlicer (orca-slicer)"],
        "finds_slicer": "SLICER_BIN, then PATH, then the usual install folders; never downloads",
        "accepts": "STL, 3MF, STEP, OBJ, AMF models",
        "outputs": [
            "slice id with time, filament and warnings",
            "interactive layer view (MCP Apps, ui://slicer/layers)",
            "plain-text G-code",
            "per-layer CSV",
        ],
        "printer_control": "none: slicing and viewing only",
    }


@mcp.prompt(tags={"3d-printing", "slicer", "workflow"})
def plan_print(model: str, printer: str, material: str = "PLA") -> list[Message]:
    """Slice a model for a printer, look at the layers, and weigh a faster or stronger option."""
    return [
        Message(
            f"Plan a print of {model} on a {printer} in {material}. Use list_printers to "
            "find the printer preset and its qualities and materials, then slice it and "
            "report the presets chosen, print time, filament and the slicer's warnings. "
            "Open the layers with show_slice and use layer_stats to say where the time "
            "goes. Slice once more with one change worth weighing (another layer height, "
            "infill or supports), put the two side by side with compare_slices, and "
            "recommend one. Export the G-code only when asked; no printer is controlled."
        )
    ]


def main() -> None:
    """Entry point for the slicer MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
