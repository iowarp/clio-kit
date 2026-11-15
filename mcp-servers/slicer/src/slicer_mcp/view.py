"""The sliced-result view: ``show_slice`` and its MCP Apps view (``ui://slicer/layers``).

``show_slice`` opens a view of a saved slice: the toolpaths layer by layer, coloured by
feature type, with a layer slider, per-type toggles, the slicer's time and filament
estimates and its warnings in words. The agent reads a compact summary; the per-layer
numbers travel to the view in ``structuredContent`` and the toolpaths through
``slice_layers``, a tool only the view calls, in chunks.

The view reports ``ui/update-model-context`` with the declared state keys
(:data:`VIEW_STATE`) and offers app tools (``go_to_layer``, ``set_feature_visibility``,
``pick_segment``). The layer and visible types it last showed are saved with the slice
through ``remember_slice_view``, so a reopen (``{slice_id}``) shows what the person
last saw.

Large prints: the view draws at most :data:`VIEW_VERTICES` toolpath points below the
shown layer. Past that it draws every n-th layer there (the shown layer always whole)
and says so in words; every number still comes from every move.
"""

from __future__ import annotations

import base64
import math
from importlib import resources
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from .errors import SlicerError
from .gcode.features import FEATURES, TRAVEL, legend
from .gcode.parser import read_paths
from .present import show_text
from .slicing import layer_stats
from .store import load_record, remember_view, slice_dir

VIEW_URI = "ui://slicer/layers"
REOPEN_META = "ai.iowarp.clio/reopen"
STATE_META = "ai.iowarp.clio/state"
#: three.js, pinned, from jsDelivr (MIT); the view loads it as an ES module.
THREE_ORIGIN = "https://cdn.jsdelivr.net"
#: Toolpath points the view draws below the shown layer; past this it skips layers.
VIEW_VERTICES = 1_500_000
#: Toolpath points one ``slice_layers`` answer carries at most (one layer always fits).
CHUNK_VERTICES = 250_000

VIEW_STATE = [
    {"key": "slice_id", "description": "The slice the view shows; show_slice reopens it."},
    {
        "key": "layer",
        "tool": "go_to_layer",
        "description": "The layer shown on top (0 is the first); set it to move the view.",
    },
    {"key": "layer_count", "description": "How many layers the print has."},
    {"key": "z_mm", "description": "The height of the shown layer, in mm."},
    {"key": "print_time_s", "description": "The slicer's estimated print time, in seconds."},
    {"key": "filament_g", "description": "The slicer's estimated filament use, in grams."},
    {"key": "printer", "description": "The printer preset sliced for."},
    {"key": "material", "description": "The filament preset sliced with."},
    {
        "key": "visible_features",
        "tool": "set_feature_visibility",
        "description": "The feature types drawn (outer wall, infill, support, travel …).",
    },
    {
        "key": "selection",
        "tool": "pick_segment",
        "description": "The toolpath the person picked: type, layer, Z and G-code line.",
    },
]
APP_TOOLS = ["go_to_layer", "set_feature_visibility", "pick_segment"]


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def _stride(record: dict[str, Any]) -> int:
    extruded = sum(
        f.get("vertices", 0)
        for layer in record["layers"]
        for key, f in layer["features"].items()
        if key != TRAVEL
    )
    return max(1, math.ceil(extruded / VIEW_VERTICES))


def view_payload(record: dict[str, Any], layer: int | None, title: str = "") -> dict[str, Any]:
    """What the view needs to draw a slice, except the toolpaths themselves."""
    layers = record["layers"]
    count = len(layers)
    saved = record.get("view") or {}
    if layer is None:
        layer = saved.get("layer", count - 1)
    if not 0 <= int(layer) < count:
        raise SlicerError(
            "layer_out_of_range", f"Layers are numbered 0 to {count - 1}; {layer} is outside that."
        )
    stats = layer_stats(record["slice_id"])
    shares = {f["feature"]: f for f in stats["by_feature"]}
    keys = [k for k in FEATURES if k in set(record["features"]) or k == TRAVEL]
    rows = []
    for row in legend(keys):
        share = shares.get(row["key"], {})
        rows.append(
            {**row, "time_s": share.get("time_s", 0.0), "time_share": share.get("time_share", 0.0)}
        )
    other = shares.get("other_moves", {})
    visible = saved.get("visible_features") or [k for k in keys if k != TRAVEL]
    stride = _stride(record)
    estimates = record["estimates"]
    return {
        "slice_id": record["slice_id"],
        "title": title or record["model_name"],
        "model_name": record["model_name"],
        "printer": record["printer"],
        "quality": record["quality"],
        "material": record["material"],
        "material_type": record.get("material_type", ""),
        "slicer": record["slicer"],
        "estimates": estimates,
        "print_time_s": estimates.get("print_time_s"),
        "filament_g": estimates.get("filament_g"),
        "filament_m": record.get("filament_m"),
        "layer_count": count,
        "layer": int(layer),
        "z_max_mm": record.get("z_max_mm"),
        "bed": record.get("bed"),
        "bounds": record.get("bounds"),
        "legend": rows,
        "other_moves": {
            "label": other.get("label", "Retracts and Z moves"),
            "time_s": other.get("time_s", 0.0),
            "time_share": other.get("time_share", 0.0),
        },
        "visible_features": visible,
        "layers": {
            "z": [r["z"] for r in layers],
            "time_s": [r["time_s"] for r in layers],
            "filament_g": [r["filament_g"] for r in layers],
        },
        "warnings": record.get("warnings") or [],
        "overrides": record.get("overrides") or [],
        "time_basis": record.get("time_basis"),
        "stride": stride,
        "vertices": record.get("vertices", 0),
    }


def show_slice(slice_id: str, layer: int | None = None, title: str = "") -> ToolResult:
    try:
        record = load_record(slice_id)
        payload = view_payload(record, layer, title)
    except SlicerError as exc:
        raise ToolError(exc.text()) from exc
    reopen = {"title": payload["title"], "slice_id": slice_id}
    return ToolResult(
        content=[_text(show_text(payload))],
        structured_content=payload,
        meta={REOPEN_META: {"arguments": reopen}},
    )


def _encode(values: Any) -> str:
    return base64.b64encode(values.tobytes()).decode("ascii")


def slice_layers(
    slice_id: str, first: int, last: int, step: int = 1, travel: bool = False
) -> dict[str, Any]:
    """Toolpaths of layers ``first..last`` (every ``step``-th), at most CHUNK_VERTICES."""
    try:
        record = load_record(slice_id)
    except SlicerError as exc:
        raise ToolError(exc.text()) from exc
    layers = record["layers"]
    if not (0 <= first <= last < len(layers)) or step < 1:
        raise ToolError(
            f"Layers are 0 to {len(layers) - 1}; asked for {first} to {last} step {step}. "
            "[layer_out_of_range]"
        )
    binary = slice_dir(slice_id) / "layers.bin"
    out: list[dict[str, Any]] = []
    sent = 0
    index = first
    next_layer: int | None = None
    while index <= last:
        entry = layers[index]
        wanted = {k: v for k, v in entry["paths"].items() if travel or k != TRAVEL}
        size = sum(v[1] for v in wanted.values())
        if out and sent + size > CHUNK_VERTICES:
            next_layer = index
            break
        paths = read_paths(binary, {"paths": wanted})
        out.append(
            {
                "layer": entry["layer"],
                "z": entry["z"],
                "paths": {
                    key: {
                        "xy": _encode(p["xy"]),
                        "counts": _encode(p["counts"]),
                        "lines": _encode(p["lines"]),
                    }
                    for key, p in paths.items()
                },
            }
        )
        sent += size
        index += step
    return {"slice_id": slice_id, "layers": out, "vertices": sent, "next": next_layer}


def register_view(mcp: FastMCP) -> None:
    """Register ``show_slice``, the view's own tools and the view resource."""

    @mcp.tool(
        name="show_slice",
        title="Show Slice",
        description=(
            "Open a slice in an interactive view the person can turn and zoom: the "
            "toolpaths layer by layer, coloured by feature type (walls, infill, support, "
            "travel), with a layer slider, a toggle per type, the print time, filament "
            "(grams and metres), layer count and height, and the slicer's warnings. The "
            "view reports the shown layer as layer and z_mm and the person's pick as "
            "selection; its tools go_to_layer, set_feature_visibility and pick_segment "
            "move it: to show another layer in a view that is already open, call "
            "go_to_layer rather than opening another view. Pass layer to open on a layer."
        ),
        app=AppConfig(resource_uri=VIEW_URI),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
        tags={"3d-printing", "slicer", "visualization"},
        meta={STATE_META: VIEW_STATE},
    )
    async def show_slice_tool(
        # The title comes first: CLIO names a view by its first short text argument.
        title: Annotated[
            str, Field(description="A title for the view, such as the model's name.")
        ] = "",
        slice_id: Annotated[str, Field(description="The slice id slice returned.")] = "",
        layer: Annotated[
            int | None,
            Field(description="The layer to open on (0 is the first); default the last shown."),
        ] = None,
    ) -> ToolResult:
        if not slice_id:
            raise ToolError("show_slice needs the slice_id that slice returned. [slice_id_missing]")
        return show_slice(slice_id, layer, title)

    @mcp.tool(
        name="slice_layers",
        title="Slice Layers",
        description=(
            "Toolpaths of a range of layers for the slice view to draw. Called by the view itself."
        ),
        app=AppConfig(visibility=["app"]),
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
        tags={"3d-printing", "slicer"},
    )
    async def slice_layers_tool(
        slice_id: Annotated[str, Field(description="The slice id.")],
        first: Annotated[int, Field(description="First layer, from 0.")],
        last: Annotated[int, Field(description="Last layer, inclusive.")],
        step: Annotated[int, Field(description="Every n-th layer.")] = 1,
        travel: Annotated[bool, Field(description="Include travel moves.")] = False,
    ) -> dict[str, Any]:
        return slice_layers(slice_id, first, last, step, travel)

    @mcp.tool(
        name="remember_slice_view",
        title="Remember Slice View",
        description=(
            "Save the layer and the feature types the slice view shows, so reopening it "
            "shows them again. Called by the view itself."
        ),
        app=AppConfig(visibility=["app"]),
        # It writes only this server's own record of the view (as geo's edit_globe_view
        # does), never the person's files: a view change must not ask for permission.
        annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
        tags={"3d-printing", "slicer"},
    )
    async def remember_slice_view_tool(
        slice_id: Annotated[str, Field(description="The slice id.")],
        layer: Annotated[int, Field(description="The shown layer.")],
        visible_features: Annotated[
            list[str], Field(description="The feature keys drawn, travel included when shown.")
        ],
    ) -> dict[str, Any]:
        try:
            record = load_record(slice_id)
            if not 0 <= layer < len(record["layers"]):
                raise SlicerError("layer_out_of_range", f"No layer {layer} in slice {slice_id}.")
            known = set(FEATURES)
            remember_view(
                slice_id,
                {"layer": layer, "visible_features": [k for k in visible_features if k in known]},
            )
        except SlicerError as exc:
            raise ToolError(exc.text()) from exc
        return {"slice_id": slice_id, "layer": layer, "saved": True}

    @mcp.resource(
        VIEW_URI,
        name="slicer_layers",
        description="An interactive view of a sliced print, layer by layer (three.js).",
        mime_type="text/html;profile=mcp-app",
        app=AppConfig(
            csp=ResourceCSP(resource_domains=[THREE_ORIGIN], connect_domains=[THREE_ORIGIN]),
            prefers_border=False,
        ),
    )
    def view_resource() -> str:
        return resources.files("slicer_mcp").joinpath("view.html").read_text("utf-8")


__all__ = [
    "APP_TOOLS",
    "CHUNK_VERTICES",
    "VIEW_STATE",
    "VIEW_URI",
    "VIEW_VERTICES",
    "register_view",
    "show_slice",
    "slice_layers",
    "view_payload",
]
