"""CAD MCP server: model parts with build123d code and see them in 3D.

``build_part`` runs a build123d script in its own process, exports the part as STL
and STEP and opens it in an interactive view (``ui://cad/part``, MCP Apps);
``show_part`` reopens a part; ``part_examples`` gives worked scripts (a wing
section, a car body, a cylinder, a pipe bend). The STL is ready for meshers and
flow solvers, the STEP for other CAD tools.
"""

from __future__ import annotations

import base64
from importlib import resources
from typing import Annotated, Any

from fastmcp import FastMCP
from fastmcp.apps import AppConfig, ResourceCSP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import parts
from .examples import EXAMPLES
from .parts import PartError

VERSION = "1.0.0"  # keep equal to mcp-server-versions.toml
VIEW_URI = "ui://cad/part"
PATHS_META = "ai.iowarp.clio/paths"
REOPEN_META = "ai.iowarp.clio/reopen"
STATE_META = "ai.iowarp.clio/state"
THREE_ORIGIN = "https://cdn.jsdelivr.net"
#: Triangles the view gets in one part_mesh answer.
CHUNK_TRIANGLES = 200_000

VIEW_STATE = [
    {"key": "part", "description": "The part folder the view shows; show_part reopens it."},
    {"key": "name", "description": "The part's name."},
    {"key": "size", "description": "Bounding box size x, y, z in model units."},
    {
        "key": "direction",
        "tool": "look_from",
        "description": "Where the camera looks from: front, back, top, bottom, left, right, iso.",
    },
    {
        "key": "edges",
        "tool": "look_from",
        "description": "Whether the triangle edges are drawn over the shading.",
    },
]

mcp = FastMCP(
    "cad",
    version=VERSION,
    instructions=(
        "Model solid parts with build123d Python code. build_part runs a script (with "
        "'from build123d import *' already done) that assigns the finished shape to "
        "result, exports part.stl and part.step and shows the part in 3D; read the "
        "returned size and volume to check it. part_examples gives worked scripts. "
        "For flow work, draw the part with x along the flow, y up and z across, in "
        "body lengths."
    ),
)


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def _fmt(values: list[float]) -> str:
    return " x ".join(f"{v:.4g}" for v in values)


def _summary(rec: dict[str, Any], built: bool) -> str:
    verb = "Built" if built else "Showing"
    lines = [
        f"{verb} {rec['name']} in {rec['part']}.",
        f"Size {_fmt(rec['size'])} (x y z), from ({_fmt(rec['lower'])}) to ({_fmt(rec['upper'])}).",
        (
            f"Volume {rec['volume']:.5g}, surface area {rec['area']:.5g}, {rec['solids']} solid(s), "
            f"{rec['triangles']:,} STL triangles, {'valid' if rec['valid'] else 'NOT valid'}."
        ),
        f"STL {rec['stl']}",
        f"STEP {rec['step']}",
    ]
    if not rec["valid"]:
        lines.append("The shape is not a valid solid; meshers and solvers may reject it.")
    if rec["solids"] != 1:
        lines.append(f"It has {rec['solids']} solids; a flow body is usually one closed solid.")
    return "\n".join(lines)


def _open(rec: dict[str, Any], built: bool, title: str) -> ToolResult:
    payload = {**rec, "title": title or rec["name"]}
    return ToolResult(
        content=[_text(_summary(rec, built))],
        structured_content=payload,
        meta={REOPEN_META: {"arguments": {"title": payload["title"], "part": rec["part"]}}},
    )


@mcp.tool(
    name="build_part",
    title="Build Part",
    description=(
        "Model a part: run build123d code (from build123d import * is already done; assign "
        "the finished shape to result), export part.stl and part.step to "
        "<out_dir>/<name>, and open it in an interactive 3D view the person can rotate. "
        "Returns the size, bounds, volume, area, solid count and validity, and the file "
        "paths. Building the same name again replaces that part. See part_examples for "
        "worked scripts."
    ),
    app=AppConfig(resource_uri=VIEW_URI),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"cad", "build123d"},
    meta={STATE_META: VIEW_STATE, PATHS_META: {"out_dir": {"machine": "local", "access": "write"}}},
)
async def build_part_tool(
    title: Annotated[str, Field(description="A title for the view, such as the part's name.")],
    name: Annotated[str, Field(description="The part's name, used as its folder name.")],
    code: Annotated[str, Field(description="build123d Python code that sets result.")],
    out_dir: Annotated[
        str,
        Field(description="Folder to keep parts in; default CAD_PARTS_DIR or ~/.clio/cad/parts."),
    ] = "",
) -> ToolResult:
    try:
        rec = parts.build(name, code, out_dir)
    except PartError as exc:
        raise ToolError(exc.text()) from exc
    return _open(rec, True, title)


@mcp.tool(
    name="show_part",
    title="Show Part",
    description="Open a part build_part made in the interactive 3D view again.",
    app=AppConfig(resource_uri=VIEW_URI),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cad", "visualization"},
    meta={STATE_META: VIEW_STATE},
)
async def show_part_tool(
    title: Annotated[str, Field(description="A title for the view.")] = "",
    part: Annotated[
        str, Field(description="The part folder build_part returned, or the part's name.")
    ] = "",
) -> ToolResult:
    if not part:
        raise ToolError("show_part needs the part folder that build_part returned. [part_missing]")
    try:
        rec = parts.record(parts.resolve(part))
    except PartError as exc:
        raise ToolError(exc.text()) from exc
    return _open(rec, False, title)


@mcp.tool(
    name="part_mesh",
    title="Part Mesh",
    description="Triangles of a part for the 3D view to draw. Called by the view itself.",
    app=AppConfig(visibility=["app"]),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cad"},
)
async def part_mesh_tool(
    part: Annotated[str, Field(description="The part folder.")],
    first: Annotated[int, Field(description="First triangle, from 0.")] = 0,
) -> dict[str, Any]:
    try:
        folder = parts.resolve(part)
    except PartError as exc:
        raise ToolError(exc.text()) from exc
    corners = parts.read_stl(folder / "part.stl")
    total = len(corners) // 9
    last = min(total, first + CHUNK_TRIANGLES)
    chunk = corners[first * 9 : last * 9]
    return {
        "part": str(folder),
        "triangles": total,
        "first": first,
        "next": last if last < total else None,
        "positions": base64.b64encode(chunk.tobytes()).decode("ascii"),
    }


@mcp.tool(
    name="remember_part_view",
    title="Remember Part View",
    description="Save the direction the part view looks from. Called by the view itself.",
    app=AppConfig(visibility=["app"]),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cad"},
)
async def remember_part_view_tool(
    part: Annotated[str, Field(description="The part folder.")],
    direction: Annotated[str, Field(description="front, back, top, bottom, left, right, iso.")],
    edges: Annotated[bool, Field(description="Whether triangle edges are drawn.")] = False,
) -> dict[str, Any]:
    try:
        folder = parts.resolve(part)
    except PartError as exc:
        raise ToolError(exc.text()) from exc
    parts.remember_view(folder, {"direction": direction, "edges": edges})
    return {"part": str(folder), "saved": True}


@mcp.tool(
    name="list_parts",
    title="List Parts",
    description="The parts built in a folder (default the parts folder), with size and volume.",
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cad"},
    meta={PATHS_META: {"out_dir": {"machine": "local", "access": "read"}}},
)
async def list_parts_tool(
    out_dir: Annotated[str, Field(description="Folder of parts; default the parts folder.")] = "",
) -> ToolResult:
    found = parts.listing(out_dir)
    text = "\n".join(f"{p['name']}: {p['part']}" for p in found) or "No parts built yet."
    return ToolResult(content=[_text(text)], structured_content={"parts": found})


@mcp.tool(
    name="part_examples",
    title="Part Examples",
    description=(
        "Worked build123d scripts to start from: naca_wing_section, ahmed_car_body, "
        "cylinder, pipe_bend. Give name for one script; empty lists them all."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cad", "build123d"},
)
async def part_examples_tool(
    name: Annotated[str, Field(description="An example name; empty for all.")] = "",
) -> ToolResult:
    if name and name not in EXAMPLES:
        raise ToolError(f"No example {name}; there are {', '.join(EXAMPLES)}. [unknown_example]")
    chosen = {name: EXAMPLES[name]} if name else EXAMPLES
    text = "\n\n".join(f"# {key}: {ex['description']}\n{ex['code']}" for key, ex in chosen.items())
    return ToolResult(content=[_text(text)], structured_content={"examples": chosen})


@mcp.resource(
    VIEW_URI,
    name="cad_part",
    description="An interactive 3D view of a part (three.js).",
    mime_type="text/html;profile=mcp-app",
    app=AppConfig(
        csp=ResourceCSP(resource_domains=[THREE_ORIGIN], connect_domains=[THREE_ORIGIN]),
        prefers_border=False,
    ),
)
def view_resource() -> str:
    return resources.files("cad_mcp").joinpath("view.html").read_text("utf-8")


@mcp.resource("cad://examples", mime_type="application/json")
def examples_resource() -> dict[str, Any]:
    """Worked build123d scripts."""
    return EXAMPLES


@mcp.resource("cad://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the CAD MCP server can do."""
    return {
        "tools": ["build_part", "show_part", "list_parts", "part_examples"],
        "kernel": "build123d (OpenCascade)",
        "exports": ["STL (binary)", "STEP"],
        "view": VIEW_URI,
    }


def main() -> None:
    """Entry point for the CAD MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
