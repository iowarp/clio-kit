"""The tools that read the scene and write out what it shows: the picked point,
measurements, captures, exports and the scene's description.

They register on the server's ``mcp`` where server.py imports this module, so
the tools keep their order.
"""

from __future__ import annotations

import base64
from typing import Annotated, Any

from fastmcp import Context
from fastmcp.tools import ToolResult
from mcp.types import ImageContent
from pydantic import Field

from . import present
from .server import _paths, _text, attacher, mcp, run


@mcp.tool(annotations={"title": "Read the picked point", "readOnlyHint": True})
async def inspect_selection(
    ctx: Context,
    position: Annotated[
        list[float] | None,
        Field(
            min_length=3,
            max_length=3,
            description="Probe here instead of what the person picked.",
        ),
    ] = None,
) -> str:
    """Exact field values at the point the person picked in the view."""
    result, _ = await run(ctx, "inspect_selection", {"position": position})
    if result.get("selection") is None:
        return result.get("message") or "Nothing is at that position."
    return present.selection_line(result["selection"])


@mcp.tool(annotations={"title": "Measure a field", "readOnlyHint": True})
async def measure_field(
    ctx: Context,
    field: Annotated[str, Field(description="Field (array) name.")],
    object: Annotated[
        str | None, Field(description="Scene object id or label; default: the dataset.")
    ] = None,
    bins: Annotated[int, Field(ge=1, le=256)] = 16,
    component: Annotated[
        int | None, Field(description="Vector component; default: magnitude.")
    ] = None,
) -> ToolResult:
    """Exact statistics and a histogram of a field at the current time step."""
    result, _ = await run(
        ctx,
        "measure_field",
        {"field": field, "object": object, "bins": bins, "component": component},
    )
    return ToolResult(
        content=[_text(present.measure_text(result))], structured_content=result
    )


@mcp.tool(annotations={"title": "Measure a field over time", "readOnlyHint": True})
async def measure_over_time(
    ctx: Context,
    field: Annotated[str, Field(description="Field (array) name.")],
    object: Annotated[
        str | None, Field(description="Scene object id or label; default: the dataset.")
    ] = None,
    component: Annotated[
        int | None, Field(description="Vector component; default: magnitude.")
    ] = None,
    start: Annotated[int | None, Field(ge=0, description="First step index.")] = None,
    end: Annotated[int | None, Field(ge=0, description="Last step index.")] = None,
    stride: Annotated[int, Field(ge=1, description="Measure every nth step.")] = 1,
) -> ToolResult:
    """Exact statistics (min, max, mean, std, median) of a field at each time step: the
    series to plot or compare over a run. The view stays on its step."""
    result, _ = await run(
        ctx,
        "measure_over_time",
        {
            "field": field,
            "object": object,
            "component": component,
            "start": start,
            "end": end,
            "stride": stride,
        },
    )
    return ToolResult(
        content=[_text(present.series_text(result))], structured_content=result
    )


@mcp.tool(annotations={"title": "Capture the view", "readOnlyHint": True})
async def capture(
    ctx: Context,
    width: Annotated[int, Field(ge=64, le=4096)] = 1280,
    height: Annotated[int, Field(ge=64, le=4096)] = 800,
) -> list[Any]:
    """A PNG rendered by ParaView at the camera the view last reported. The text
    says when the frame is blank, so look before describing it."""
    result, _ = await run(ctx, "capture", {"width": width, "height": height})
    return [
        _text(
            present.blank_warning(result.get("drawn"))
            + f"ParaView render, {width}x{height}, saved on the ParaView machine at "
            f"{result['path']}. "
            + present.camera_text(result["camera"])
            + present.capture_note(result.get("seconds"), result.get("view") or {})
            + present.drawn_text(result.get("drawn"), result.get("shown"))
        ),
        ImageContent(type="image", data=result["png_base64"], mime_type="image/png"),
    ]


@mcp.tool(annotations={"title": "Export a figure"}, meta=_paths(path="write"))
async def export_artifact(
    ctx: Context,
    path: Annotated[
        str | None,
        Field(description="Where to write the PNG on the ParaView machine."),
    ] = None,
    width: Annotated[int, Field(ge=64, le=8192)] = 1920,
    height: Annotated[int, Field(ge=64, le=8192)] = 1080,
    transparent: bool = False,
) -> list[Any]:
    """Write a figure of the current view as a PNG file."""
    result, _ = await run(
        ctx,
        "export_artifact",
        {"path": path, "width": width, "height": height, "transparent": transparent},
    )
    size = len(base64.b64decode(result["png_base64"]))
    return [
        _text(
            f"Wrote {result['path']} ({size} bytes, {width}x{height}) on the ParaView machine."
        ),
        ImageContent(type="image", data=result["png_base64"], mime_type="image/png"),
    ]


@mcp.tool(annotations={"title": "Export an animation"}, meta=_paths(path="write"))
async def export_animation(
    ctx: Context,
    path: Annotated[
        str,
        Field(
            description="Where to write the video on the ParaView machine: in the person's "
            "workspace when it is to be shared or put on a dashboard. The extension is set "
            "by the encoder (.mp4, else .ogv)."
        ),
    ],
    width: Annotated[int, Field(ge=64, le=3840)] = 1280,
    height: Annotated[int, Field(ge=64, le=3840)] = 720,
    fps: Annotated[
        float, Field(gt=0, le=60, description="Frames (time steps) per second.")
    ] = 12,
    start: Annotated[
        int | None, Field(ge=0, description="First time step index.")
    ] = None,
    end: Annotated[int | None, Field(ge=0, description="Last time step index.")] = None,
    labels: Annotated[
        list[str] | None,
        Field(
            description="One readable label per time step of the scene (e.g. its date and "
            "hour), stamped in the corner of each frame. Omit for no stamp."
        ),
    ] = None,
) -> str:
    """Write a video of the scene's time steps as the live view shows them: rendered in the
    scene's own view, at its camera, colours and filters. Use it for a clip to send around,
    and put it beside a live view on a dashboard so the dashboard still shows the motion
    when ParaView is not running."""
    result, _ = await run(
        ctx,
        "export_animation",
        {
            "path": path,
            "width": width,
            "height": height,
            "fps": fps,
            "start": start,
            "end": end,
            "labels": labels,
        },
    )
    return present.animation_text(result)


@mcp.tool(annotations={"title": "Export the scene"}, meta=_paths(path="write"))
async def export_scene(
    ctx: Context,
    path: Annotated[
        str | None,
        Field(description="Where to write the manifest JSON on the ParaView machine."),
    ] = None,
) -> ToolResult:
    """Write the scene manifest, and a ParaView state file beside it."""
    result, _ = await run(ctx, "export_scene", {"path": path})
    return ToolResult(
        content=[
            _text(
                f"Scene manifest: {result['manifest_path']}. ParaView state file: "
                f"{result['state_path']}."
            )
        ],
        structured_content={"manifest": result["manifest"]},
    )


@mcp.tool(annotations={"title": "Describe the ParaView scene", "readOnlyHint": True})
async def scene_state(ctx: Context) -> ToolResult:
    """Where ParaView is attached, the dataset, the pipeline, time and view."""
    result, _ = await run(ctx, "scene_state", {})
    attachment = attacher().describe()
    lines = [
        f"Attached {attachment['where']}: ParaView {attachment.get('paraview')} "
        f"({attachment.get('pvpython')}), folder {attachment.get('folder')}"
        + (
            f", connected to the ParaView session at {attachment['pvserver_session']}"
            if attachment.get("pvserver_session")
            else ""
        )
        + "."
    ]
    timings = attachment.get("attach_seconds") or {}
    if timings:
        lines.append(
            f"The last attach took {timings.get('total', 0):g} s"
            + (
                f" and installed trame (from {attachment.get('environment_built_from')})"
                if attachment.get("built_this_attach")
                else "; the prepared folder was reused"
            )
            + "."
        )
    if result.get("scene") is None:
        known = ", ".join(result.get("known_scenes") or []) or "none"
        lines.append(f"No scene is open. Saved scenes: {known}.")
    else:
        lines.append(present.scene_text(result, attachment["where"]))
        lines.extend(present.changes_lines(result.get("changes") or []))
    return ToolResult(
        content=[_text("\n".join(lines))],
        structured_content={"attachment": attachment, "scene": result},
    )
