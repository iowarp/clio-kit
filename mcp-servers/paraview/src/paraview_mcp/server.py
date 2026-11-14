"""ParaView MCP server: the front.

The front is plain Python and speaks MCP over stdio. It lists its tools with
no ParaView installed. The first scene tool attaches to a node half that runs
in ParaView's own Python, on this machine or over one SSH connection, and
every scene tool is passed to it as one command.
"""

from __future__ import annotations

import atexit
import base64
from importlib import resources
from typing import Annotated, Any, Literal

from fastmcp import FastMCP
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import ImageContent, TextContent
from pydantic import Field

from . import present
from .attach import Attacher
from .config import Target, from_environment
from .errors import ParaViewError
from .link import LinkError

VIEW_URI = "ui://paraview/view"
SERVICE_META = "ai.iowarp.clio/service"
# What a host replays to reopen the view: the scene, never the call. A call with
# new=true started an empty scene; replaying it would wipe what was built since.
REOPEN_META = "ai.iowarp.clio/reopen"
PATHS_META = "ai.iowarp.clio/paths"
#: Set by the host on a call a view or a person made, not the agent ("view").
ORIGIN_META = "ai.iowarp.clio/origin"
#: What the view reports in ui/update-model-context, for a host linking it to a dashboard.
STATE_META = "ai.iowarp.clio/state"
VIEW_STATE = [
    {"key": "time", "tool": "set_timestep", "description": "The shown time value."},
    {"key": "time_index", "description": "The shown step, from 0."},
    {"key": "time_steps", "description": "How many steps the data has."},
    {"key": "latest_time", "description": "The newest time value written."},
    {
        "key": "follow_latest",
        "description": "Whether the view follows the newest step.",
    },
    {
        "key": "selection",
        "description": "The point the person picked, with its values.",
    },
    {"key": "scene", "description": "The scene id."},
    {"key": "dataset", "description": "The open dataset."},
    {"key": "active_object", "description": "The object the panel shows."},
    {"key": "view_mode", "description": "Local or remote rendering."},
]

mcp: FastMCP = FastMCP(
    "paraview",
    version="3.1.0",  # keep equal to mcp-server-versions.toml and SERVER_VERSION
    instructions=(
        "Open, look at and measure 3D scientific data in ParaView, on this machine "
        "or on another one over SSH, and show it to the person in a live view. "
        "open_view opens a dataset (paths are on the machine where ParaView runs) "
        "and returns a scene id; keep it in later open_view calls to reopen the "
        "same scene. Read numbers with measure_field and inspect_selection, and "
        "look with capture, before making a claim. Where ParaView runs is the "
        "connector's setting, not a tool."
    ),
)

_attacher: Attacher | None = None


def attacher() -> Attacher:
    """The process's one attacher, built from the connector's environment."""
    global _attacher
    if _attacher is None:
        _attacher = Attacher(from_environment())
        atexit.register(_attacher.close)
    return _attacher


def _fail(exc: ParaViewError) -> ToolError:
    return ToolError(f"{exc.message} [{exc.code}]")


def actor() -> str:
    """Who asked for this call: ``person`` when the host marks it as a view's, else ``agent``.

    CLIO marks a call a view makes (the person's click, a dashboard's linked chart) with
    ``_meta["ai.iowarp.clio/origin"] = "view"``; every other call is the agent's.
    """
    from fastmcp.server.dependencies import fastmcp_request_ctx

    request = fastmcp_request_ctx.get()
    meta = getattr(request, "meta", None) if request is not None else None
    origin = meta.get(ORIGIN_META) if isinstance(meta, dict) else None
    return "person" if origin in ("view", "person") else "agent"


def command(name: str, args: dict[str, Any]) -> tuple[Any, Any]:
    """Run one scene command on the node half; return (result, link)."""
    clean = {k: v for k, v in args.items() if v is not None}
    try:
        link = attacher().ensure()
        return link.client.command(name, clean, by=actor()), link
    except ParaViewError as exc:
        raise _fail(exc) from exc
    except LinkError as exc:
        raise ToolError(
            "The link to ParaView dropped during this call; the scene is kept on "
            "the node. Call scene_state to see whether the change applied, then "
            f"retry. ({exc}) [link_lost]"
        ) from exc


def path_arguments(target: Target, **access: str) -> dict[str, Any]:
    """Declare which arguments are paths, for the tool definition's ``_meta``.

    Paths are on the machine where ParaView runs: CLIO's own machine, or the SSH host.
    CLIO restarts the server when the connector's address changes, so the declaration
    listed with the tools always names the current one.
    """
    if target.remote:
        where: dict[str, str] = {"machine": "remote", "host": target.address or ""}
    else:
        where = {"machine": "local"}
    return {PATHS_META: {arg: {**where, "access": use} for arg, use in access.items()}}


def _paths(**access: str) -> dict[str, Any]:
    return path_arguments(from_environment().target, **access)


def _where() -> str:
    return attacher().settings.target.describe()


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


@mcp.tool(
    app=AppConfig(resource_uri=VIEW_URI),
    annotations={"title": "Open a ParaView view", "openWorldHint": True},
    meta={**_paths(path="read"), STATE_META: VIEW_STATE},
)
def open_view(
    path: Annotated[
        str | None,
        Field(
            description="Dataset path on the machine where ParaView runs. A glob "
            "such as /data/run/step_*.vti opens a file series."
        ),
    ] = None,
    scene: Annotated[
        str | None,
        Field(description="A scene id from an earlier open_view, to restore it."),
    ] = None,
    growing: Annotated[
        bool, Field(description="The source is still being written; follow it.")
    ] = False,
    new: Annotated[
        bool, Field(description="Start a fresh scene even if one is saved.")
    ] = False,
) -> ToolResult:
    """Open a dataset or restore a saved scene, and show it in a live ParaView view.

    With neither path nor scene, it shows the ParaView session already running
    where ParaView is, or the current scene.
    """
    summary, link = command(
        "open_view", {"path": path, "scene": scene, "growing": growing, "new": new}
    )
    text = present.scene_text(summary, _where())
    return ToolResult(
        content=[_text(text)],
        structured_content={"scene": summary["scene"], "summary": summary},
        meta={
            SERVICE_META: {"port": link.local_port, "token": link.token},
            REOPEN_META: {"arguments": {"scene": summary["scene"]}},
        },
    )


@mcp.tool(annotations={"title": "Add a ParaView filter"})
def create_filter(
    filter: Annotated[
        str,
        Field(
            description="ParaView filter name, e.g. Contour, Slice, Clip, Threshold."
        ),
    ],
    input: Annotated[
        str | None,
        Field(description="Scene object id or label to filter; default: the last."),
    ] = None,
    properties: Annotated[
        dict[str, Any] | None,
        Field(
            description="Filter properties by ParaView name, e.g. "
            '{"ContourBy": ["POINTS", "p"], "Isosurfaces": [0.5]}. A dotted name '
            'sets a property of a part, e.g. {"SliceType.Normal": [0, 0, 1]}.'
        ),
    ] = None,
    label: Annotated[str | None, Field(description="A name for the object.")] = None,
    hide_input: Annotated[bool, Field(description="Hide the input object.")] = False,
) -> str:
    """Add one filter to the scene's pipeline. It commits whole or not at all."""
    result, _ = command(
        "create_filter",
        {
            "filter": filter,
            "input": input,
            "properties": properties,
            "label": label,
            "hide_input": hide_input,
        },
    )
    return "Added " + present.object_line(result["object"]) + "."


@mcp.tool(annotations={"title": "Change a ParaView filter"})
def set_filter_property(
    object: Annotated[str, Field(description="Scene object id or label of a filter.")],
    properties: Annotated[
        dict[str, Any],
        Field(
            description="Filter properties by ParaView name, e.g. "
            '{"Isosurfaces": [0.0003]} for a Contour, or {"SliceType.Origin": [0, 0, 5]}.'
        ),
    ],
) -> str:
    """Change settings of a filter already in the scene. It commits whole or not at all."""
    result, _ = command(
        "set_filter_property", {"object": object, "properties": properties}
    )
    before = result.get("before") or {}
    changed = ", ".join(
        f"{key} {before.get(key)!r} -> {value!r}" for key, value in properties.items()
    )
    return "Changed " + present.object_line(result["object"]) + f": {changed}."


@mcp.tool(annotations={"title": "Remove a scene object", "destructiveHint": True})
def remove_scene_object(
    object: Annotated[str, Field(description="Scene object id or label.")],
) -> str:
    """Remove an object and everything built on it from the pipeline."""
    result, _ = command("remove_scene_object", {"object": object})
    return f"Removed {', '.join(result['removed'])}."


@mcp.tool(annotations={"title": "Change how an object is shown"})
def set_representation(
    object: Annotated[str, Field(description="Scene object id or label.")],
    representation: Annotated[
        str | None,
        Field(
            description="Surface, Surface With Edges, Wireframe, Points, Outline, Volume."
        ),
    ] = None,
    color_by: Annotated[
        str | None, Field(description="Field to colour by, or 'none' for solid.")
    ] = None,
    colormap: Annotated[
        str | None, Field(description="ParaView colour preset, e.g. Viridis.")
    ] = None,
    opacity: Annotated[float | None, Field(ge=0, le=1)] = None,
    visible: Annotated[bool | None, Field(description="Show or hide it.")] = None,
    component: Annotated[
        int | Literal["Magnitude"] | None,
        Field(
            description="Colour by this component of a vector field (0 is X), or "
            "Magnitude."
        ),
    ] = None,
    scalar_bar: Annotated[
        bool | None, Field(description="Show or hide the colour legend.")
    ] = None,
    rescale: Annotated[
        bool | None,
        Field(description="Rescale the colour range to the data at this time step."),
    ] = None,
    color_range: Annotated[
        list[float] | None,
        Field(
            min_length=2,
            max_length=2,
            description="Set the colour range by hand, [low, high], for the field this "
            "object is coloured by. Values beyond it take the end colours. For a volume "
            "it also sets the opacity ramp: low is clear, high is opaque.",
        ),
    ] = None,
) -> str:
    """Change an object's representation, colouring, opacity or visibility."""
    result, _ = command(
        "set_representation",
        {
            "object": object,
            "representation": representation,
            "color_by": color_by,
            "colormap": colormap,
            "opacity": opacity,
            "visible": visible,
            "component": component,
            "scalar_bar": scalar_bar,
            "rescale": rescale,
            "color_range": color_range,
        },
    )
    return "Now " + present.object_line(result["object"]) + "."


@mcp.tool(annotations={"title": "Change the timestep"})
def set_timestep(
    index: Annotated[
        int | None, Field(description="Step index; -1 is the last.")
    ] = None,
    time: Annotated[
        float | None, Field(description="Time value; the nearest step.")
    ] = None,
    latest: Annotated[
        bool | None, Field(description="True to follow the latest step as it grows.")
    ] = None,
) -> str:
    """Show another time step, or follow the latest one."""
    result, _ = command(
        "set_timestep", {"index": index, "time": time, "latest": latest}
    )
    return present.time_line(result["time"])


@mcp.tool(annotations={"title": "Move the camera"})
def set_camera(
    position: Annotated[list[float] | None, Field(min_length=3, max_length=3)] = None,
    focal_point: Annotated[
        list[float] | None, Field(min_length=3, max_length=3)
    ] = None,
    view_up: Annotated[list[float] | None, Field(min_length=3, max_length=3)] = None,
    azimuth: Annotated[float, Field(description="Degrees around the view up.")] = 0.0,
    elevation: Annotated[float, Field(description="Degrees up or down.")] = 0.0,
    zoom: Annotated[float, Field(gt=0, description=">1 moves closer.")] = 1.0,
    direction: Annotated[
        Literal["+x", "-x", "+y", "-y", "+z", "-z"] | None,
        Field(description="Look along this axis at the scene, fitted, first."),
    ] = None,
    parallel_projection: Annotated[
        bool | None, Field(description="True for parallel, False for perspective.")
    ] = None,
) -> str:
    """Place the camera, or turn and zoom it from where it is."""
    result, _ = command(
        "set_camera",
        {
            "position": position,
            "focal_point": focal_point,
            "view_up": view_up,
            "azimuth": azimuth,
            "elevation": elevation,
            "zoom": zoom,
            "direction": direction,
            "parallel_projection": parallel_projection,
        },
    )
    return present.camera_text(result["camera"])


@mcp.tool(annotations={"title": "Fit the camera"})
def fit_camera(
    object: Annotated[
        str | None, Field(description="Fit this object; default: everything.")
    ] = None,
) -> str:
    """Point the camera so everything, or one object, fills the view."""
    result, _ = command("fit_camera", {"object": object})
    return present.camera_text(result["camera"])


@mcp.tool(annotations={"title": "Read the picked point", "readOnlyHint": True})
def inspect_selection(
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
    result, _ = command("inspect_selection", {"position": position})
    if result.get("selection") is None:
        return result.get("message") or "Nothing is at that position."
    return present.selection_line(result["selection"])


@mcp.tool(annotations={"title": "Measure a field", "readOnlyHint": True})
def measure_field(
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
    result, _ = command(
        "measure_field",
        {"field": field, "object": object, "bins": bins, "component": component},
    )
    return ToolResult(
        content=[_text(present.measure_text(result))], structured_content=result
    )


@mcp.tool(annotations={"title": "Capture the view", "readOnlyHint": True})
def capture(
    width: Annotated[int, Field(ge=64, le=4096)] = 1280,
    height: Annotated[int, Field(ge=64, le=4096)] = 800,
) -> list[Any]:
    """A PNG rendered by ParaView at the camera the view last reported."""
    result, _ = command("capture", {"width": width, "height": height})
    return [
        _text(
            f"ParaView render, {width}x{height}, saved on the ParaView machine at "
            f"{result['path']}. "
            + present.camera_text(result["camera"])
            + present.capture_note(result.get("seconds"), result.get("view") or {})
        ),
        ImageContent(type="image", data=result["png_base64"], mime_type="image/png"),
    ]


@mcp.tool(annotations={"title": "Export a figure"}, meta=_paths(path="write"))
def export_artifact(
    path: Annotated[
        str | None,
        Field(description="Where to write the PNG on the ParaView machine."),
    ] = None,
    width: Annotated[int, Field(ge=64, le=8192)] = 1920,
    height: Annotated[int, Field(ge=64, le=8192)] = 1080,
    transparent: bool = False,
) -> list[Any]:
    """Write a figure of the current view as a PNG file."""
    result, _ = command(
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


@mcp.tool(annotations={"title": "Export the scene"}, meta=_paths(path="write"))
def export_scene(
    path: Annotated[
        str | None,
        Field(description="Where to write the manifest JSON on the ParaView machine."),
    ] = None,
) -> ToolResult:
    """Write the scene manifest, and a ParaView state file beside it."""
    result, _ = command("export_scene", {"path": path})
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
def scene_state() -> ToolResult:
    """Where ParaView is attached, the dataset, the pipeline, time and view."""
    result, _ = command("scene_state", {})
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


@mcp.resource(
    VIEW_URI,
    name="paraview_view",
    description="The live ParaView view: trame, framed through the host's service route.",
    mime_type="text/html;profile=mcp-app",
    app=AppConfig(prefers_border=False),
)
def view_resource() -> str:
    """The view page."""
    return resources.files("paraview_mcp").joinpath("view.html").read_text("utf-8")


def main() -> None:
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
