"""ParaView MCP server: the front.

The front is plain Python and speaks MCP over stdio. It lists its tools with
no ParaView installed. The first scene tool attaches to a node half that runs
in ParaView's own Python, on this machine or over one SSH connection, and
every scene tool is passed to it as one command.
"""

from __future__ import annotations

import asyncio
import atexit
import base64
from importlib import resources
from typing import Annotated, Any, Literal

from fastmcp import Context, FastMCP
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from fastmcp.utilities.tasks import TaskConfig
from fastmcp_tasks import TasksExtension
from mcp.types import ImageContent, TextContent
from pydantic import Field

from . import follow, present
from .attach import Attacher
from .config import Target, from_environment
from .descriptor import ViewDescriptor
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
    {"key": "newer_steps", "description": "Steps newer than the shown one."},
    {
        "key": "catching_up",
        "description": "Whether a source opened late is playing from its first step "
        "to the latest before following it.",
    },
    {"key": "growing", "description": "Whether the source is still being written."},
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
    version="3.3.0",  # keep equal to mcp-server-versions.toml and SERVER_VERSION
    instructions=(
        "Open, look at and measure 3D scientific data in ParaView, on this machine "
        "or on another one over SSH, and show it to the person in a live view. "
        "open_view opens a dataset (paths are on the machine where ParaView runs) "
        "and returns a scene id; keep it in later open_view calls to reopen the "
        "same scene. Read numbers with measure_field and inspect_selection, and "
        "look with capture, before making a claim. For a source still being written "
        "(a running simulation), pass growing in open_view's view descriptor; the view "
        "follows the newest step, and watch_steps reports each new step as it is "
        "written. Where ParaView runs is the connector's setting, not a tool."
    ),
)
# Background tasks (the MCP tasks extension) for watch_steps; a client that does not
# negotiate them calls every tool directly, as before.
mcp.add_extension(TasksExtension())

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


def command(name: str, args: dict[str, Any], timeout: float = 900.0) -> tuple[Any, Any]:
    """Run one scene command on the node half; return (result, link)."""
    clean = {k: v for k, v in args.items() if v is not None}
    try:
        link = attacher().ensure()
        return link.client.command(name, clean, timeout=timeout, by=actor()), link
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
    settings = attacher().settings
    if settings.pvserver:
        return f"on the ParaView server at {settings.pvserver}"
    return settings.target.describe()


def _text(text: str) -> TextContent:
    return TextContent(type="text", text=text)


def open_arguments(
    path: str | None,
    scene: str | None,
    growing: bool,
    new: bool,
    view: ViewDescriptor | None,
) -> dict[str, Any]:
    """The node's open_view arguments, from the plain arguments and a view descriptor."""
    args: dict[str, Any] = {
        "path": path,
        "scene": scene,
        "growing": growing,
        "new": new,
    }
    if view is None:
        return args
    if path is not None and path != view.path:
        raise ToolError(
            f"open_view got path {path} and a view descriptor for {view.path}; give "
            "one of them. [bad_arguments]"
        )
    args.update(
        path=view.path,
        growing=growing or view.growing,
        format=view.format,
        reader=view.reader,
        reader_properties=view.reader_properties or None,
        descriptor=view.extras(),
    )
    return args


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
    view: Annotated[
        ViewDescriptor | None,
        Field(
            description="A view descriptor: the path or pattern, the format, the "
            "ParaView reader and its properties, and whether the source is still "
            "being written. Use it instead of path when a run reported one."
        ),
    ] = None,
) -> ToolResult:
    """Open a dataset or restore a saved scene, and show it in a live ParaView view.

    With neither path nor scene, it shows the ParaView session already running
    where ParaView is, or the current scene, or else an empty scene that
    run_scene_script can fill with sources. A growing source is followed: the view
    moves to each new step unless the person has parked it on one. A growing source
    that cannot be read yet (not there yet, or an ADIOS2 BP5 folder before its first
    whole step) is waited for while it changes on disk, so open it as soon as the
    run starts.
    """
    summary, link = command(
        "open_view", open_arguments(path, scene, growing, new, view)
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


@mcp.tool(annotations={"title": "Build the ParaView scene"})
def run_scene_script(
    code: Annotated[
        str,
        Field(
            description="Python using paraview.simple, run inside the open scene. It sees "
            "`simple` (and its names, e.g. Contour, OpenDataFile, Show, ColorBy), `view` "
            "(the scene's render view: show things in it, never make another), "
            "`objects` (the scene's objects by id or label, e.g. objects['o1']) and "
            "`folder` (the scene's folder, for files it writes)."
        ),
    ],
    label: Annotated[
        str | None, Field(description="A short name for this step, e.g. 'isobars'.")
    ] = None,
) -> str:
    """Build or change the scene with a ParaView script: readers (a second dataset too),
    filters, colours, anything paraview.simple can do. It runs in the scene the person
    sees, so the live view, captures and the saved scene include what it makes. It applies
    whole or not at all; a failing script leaves nothing behind. The step is saved with the
    scene and replayed when the scene is restored. Names are those of the ParaView that
    runs it: ParaView 6 colour presets drop " (matplotlib)" (either form works), and a
    scalar bar's range labels are AddRangeLabels."""
    result, _ = command("run_scene_script", {"code": code, "label": label})
    return present.script_text(result, _where())


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


@mcp.tool(
    task=TaskConfig(mode="optional"),
    annotations={"title": "Watch new time steps", "readOnlyHint": True},
)
async def watch_steps(
    ctx: Context,
    idle_seconds: Annotated[
        float,
        Field(ge=5, le=86400, description="Stop after this long with no new step."),
    ] = 300.0,
    max_seconds: Annotated[
        float | None,
        Field(ge=1, le=604800, description="Stop after this long in any case."),
    ] = None,
    after: Annotated[
        int | None,
        Field(
            ge=0,
            description="Report steps from this index on; default: the steps written "
            "from now.",
        ),
    ] = None,
) -> ToolResult:
    """Report each new time step of the open scene's growing source as it is written,
    as progress: run as a background task, each step is the task's status; called
    directly, each step is a progress notification and the call lasts until it ends,
    so give max_seconds. Progress counts the steps written, against the view
    descriptor's expected_steps when it gives them. It ends once those are written,
    when no step comes for idle_seconds, after max_seconds, or when the source is
    not growing, and returns every step it saw. The view follows the newest step
    unless the person parked it."""

    async def fetch(index: int) -> dict[str, Any]:
        result, _ = await asyncio.to_thread(command, "steps", {"after": index})
        return result

    async def report(written: int, expected: int | None, message: str) -> None:
        await ctx.report_progress(written, expected, message)

    outcome = await follow.watch(
        fetch,
        report,
        after=after,
        idle_seconds=idle_seconds,
        max_seconds=max_seconds,
    )
    return ToolResult(
        content=[_text(follow.watch_text(outcome))], structured_content=outcome
    )


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


@mcp.tool(annotations={"title": "Measure a field over time", "readOnlyHint": True})
def measure_over_time(
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
    result, _ = command(
        "measure_over_time",
        {
            "field": field,
            "object": object,
            "component": component,
            "start": start,
            "end": end,
            "stride": stride,
        },
        timeout=3600.0,
    )
    return ToolResult(
        content=[_text(present.series_text(result))], structured_content=result
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


@mcp.tool(annotations={"title": "Export an animation"}, meta=_paths(path="write"))
def export_animation(
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
    result, _ = command(
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
        timeout=3600.0,
    )
    return present.animation_text(result)


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


def _warm_client() -> None:
    """Start fetching the ParaView a joined server needs, when this machine lacks it,
    so the first attach waits less."""
    import shutil

    from . import client, native
    from .errors import ParaViewError

    settings = from_environment()
    want = settings.version
    if not want or settings.target.remote or client.installed(want):
        return
    found = shutil.which("pvpython")
    if found and native.version_of(found) == want:
        return
    try:
        client.start(want)
    except ParaViewError:
        pass  # the attach reports it


def main() -> None:
    import threading

    threading.Thread(target=_warm_client, daemon=True).start()
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
