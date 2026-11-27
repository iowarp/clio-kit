"""ParaView MCP server: the front.

The front is plain Python and speaks MCP over stdio. It lists its tools with
no ParaView installed. The first scene tool attaches to a node half that runs
in ParaView's own Python, on this machine or over one SSH connection, and
every scene tool is passed to it as one command.
"""

from __future__ import annotations

import asyncio
import atexit
import threading
from collections.abc import Callable
from importlib import resources
from typing import Annotated, Any

from fastmcp import Context, FastMCP
from fastmcp.apps import AppConfig
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from fastmcp.tools import ToolResult
from fastmcp_tasks import TasksExtension
from mcp.types import TextContent
from pydantic import Field

from . import present
from .attach import Attacher
from .config import Target, from_environment
from .descriptor import ViewDescriptor
from .errors import ParaViewError
from .link import LinkError
from .node.manifest import SCENE_ID

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
    {"key": "link_state", "description": "Live, waiting, reconnecting or stopped."},
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
        "written. Each scene (a dataset, or one variable of it) has a view of its "
        "own; the other tools act on the view opened last, and open_view(scene=id) "
        "switches back to an open one. Where ParaView runs is the connector's "
        "setting, not a tool. When a run has a body or surface made as real "
        "geometry (an STL the person made, say), its descriptor's geometry draws "
        "that surface in the scene; a contour of a mask or indicator field shows "
        "only what the grid resolved, stair-stepped, and suits checking the mask "
        "against the surface. For a live view, choose what the run changes: a fixed "
        "plane through a region that settles soon looks still, while an isosurface "
        "of a vortex measure the run writes (vorticity, Q) or a plane across the "
        "wake downstream tends to show the run evolving. measure_over_time on the "
        "shown object says whether it still changes over recent steps."
    ),
)
# Background tasks (the MCP tasks extension) for watch_steps; a client that does not
# negotiate them calls every tool directly, as before.
mcp.add_extension(TasksExtension())

_attachers: dict[str, Attacher] = {}
#: The attacher the last open_view used; scene tools act on its view.
_current = ""
#: Which attacher each opened scene lives on.
_scene_keys: dict[str, str] = {}
PVSERVER_ADDRESS = r"^(connector|[A-Za-z0-9.\-\[\]]+:\d{1,5})$"
#: The pvserver value that opens a view where the connector points.
CONNECTOR = "connector"


def _key(pvserver: str | None) -> str:
    """The attacher key for a pvserver: empty for the connector's own setting."""
    if not pvserver or pvserver == (from_environment().pvserver or ""):
        return ""
    return pvserver


def attacher(key: str | None = None) -> Attacher:
    """The attacher for one pvserver (default: the current view's), built from the
    connector's environment. Each pvserver has its own node half, so a view opened
    on another pvserver stays beside the first one."""
    from dataclasses import replace

    key = _current if key is None else key
    found = _attachers.get(key)
    if found is None:
        settings = from_environment()
        if key:
            settings = replace(settings, pvserver=key)
        found = _attachers[key] = Attacher(settings)
        atexit.register(found.close)
    return found


def _hints(*, read_only: bool, destructive: bool, idempotent: bool) -> dict[str, bool]:
    """The three behaviour hints every tool declares, in MCP's spelling."""
    return {
        "readOnlyHint": read_only,
        "destructiveHint": destructive,
        "idempotentHint": idempotent,
    }


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


def command(
    name: str,
    args: dict[str, Any],
    opening: bool = False,
    key: str | None = None,
    stop: threading.Event | None = None,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[Any, Any]:
    """Run one scene command on the node half; return (result, link).

    Only ``opening`` (open_view) starts a node half where none runs. Any other
    command on a view whose node half has gone says why, instead of starting a
    fresh one that shows nothing. A command takes as long as its work: ``progress``
    hears how it goes, and ``stop`` cancels it.
    """
    clean = {k: v for k, v in args.items() if v is not None}

    def waiting(note: str) -> None:
        if progress is not None:
            progress({"note": note})

    try:
        link = attacher(key).ensure(opening=opening, stop=stop, progress=waiting)
        try:
            result = link.client.command(
                name, clean, by=actor(), stop=stop, progress=progress
            )
        except ParaViewError as exc:
            if exc.code != "pvserver_lost":
                raise
            # The pvserver connection closed under the call; the pvserver may run on
            # behind a rebuilt forward. Join it again with the same scenes and go on.
            link = attacher(key).rejoin(exc, stop=stop, progress=waiting)
            result = link.client.command(
                name, clean, by=actor(), stop=stop, progress=progress
            )
        return result, link
    except ParaViewError as exc:
        raise _fail(exc) from exc
    except LinkError as exc:
        if not exc.busy:
            # A node half ended under the call (its watchdog does, once its pvserver
            # has gone): say why, rather than asking for a retry.
            state = attacher(key).status()
            if state.get("state") == "lost":
                raise ToolError(f"{state['text']} [{state['code']}]") from exc
        raise ToolError(
            "The link to ParaView dropped during this call; the scene is kept on "
            "the node. Call scene_state to see whether the change applied, then "
            f"retry. ({exc}) [link_lost]"
        ) from exc


async def run(
    ctx: Context,
    name: str,
    args: dict[str, Any],
    opening: bool = False,
    key: str | None = None,
) -> tuple[Any, Any]:
    """``command`` for a tool: it lasts as long as the work does, each step or wait
    is reported as MCP progress (steps done of all of them where the command counts
    them, else the seconds so far), and cancelling the tool call cancels it."""
    loop = asyncio.get_running_loop()
    stop = threading.Event()
    began = loop.time()
    last = [0.0]
    #: Whether the command counts its steps; its progress is then those counts.
    counting = [False]

    def tell(value: float, total: float | None, message: str | None) -> None:
        # Progress must grow; a note that does not move it is left out.
        if value <= last[0]:
            return
        last[0] = value
        asyncio.run_coroutine_threadsafe(
            ctx.report_progress(value, total, message), loop
        )

    def progress(note: dict[str, Any]) -> None:
        done, total = note.get("done"), note.get("total")
        if isinstance(done, int) and isinstance(total, int) and total > 0:
            counting[0] = True
            tell(float(done), float(total), note.get("note"))
        elif not counting[0]:
            tell(round(loop.time() - began, 1), None, note.get("note"))

    chosen = attacher(key)
    if opening:
        chosen.cancel.clear()
        chosen.progress = lambda message: progress({"note": message})
    try:
        return await asyncio.to_thread(
            command, name, args, opening, key, stop, progress
        )
    except asyncio.CancelledError:
        stop.set()
        if opening:
            chosen.cancel.set()
        raise
    finally:
        if opening:
            chosen.progress = None


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
    return _where_for(_current)


def _where_for(key: str) -> str:
    settings = attacher(key).settings
    if settings.pvserver:
        return f"on the ParaView server at {settings.pvserver}"
    return settings.target.describe()


def _pvserver_ended(key: str) -> bool:
    """Whether the pvserver an attacher drew on has ended for good."""
    found = _attachers.get(key)
    if found is None or not key:
        return False
    state = found.status()
    return state.get("state") == "lost" and state.get("code") == "pvserver_lost"


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
    annotations={
        "title": "Open a ParaView view",
        # new=true starts an empty scene that replaces the saved one for that path.
        **_hints(read_only=False, destructive=True, idempotent=True),
        "openWorldHint": True,
    },
    meta={**_paths(path="read"), STATE_META: VIEW_STATE},
    tags={"paraview", "scene", "view"},
)
async def open_view(
    ctx: Context,
    path: Annotated[
        str | None,
        Field(
            description="Dataset path on the machine where ParaView runs. A glob "
            "such as /data/run/step_*.vti opens a file series."
        ),
    ] = None,
    scene: Annotated[
        str | None,
        Field(
            description="A scene id: letters, digits, '.', '_' and '-', e.g. wrf-t2. "
            "One from an earlier open_view restores that "
            "scene, or, while its view is open, makes it the view later tools act "
            "on. A new id opens a scene of its own, e.g. one per variable.",
            pattern=SCENE_ID,
        ),
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
    pvserver: Annotated[
        str | None,
        Field(
            pattern=PVSERVER_ADDRESS,
            description="host:port of a running pvserver to draw this view on, as a "
            "run reported it, or 'connector' for where the connector points (this "
            "machine when it names no pvserver), e.g. for a file copied here. "
            "Default: the pvserver of the view opened last, or the connector once "
            "that pvserver has ended. A view on another pvserver opens beside the "
            "ones already open; they stay open.",
        ),
    ] = None,
) -> ToolResult:
    """Open a dataset or restore a saved scene, and show it in a live ParaView view.

    With neither path nor scene, it shows the ParaView session already running
    where ParaView is, or the current scene, or else an empty scene that
    run_scene_script can fill with sources. A growing source is followed: the view
    moves to each new step unless the person has parked it on one. A growing source
    that cannot be read yet (not there yet, or no file of its pattern written whole)
    is waited for while it changes on disk, so open it as soon as the
    run starts. Starting ParaView is reported as progress and can be cancelled; it
    fails only when its start stops making progress. The result says how much of
    a test frame the view really draws.

    Each scene has a view of its own beside the others: another dataset, or the same
    one coloured by another variable (the descriptor's display array), opens a new
    view and leaves the open ones as they are, each at its own step. The other tools
    act on the view opened last; open_view(scene=id) of an open scene makes its view
    the one they act on again. A view is open once: opening a scene that is already
    open opens nothing new, brings its view forward and says already_open. To read a
    view's step or state, call scene_state(scene=id), never open_view.
    """
    global _current
    if pvserver == CONNECTOR:
        key = ""
    elif pvserver:
        key = _key(pvserver)
    else:
        key = _scene_keys.get(scene or "", _current)
    moved = ""
    if not pvserver and key and (scene or "") not in _scene_keys:
        # A new scene with no pvserver named follows the current view's pvserver,
        # unless that pvserver has ended: then it opens where the connector says.
        ended = await asyncio.to_thread(_pvserver_ended, key)
        if ended:
            moved = (
                f"The pvserver at {key} has ended, so this view opens where the "
                f"connector points instead: {_where_for('')}.\n"
            )
            key = ""
    chosen = attacher(key)
    summary, link = await run(
        ctx,
        "open_view",
        open_arguments(path, scene, growing, new, view),
        opening=True,
        key=key,
    )
    _current = key
    _scene_keys[summary["scene"]] = key
    chosen.scene = summary["scene"]
    text = (
        moved
        + present.scene_text(summary, _where())
        + present.drawn_text(summary.get("drawn"), summary.get("shown"))
    )
    if summary.get("already_open"):
        text = (
            f"Scene {summary['scene']} is already open in its view; that view is "
            "shown again and no new view was opened. Use scene_state(scene=...) to "
            "read a view without opening it.\n" + text
        )
    views = present.views_lines(summary.get("views") or [])
    if len(views) > 2:
        text += "\n" + "\n".join(views)
    reopen: dict[str, Any] = {"scene": summary["scene"]}
    if key:
        reopen["pvserver"] = key
    # Each open scene has a port of its own on the node half, so views of different
    # scenes stay open side by side.
    try:
        port = await asyncio.to_thread(
            link.port_for, (summary.get("service") or {}).get("port")
        )
    except ParaViewError as exc:
        raise _fail(exc) from exc
    chosen.remember(summary["scene"], (summary.get("service") or {}).get("port"), port)
    return ToolResult(
        content=[_text(text)],
        structured_content={
            "scene": summary["scene"],
            "already_open": bool(summary.get("already_open")),
            "summary": summary,
        },
        meta={
            SERVICE_META: {"port": port, "token": link.token},
            REOPEN_META: {"arguments": reopen},
        },
    )


@mcp.tool(
    annotations={
        "title": "Build the ParaView scene",
        # Any paraview.simple code: it can delete objects or write files.
        **_hints(read_only=False, destructive=True, idempotent=False),
    },
    tags={"paraview", "scene", "script"},
)
async def run_scene_script(
    ctx: Context,
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
    scalar bar's range labels are AddRangeLabels. A field's colour range defaults to the
    2nd to 98th percentile of the values shown, so extremes do not wash out the rest; a
    range the script sets (RescaleTransferFunction, e.g. from measure_field) is kept.

    A body's real surface reads better than a contour of its mask: the mask shows what
    the grid resolved, in steps of one cell. When the run made the body as geometry (an
    STL, say), open that file here, or pass it in open_view's descriptor geometry, and
    keep a mask contour for checking the mask against it. For a view that shows a run
    evolving, a contour of a vortex measure the run writes (vorticity, Q) or a slice
    across the wake downstream usually changes more than a fixed plane through a region
    that settles; measure_over_time on the shown object says whether it still does."""
    result, _ = await run(ctx, "run_scene_script", {"code": code, "label": label})
    return present.script_text(result, _where())


# The tools that move through time and the camera, then those that read the scene
# and write out what it shows; importing them here registers them in this place in
# the tool list.
from . import controls, outputs  # noqa: F401


@mcp.tool(
    app=AppConfig(visibility=["app"]),
    annotations={
        "title": "Check the live view's ParaView",
        **_hints(read_only=True, destructive=False, idempotent=True),
    },
    tags={"paraview", "view"},
)
def view_status(
    scene: Annotated[
        str | None, Field(description="The scene id the view shows.")
    ] = None,
) -> ToolResult:
    """For the live view: whether the ParaView behind it still runs, and if it went,
    why. A view whose page cannot reach ParaView asks this, so it shows the reason
    instead of drawing nothing. A node half that went while its pvserver runs is
    started again with its scenes; this says each try."""
    key = _scene_keys.get(scene or "", _current)
    status = (
        _attachers[key].status(scene)
        if key in _attachers
        else {
            "state": "closed",
            "text": "No ParaView node half is open.",
        }
    )
    return ToolResult(content=[_text(status["text"])], structured_content=status)


@mcp.tool(
    app=AppConfig(visibility=["app"]),
    annotations={
        "title": "Stop reconnecting the live view",
        **_hints(read_only=False, destructive=False, idempotent=True),
    },
    tags={"paraview", "view"},
)
def stop_view_relaunch(
    scene: Annotated[
        str | None, Field(description="The scene id the view shows.")
    ] = None,
) -> ToolResult:
    """For the live view: stop starting ParaView's gone node half again. The scenes
    stay saved; open_view(scene=id) starts it and opens one again."""
    key = _scene_keys.get(scene or "", _current)
    found = _attachers.get(key)
    going = found.stop_relaunch() if found is not None else False
    text = (
        "Stopped starting ParaView again. The scene is saved; opening the view "
        "again starts it."
        if going
        else "ParaView was not being started again."
    )
    return ToolResult(
        content=[_text(text)], structured_content={"stopped": going, "text": text}
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


@mcp.prompt(tags={"paraview", "workflow"})
def explore_dataset(path: str) -> list[Message]:
    """Open a dataset in a live ParaView view, measure it, and show what it holds."""
    return [
        Message(
            f"Open {path} in ParaView with open_view (the path is on the machine where "
            "ParaView runs) and call scene_state to see its fields, time steps and "
            "bounds. For the main fields, read exact numbers with measure_field before "
            "describing them, colour the scene by the most informative one with "
            "run_scene_script, and capture the view to check what it shows. Report the "
            "scene id so the scene can be reopened, and keep every claim to what the "
            "measurements and the capture support."
        )
    ]


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
