"""The tools that move through the view: its time steps, watching new ones as they
are written, and the camera.

They register on the server's ``mcp`` where server.py imports this module, so
the tools keep their order.
"""

from __future__ import annotations

import asyncio
import threading
from typing import Annotated, Any, Literal

from fastmcp import Context
from fastmcp.tools import ToolResult
from fastmcp.utilities.tasks import TaskConfig
from pydantic import Field

from . import follow, present
from .server import _hints, _text, command, mcp, run


@mcp.tool(
    annotations={
        "title": "Change the timestep",
        **_hints(read_only=False, destructive=False, idempotent=True),
    },
    tags={"paraview", "time"},
)
async def set_timestep(
    ctx: Context,
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
    result, _ = await run(
        ctx, "set_timestep", {"index": index, "time": time, "latest": latest}
    )
    return present.time_line(result["time"])


@mcp.tool(
    task=TaskConfig(mode="optional"),
    annotations={
        "title": "Watch new time steps",
        **_hints(read_only=True, destructive=False, idempotent=True),
    },
    tags={"paraview", "time"},
)
async def watch_steps(
    ctx: Context,
    idle_seconds: Annotated[
        float,
        Field(
            ge=5,
            le=86400,
            description="The shortest stall that ends the watch: no new step and "
            "nothing changing on disk for this long, or three times the longest gap "
            "between steps so far, whichever is longer.",
        ),
    ] = 300.0,
    max_seconds: Annotated[
        float | None,
        Field(
            ge=1,
            le=604800,
            description="Only when the person asks for a time limit; omit it "
            "otherwise. The watch ends by itself when the run does.",
        ),
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
    until the run ends. Prefer running it as a background task: each step is then
    the task's status. Called directly, each step is a progress notification and the
    call lasts until the watch ends; cancelling it ends it. Progress counts the
    steps written, against the view descriptor's expected_steps when it gives them.
    It ends once those are written ("complete"), when the writer closes its output
    ("finished"), when the source stalls ("stalled": no new step and nothing
    changing on disk), when the source is not growing, or when the person cancels,
    and returns every step it saw. The view follows the newest step unless the
    person parked it."""

    async def fetch(index: int) -> dict[str, Any]:
        stop = threading.Event()
        try:
            result, _ = await asyncio.to_thread(
                command, "steps", {"after": index}, stop=stop
            )
        except asyncio.CancelledError:
            stop.set()
            raise
        return dict(result)

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


@mcp.tool(
    annotations={
        "title": "Move the camera",
        # azimuth, elevation and zoom turn from where the camera is.
        **_hints(read_only=False, destructive=False, idempotent=False),
    },
    tags={"paraview", "camera"},
)
async def set_camera(
    ctx: Context,
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
    """Place the camera, or turn and zoom it from where it is. A placed camera
    stays put while a followed growing source moves; looking along an axis lets it
    follow the data again."""
    result, _ = await run(
        ctx,
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


@mcp.tool(
    annotations={
        "title": "Fit the camera",
        **_hints(read_only=False, destructive=False, idempotent=True),
    },
    tags={"paraview", "camera"},
)
async def fit_camera(
    ctx: Context,
    object: Annotated[
        str | None, Field(description="Fit this object; default: everything.")
    ] = None,
) -> str:
    """Point the camera so everything, or one object, fills the view. Fitting
    everything lets the camera follow a growing source's moving data again."""
    result, _ = await run(ctx, "fit_camera", {"object": object})
    return present.camera_text(result["camera"])
