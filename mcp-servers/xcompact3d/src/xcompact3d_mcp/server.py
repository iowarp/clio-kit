"""Xcompact3d MCP server: flow past a CAD body with Xcompact3d, set up and read back.

The server prepares a run directory from an STL body (``prepare_case``), reports
how far a run is (``check_case``) and reads the forces on the body
(``read_forces``). It does not start the solver: the run directory carries a
``run.sh`` and a job spec any batch launcher can submit (for example clio-relay's
``run``), and a ``clio.view-descriptor.v1`` a viewer can follow while the run
writes its snapshots (for example the ParaView MCP's ``open_view``).
"""

from __future__ import annotations

from collections.abc import Callable
from importlib import resources
from typing import Annotated, Any, Literal, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import case, results
from .errors import CaseError

PATHS_META = "ai.iowarp.clio/paths"
VERSION = "1.0.0"  # keep equal to mcp-server-versions.toml

mcp = FastMCP(
    "xcompact3d",
    version=VERSION,
    instructions=(
        "Set up and read back Xcompact3d flow simulations around a CAD body. "
        "prepare_case turns an STL into a run directory (grid, time step, immersed body, "
        "ADIOS2 snapshot output) and returns a job spec to submit and a view descriptor "
        "to follow the snapshots live; check_case says how far a run is; read_forces gives "
        "drag and lift coefficients and the shedding frequency. build_instructions says "
        "how to build the patched solver. Lengths are in body lengths L, speeds in the "
        "inflow speed U, so Re = U L / nu."
    ),
)

T = TypeVar("T")


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except CaseError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


def _size(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024 or unit == "TB":
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} TB"


def _plan_text(record: dict[str, Any]) -> str:
    p = record["plan"]
    g, d, r = p["grid"], p["domain"], p["resolution"]
    lines = [
        f"Prepared {p['name']} in {p['run_dir']}.",
        (
            f"Flow along +x at Re {p['reynolds']:g}, angle {p['angle_deg']:g} degrees, "
            f"span {p['span']}."
        ),
        (
            f"Grid {g['nx']} x {g['ny']} x {g['nz']} ({g['points']:,} points) over "
            f"{d['xlx']:.2f} x {d['yly']:.2f} x {d['zlz']:.2f} L, "
            f"{r['points_per_length']:g} points per L, {r['points_across_body']:.1f} across the body."
        ),
        (
            f"dt {p['dt']:g} (CFL {r['cfl']:.2f}), {p['ilast']} steps to t = {p['end_time']:g} L/U, "
            f"a snapshot every {p['ioutput']} steps ({p['snapshots']} in all)."
        ),
        f"Each snapshot is {_size(p['snapshot_bytes'])}, {_size(p['total_bytes'])} for the run.",
        (
            f"Body: {p['body']['solid_voxels']:,} solid mask voxels, frontal height "
            f"{p['body']['frontal_height']:.3f} L."
        ),
        (
            f"Run it with run.sh ({p['decomposition']['nprocs']} MPI ranks); relay_job is the "
            "same as a job spec, and view follows data.bp5 while it grows."
        ),
    ]
    lines += [f"Warning: {w}" for w in p["warnings"]]
    return "\n".join(lines)


@mcp.tool(
    name="prepare_case",
    title="Prepare Flow Case",
    description=(
        "Write an Xcompact3d run directory for the flow past a body given as an STL. "
        "The body is scaled so reference_length (default its length along x) is 1, turned "
        "nose-up by angle_deg about +z, and placed upstream in a box with a uniform inflow "
        "along +x, a convective outflow and periodic sides. span 'periodic' treats an "
        "extruded section (a wing section, a cylinder) as spanning the periodic z "
        "direction; 'free' puts a 3D body (a car, a finite wing) in the middle; 'auto' "
        "decides from the shape. Returns the grid, time step, snapshot size, the "
        "run.sh to start it, relay_job (a job spec for a batch launcher such as "
        "clio-relay run) and view (a clio.view-descriptor.v1 for ParaView's open_view)."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"cfd", "xcompact3d"},
    meta={
        PATHS_META: {
            "stl": {"machine": "local", "access": "read"},
            "run_dir": {"machine": "local", "access": "write"},
        }
    },
)
async def prepare_case_tool(
    stl: Annotated[str, Field(description="Path of the body's STL file (closed surface).")],
    run_dir: Annotated[
        str, Field(description="Folder to write the case to, on the filesystem the solver sees.")
    ],
    reynolds: Annotated[float, Field(description="Re = U L / nu on the reference length.")] = 500.0,
    angle_deg: Annotated[
        float, Field(description="Angle of attack: degrees nose-up about +z.")
    ] = 0.0,
    reference_length: Annotated[
        float | None,
        Field(description="The body length L in STL units; default its length along x."),
    ] = None,
    span: Annotated[
        Literal["auto", "periodic", "free"],
        Field(description="'periodic' for an extruded section, 'free' for a 3D body."),
    ] = "auto",
    points_per_length: Annotated[
        int, Field(description="Grid points per body length (cost grows with its cube).")
    ] = 24,
    end_time: Annotated[
        float, Field(description="Simulated time in L/U (a body passes 1 L per unit).")
    ] = 40.0,
    snapshots: Annotated[int, Field(description="Snapshots to write over the run.")] = 80,
    upstream: Annotated[float, Field(description="Box length ahead of the body, in L.")] = 3.0,
    downstream: Annotated[float, Field(description="Box length behind the body, in L.")] = 10.0,
    lateral: Annotated[float, Field(description="Clearance on each side of the body, in L.")] = 3.0,
    nprocs: Annotated[int, Field(description="MPI ranks the run will use.")] = 32,
    iibm: Annotated[
        int,
        Field(description="Immersed boundary: 1 direct forcing (robust), 2 reconstruction."),
    ] = 1,
    init_noise: Annotated[
        float, Field(description="Initial noise level that lets the wake break symmetry.")
    ] = 0.05,
    cfl: Annotated[
        float,
        Field(
            description=(
                "CFL number on the inflow speed. The wake runs up to about 2 U, and AB3 with "
                "the compact scheme holds to about 0.35 locally, so keep it near 0.1."
            )
        ),
    ] = 0.1,
    executable: Annotated[
        str,
        Field(description="The xcompact3d program; default XCOMPACT3D_BIN or the build path."),
    ] = "",
    prepare: Annotated[
        str, Field(description="A shell line run.sh runs first, e.g. 'module load openmpi'.")
    ] = "",
    name: Annotated[str, Field(description="A short case name; default the STL name.")] = "",
    overwrite: Annotated[
        bool, Field(description="Replace a run directory that already holds results.")
    ] = False,
) -> ToolResult:
    if iibm not in (1, 2):
        raise ToolError("iibm is 1 or 2. [bad_parameters]")
    request = case.CaseRequest(
        stl=stl,
        run_dir=run_dir,
        reynolds=reynolds,
        angle_deg=angle_deg,
        reference_length=reference_length,
        span=span,
        points_per_length=points_per_length,
        upstream=upstream,
        downstream=downstream,
        lateral=lateral,
        end_time=end_time,
        snapshots=snapshots,
        cfl=cfl,
        iibm=iibm,
        init_noise=init_noise,
        nprocs=nprocs,
        executable=executable,
        prepare=prepare,
        name=name,
    )
    record = _run(lambda: case.prepare_case(request, overwrite=overwrite))
    return _result(_plan_text(record), record)


@mcp.tool(
    name="check_case",
    title="Check Flow Run",
    description=(
        "How far a run is: the time step and simulated time from its log, the snapshots "
        "written, the seconds per step and left, whether the body mask was loaded and "
        "whether the run failed or diverged, with the log's last lines and the view "
        "descriptor."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cfd", "xcompact3d"},
    meta={PATHS_META: {"run_dir": {"machine": "local", "access": "read"}}},
)
async def check_case_tool(
    run_dir: Annotated[str, Field(description="The run directory prepare_case wrote.")],
) -> ToolResult:
    data = _run(lambda: results.case_status(run_dir))
    if not data["started"]:
        text = f"{data['run_dir']} has not started: there is no xcompact3d.log yet."
    else:
        state = "failed" if data["failed"] else "finished" if data["finished"] else "is running"
        text = (
            f"The run {state}: step {data['step']} of {data['last_step']} "
            f"(t = {data['time']:.3f} of {data['end_time']:g} L/U), "
            f"{data['snapshots_written']} of {data['snapshots']} snapshots, "
            f"{_size(data['data_bytes'])} written."
        )
        if data["seconds_per_step"]:
            text += f" {data['seconds_per_step']:.3f} s per step"
            if data["seconds_left"] and not data["finished"]:
                text += f", about {data['seconds_left'] / 60:.1f} min left"
            text += "."
        if not data["geometry_loaded"]:
            text += " The log does not show the body mask loading."
        if data["failed"]:
            text += "\n" + "\n".join(data["log_tail"])
    return _result(text, data)


@mcp.tool(
    name="read_forces",
    title="Read Forces",
    description=(
        "Drag and lift coefficients on the body over the run: mean, rms, extremes, lift to "
        "drag, the dominant lift frequency and its Strouhal number on the body length and "
        "on the frontal height, plus the time series (at most 400 points) for a chart. The "
        "first discard_fraction of the run (the start-up) is left out of the statistics."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"cfd", "xcompact3d"},
    meta={PATHS_META: {"run_dir": {"machine": "local", "access": "read"}}},
)
async def read_forces_tool(
    run_dir: Annotated[str, Field(description="The run directory prepare_case wrote.")],
    discard_fraction: Annotated[
        float, Field(description="Share of the run, from the start, left out of the statistics.")
    ] = 0.4,
) -> ToolResult:
    data = _run(lambda: results.read_forces(run_dir, discard_fraction))
    text = (
        f"From t = {data['averaged_from']:.2f} to {data['time_range'][1]:.2f} L/U "
        f"({data['samples']} samples): Cd {data['cd_mean']:.4f} (rms {data['cd_rms']:.4f}), "
        f"Cl {data['cl_mean']:.4f} (rms {data['cl_rms']:.4f}, {data['cl_min']:.3f} to "
        f"{data['cl_max']:.3f})"
    )
    if data["lift_to_drag"] is not None:
        text += f", L/D {data['lift_to_drag']:.2f}"
    text += f". Coefficients are {data['basis']}."
    if data["shedding_frequency"]:
        text += (
            f" The lift oscillates at f = {data['shedding_frequency']:.4f} U/L: St "
            f"{data['strouhal_length']:.4f} on L, {data['strouhal_frontal']:.4f} on the frontal "
            "height."
        )
    else:
        text += " No periodic shedding shows yet (fewer than two periods in the window)."
    return _result(text, data)


def _data(name: str) -> str:
    return resources.files("xcompact3d_mcp").joinpath(f"data/{name}").read_text("utf-8")


@mcp.tool(
    name="build_instructions",
    title="Build Instructions",
    description=(
        "How to build the patched Xcompact3d this server prepares cases for: the needs, "
        "the two patch scripts (written to patch_dir when given) and the build commands."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"cfd", "xcompact3d"},
    meta={PATHS_META: {"patch_dir": {"machine": "local", "access": "write"}}},
)
async def build_instructions_tool(
    patch_dir: Annotated[
        str, Field(description="Folder to write the two patch scripts to; empty writes none.")
    ] = "",
) -> ToolResult:
    from pathlib import Path

    written = []
    if patch_dir:
        folder = Path(patch_dir).expanduser()
        folder.mkdir(parents=True, exist_ok=True)
        for script in ("clio_vtx_patch.py", "clio_geometry_patch.py"):
            (folder / script).write_text(_data(script), encoding="utf-8")
            written.append(str(folder / script))
    guide = _data("BUILD.md")
    data = {
        "guide": guide,
        "patches": written,
        "default_executable": case.default_executable(),
    }
    return _result(guide + ("\n\nPatch scripts: " + ", ".join(written) if written else ""), data)


@mcp.resource("xcompact3d://build", mime_type="text/markdown")
def build_guide() -> str:
    """How to build the patched Xcompact3d."""
    return _data("BUILD.md")


@mcp.resource("xcompact3d://patches/{name}", mime_type="text/x-python")
def patch_script(name: str) -> str:
    """One of the two patch scripts: clio_vtx_patch.py or clio_geometry_patch.py."""
    if name not in ("clio_vtx_patch.py", "clio_geometry_patch.py"):
        raise ValueError("The patches are clio_vtx_patch.py and clio_geometry_patch.py.")
    return _data(name)


@mcp.resource("xcompact3d://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the xcompact3d MCP server can do."""
    return {
        "tools": ["prepare_case", "check_case", "read_forces", "build_instructions"],
        "solver": "Xcompact3d (Incompact3d) v5.0 with the clio VTX and geometry patches",
        "case": "flow past an immersed STL body (itype 5), periodic span or free 3D body",
        "outputs": [
            "run directory with input.i3d, body mask, ADIOS2 config, run.sh",
            "job spec for a batch launcher (relay_job)",
            "clio.view-descriptor.v1 following data.bp5 (ParaView ADIOS2VTXReader)",
            "drag, lift and Strouhal number from forces.dat",
        ],
        "runs_solver": "no: submit run.sh or relay_job with a launcher",
    }


def main() -> None:
    """Entry point for the xcompact3d MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
