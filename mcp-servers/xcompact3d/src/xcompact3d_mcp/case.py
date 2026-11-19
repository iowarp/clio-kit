"""Turn a CAD body into an Xcompact3d run directory: the flow past (or through) it.

The case is Xcompact3d's cylinder wake case (``itype = 5``) with the clio
geometry patch: a uniform inflow along +x, a convective outflow, periodic sides,
and the body immersed through the immersed boundary method from a voxel mask.

Everything is in units of the body's reference length L and the inflow speed U,
so the Reynolds number is ``U L / nu`` and one time unit is L / U. The body is
scaled so that L is 1.
"""

from __future__ import annotations

import json
import math
import os
import shlex
from dataclasses import asdict, dataclass, field
from importlib import resources
from pathlib import Path
from typing import Any, Literal

import numpy as np

from .errors import CaseError
from .geometry import Mask, Mesh, mask_for, read_stl, write_stl

SpanMode = Literal["auto", "periodic", "free"]

#: Point fields each snapshot holds (the cylinder case's, plus ``ep`` from the patch).
FIELDS = ("ux", "uy", "uz", "pp", "vort", "critq", "ep")
#: Where ``build_guide`` installs the solver; the default executable.
DEFAULT_PREFIX = Path("~/.clio/xcompact3d/current").expanduser()


def default_executable() -> str:
    env = os.environ.get("XCOMPACT3D_BIN", "").strip()
    if env:
        return env
    return str(DEFAULT_PREFIX / "bin" / "xcompact3d")


def nice(n: int, periodic: bool) -> int:
    """Smallest count >= n whose FFT length (n, or n - 1 when not periodic) is 2^a 3^b 5^c."""
    n = max(n, 8)
    while True:
        m = n if periodic else n - 1
        for p in (2, 3, 5):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def decomposition(nprocs: int, nx: int, ny: int, nz: int) -> tuple[int, int]:
    """The 2D pencil grid (p_row, p_col) closest to square that the grid allows."""
    best: tuple[int, int] | None = None
    for p_col in range(1, nprocs + 1):
        if nprocs % p_col:
            continue
        p_row = nprocs // p_col
        if p_row > min(nx, ny) or p_col > min(ny, nz):
            continue
        if best is None or max(p_row, p_col) < max(best):
            best = (p_row, p_col)
    if best is None:
        raise CaseError(
            "too_many_ranks", f"{nprocs} MPI ranks cannot split a {nx} x {ny} x {nz} grid."
        )
    return best


DIFFUSION_LIMIT = 0.0075


def nice_dt(limit: float) -> float:
    """The largest of 1, 1.5, 2, 2.5, 3, 4 or 5 times a power of ten at most ``limit``."""
    power = 10.0 ** math.floor(math.log10(limit))
    for factor in (5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0):
        if factor * power <= limit:
            return factor * power
    return power


@dataclass
class CaseRequest:
    stl: str
    run_dir: str
    reynolds: float = 500.0
    angle_deg: float = 0.0
    reference_length: float | None = None
    span: SpanMode = "auto"
    points_per_length: int = 24
    upstream: float = 3.0
    downstream: float = 10.0
    lateral: float = 3.0
    end_time: float = 40.0
    snapshots: int = 80
    cfl: float = 0.1
    iibm: int = 1
    init_noise: float = 0.05
    nprocs: int = 32
    executable: str = ""
    prepare: str = ""
    mpirun: str = "mpirun"
    name: str = ""


@dataclass
class CasePlan:
    name: str
    run_dir: str
    span: str
    reynolds: float
    angle_deg: float
    reference_length_cad: float
    scale: float
    grid: dict[str, int]
    domain: dict[str, float]
    spacing: float
    dt: float
    ilast: int
    ioutput: int
    snapshots: int
    end_time: float
    body: dict[str, Any]
    resolution: dict[str, float]
    force_volume: dict[str, float]
    decomposition: dict[str, int]
    snapshot_bytes: int
    total_bytes: int
    warnings: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)


def _span_mode(mesh: Mesh, requested: SpanMode, spacing: float) -> str:
    if requested != "auto":
        return requested
    # An extruded section has volume = (area of its middle section) x span.
    size = mesh.size
    if size[2] <= 0.0:
        return "free"
    step = min(size[0], size[1]) / 200.0
    plane = mask_for(mesh, (step, step, step), span=(0.0, 0.0))
    area = float(plane.solid.sum()) * step * step
    if area <= 0.0:
        return "free"
    prism = area * size[2]
    return "periodic" if abs(mesh.volume() - prism) <= 0.05 * prism else "free"


def plan_case(request: CaseRequest) -> tuple[CasePlan, Mesh, Mask]:
    """Size the domain, grid and time step for a body; nothing is written."""
    raw = read_stl(request.stl)
    if request.reynolds <= 0 or request.points_per_length < 4 or request.snapshots < 1:
        raise CaseError(
            "bad_parameters", "reynolds > 0, points_per_length >= 4 and snapshots >= 1."
        )
    reference = request.reference_length or float(raw.size[0])
    if reference <= 0:
        raise CaseError("bad_reference", "The body has no length along x; give reference_length.")
    h = 1.0 / request.points_per_length
    # Angle of attack: nose up for flow along +x is a turn by -angle about +z.
    body = raw.transformed(1.0 / reference, -request.angle_deg, np.zeros(3))
    span = _span_mode(body, request.span, h)
    size = body.size
    warnings: list[str] = []

    nx = nice(math.ceil((request.upstream + size[0] + request.downstream) / h) + 1, False)
    ny = nice(math.ceil((size[1] + 2 * request.lateral) / h), True)
    if span == "periodic":
        # An extruded section is close to 2D: twice the in-plane spacing across the span
        # costs half as much and eases the diffusion limit on the time step.
        nz = nice(max(8, round(size[2] / (2 * h))), True)
        zlz = float(size[2])
        dz = zlz / nz
    else:
        nz = nice(math.ceil((size[2] + 2 * request.lateral) / h), True)
        dz = h
        zlz = nz * h
    xlx, yly = (nx - 1) * h, ny * h

    shift = np.array(
        [
            request.upstream - body.lower[0],
            0.5 * yly - 0.5 * (body.lower[1] + body.upper[1]),
            -body.lower[2]
            if span == "periodic"
            else 0.5 * zlz - 0.5 * (body.lower[2] + body.upper[2]),
        ]
    )
    body = body.transformed(1.0, 0.0, shift)
    mask = mask_for(
        body,
        (h, h, dz),
        refine=1 if request.iibm == 1 else 2,
        span=(0.0, zlz - dz) if span == "periodic" else None,
    )
    solid = int(mask.solid.sum())
    if solid == 0:
        raise CaseError(
            "body_unresolved",
            f"No grid point falls inside the body at {request.points_per_length} points per "
            "length; raise points_per_length or check that the STL is closed.",
        )

    dt_limit = request.cfl * h  # U = 1
    dt = nice_dt(dt_limit)
    # Diffusion is explicit, and the hyperviscous second derivative (isecondder = 5,
    # nu0nu = 4) damps the grid cutoff about pi^2 (1 + nu0nu) ~ 49 times harder than
    # dt nu / dx^2 suggests. AB3 holds while dt nu sum(1/dx^2) stays under ~0.011
    # (0.0077 runs, 0.013 diverges), so keep it at 0.0075.
    inverse_squares = 2.0 / (h * h) + 1.0 / (dz * dz)
    viscous_limit = DIFFUSION_LIMIT * request.reynolds / inverse_squares
    if dt > viscous_limit:
        dt = nice_dt(viscous_limit)
        warnings.append(
            "The time step is limited by viscous stability, not by the CFL number; fewer "
            "points per length or a higher Reynolds number make each time unit cheaper."
        )
    ilast = math.ceil(request.end_time / dt)
    ioutput = max(1, ilast // request.snapshots)
    ilast = ioutput * request.snapshots

    lo, hi = body.lower, body.upper
    margin = max(0.25, 4 * h)
    force = {
        "xld": float(max(lo[0] - margin, h)),
        "xrd": float(min(hi[0] + margin, xlx - h)),
        "yld": float(max(lo[1] - margin, h)),
        "yud": float(min(hi[1] + margin, yly - h)),
    }
    if span == "free":
        force["zfr"] = float(max(lo[2] - margin, dz))
        force["zbk"] = float(min(hi[2] + margin, zlz - dz))

    thickness = float(size[1])
    across = thickness / h
    if across < 6:
        warnings.append(
            f"The body is only {across:.1f} grid points thick across the flow; forces will be "
            "rough. Raise points_per_length for numbers you want to quote."
        )
    p_row, p_col = decomposition(request.nprocs, nx, ny, nz)
    points = nx * ny * nz
    snapshot_bytes = points * len(FIELDS) * 8
    frontal = float(size[1] * (size[2] if span == "free" else 1.0))
    name = request.name or Path(request.stl).stem
    plan = CasePlan(
        name=name,
        run_dir=str(Path(request.run_dir).expanduser().resolve()),
        span=span,
        reynolds=request.reynolds,
        angle_deg=request.angle_deg,
        reference_length_cad=reference,
        scale=1.0 / reference,
        grid={"nx": nx, "ny": ny, "nz": nz, "points": points},
        domain={"xlx": xlx, "yly": yly, "zlz": zlz},
        spacing=h,
        dt=dt,
        ilast=ilast,
        ioutput=ioutput,
        snapshots=request.snapshots,
        end_time=ilast * dt,
        body={
            "lower": [float(v) for v in lo],
            "upper": [float(v) for v in hi],
            "size": [float(v) for v in size],
            "volume": body.volume(),
            "frontal_height": thickness,
            "frontal_area": frontal,
            "triangles": len(body.triangles),
            "solid_voxels": solid,
            "mask_shape": list(mask.shape_xyz),
        },
        resolution={
            "points_per_length": request.points_per_length,
            "points_across_body": across,
            "cfl": dt / h,
            "diffusion_number": dt / request.reynolds * inverse_squares,
        },
        force_volume=force,
        decomposition={"p_row": p_row, "p_col": p_col, "nprocs": request.nprocs},
        snapshot_bytes=snapshot_bytes,
        total_bytes=snapshot_bytes * request.snapshots,
        warnings=warnings,
    )
    return plan, body, mask


def input_i3d(plan: CasePlan, request: CaseRequest) -> str:
    g, d, f = plan.grid, plan.domain, plan.force_volume
    nobjmax = 1 if request.iibm == 1 else 4
    span3d = plan.span == "free"
    volume = [
        f"xld(1) = {f['xld']:.6f}",
        f"xrd(1) = {f['xrd']:.6f}",
        f"yld(1) = {f['yld']:.6f}",
        f"yud(1) = {f['yud']:.6f}",
    ]
    if span3d:
        volume += [f"zfr(1) = {f['zfr']:.6f}", f"zbk(1) = {f['zbk']:.6f}", "i2dsim = 0"]
    else:
        volume.append("i2dsim = 1")
    ilist = max(1, plan.ioutput // 5)
    return f"""! -*- mode: f90 -*-
! Written by xcompact3d-mcp prepare_case for {plan.name}.
! Flow past a CAD body (clio-geometry.txt and clio-geometry.bin) along +x,
! lengths in body lengths L, velocities in the inflow speed U, Re = U L / nu.

&BasicParam
itype = 5
p_row = {plan.decomposition["p_row"]}
p_col = {plan.decomposition["p_col"]}
nx = {g["nx"]}
ny = {g["ny"]}
nz = {g["nz"]}
istret = 0
beta = 0.259065151
xlx = {d["xlx"]:.10g}
yly = {d["yly"]:.10g}
zlz = {d["zlz"]:.10g}
nclx1 = 2
nclxn = 2
ncly1 = 0
nclyn = 0
nclz1 = 0
nclzn = 0
iin = 1
re = {plan.reynolds:.10g}
u1 = 1.
u2 = 1.
init_noise = {request.init_noise:.6g}
inflow_noise = 0.0
dt = {plan.dt:.10g}
ifirst = 1
ilast = {plan.ilast}
ilesmod = 0
iscalar = 0
iibm = {request.iibm}
/End

&NumOptions
ifirstder = 4
isecondder = 5
ipinter = 3
itimescheme = 3
/End

&InOutParam
irestart = 0
icheckpoint = 100000000
ioutput = {plan.ioutput}
ilist = {ilist}
nvisu = 1
/End

&Statistics
nstat = 1
initstat = 100000000
/End

&ScalarParam
numscalar = 0
Sc = 0.2
/End

&LESModel
jles = 0
smagcst = 0.14
SmagWallDamp = 1
walecst = 0.5
iconserv = 0
/End

&ibmstuff
cex = 0.
cey = 0.
ra = 0.
nraf = 10
nobjmax = {nobjmax}
iforces = 1
nvol = 1
/End

&ForceCVs
{chr(10).join(volume)}
/End

&CASE
/End
"""


def vtk_xml(plan: CasePlan) -> str:
    """The ``vtk.xml`` attribute ParaView's ADIOS2VTXReader needs: an image grid, x fastest.

    The reader takes WholeExtent in ADIOS2's row-major order (slowest axis first, so z y x
    for Fortran's x-fastest arrays) and reverses it into VTK's, while Spacing and Origin
    stay in VTK's x y z order.
    """
    g, d = plan.grid, plan.domain
    spacing = (d["xlx"] / (g["nx"] - 1), d["yly"] / g["ny"], d["zlz"] / g["nz"])
    extent = f"0 {g['nz'] - 1} 0 {g['ny'] - 1} 0 {g['nx'] - 1}"
    arrays = "\n".join(f'        <DataArray Name="{name}" />' for name in FIELDS)
    return (
        '<?xml version="1.0"?>\n'
        '<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">\n'
        f'  <ImageData WholeExtent="{extent}" Origin="0 0 0" '
        f'Spacing="{spacing[0]!r} {spacing[1]!r} {spacing[2]!r}">\n'
        f'    <Piece Extent="{extent}">\n'
        '      <PointData Scalars="vort">\n'
        f"{arrays}\n"
        '        <DataArray Name="TIME">time</DataArray>\n'
        "      </PointData>\n"
        "    </Piece>\n"
        "  </ImageData>\n"
        "</VTKFile>\n"
    )


def run_script(plan: CasePlan, request: CaseRequest) -> str:
    executable = request.executable or default_executable()
    prepare = request.prepare or os.environ.get("XCOMPACT3D_PREPARE", "")
    env_file = Path(executable).expanduser().parent / "xcompact3d-env.sh"
    lines = [
        "#!/usr/bin/env bash",
        f"# Written by xcompact3d-mcp prepare_case: runs {plan.name} in this directory.",
        "set -eo pipefail",
        'cd "$(dirname "$0")"',
    ]
    if prepare:
        lines.append(prepare)
    lines.append(f"if [ -f {shlex.quote(str(env_file))} ]; then . {shlex.quote(str(env_file))}; fi")
    lines += [
        "rm -rf data.bp5 data forces.dat xcompact3d.log",
        (
            f"{request.mpirun} -np {request.nprocs} {shlex.quote(executable)} input.i3d "
            "> xcompact3d.log 2>&1"
        ),
        "",
    ]
    return "\n".join(lines)


def view_descriptor(plan: CasePlan) -> dict[str, Any]:
    return {
        "schema": "clio.view-descriptor.v1",
        "name": "fields",
        "path": str(Path(plan.run_dir) / "data.bp5"),
        "path_kind": "directory",
        "format": "adios2-bp5",
        "reader": "ADIOS2VTXReader",
        "reader_properties": {},
        "growing": True,
        "follow": {
            "signal": "reader-time-steps",
            "poll_seconds": 2,
            "expected_steps": plan.snapshots,
        },
        "time": {"unit": "L/U", "step_interval": plan.ioutput * plan.dt},
        "display": {
            "array": "vort",
            "association": "POINTS",
            "representation": "Slice",
            "slice_normal": [0, 0, 1],
            "color_map": "Inferno",
            # Wake vorticity is O(1) in U/L; the body's own shear layers saturate.
            "range": [0.0, 3.0],
        },
        "producer": {"package": "xcompact3d-mcp", "version": _version()},
    }


def relay_job(plan: CasePlan) -> dict[str, Any]:
    """A relay ``run`` job spec: one batch job on one node with nprocs task slots."""
    return {
        "name": f"x3d-{plan.name}"[:40],
        "executable": "/bin/bash",
        "arguments": [str(Path(plan.run_dir) / "run.sh")],
        "directory": plan.run_dir,
        "resources": {"node_count": 1, "process_count": plan.decomposition["nprocs"]},
        # run.sh starts mpirun itself; a per-rank launcher would start nprocs copies.
        "launcher": "single",
    }


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("xcompact3d-mcp")
    except Exception:  # noqa: BLE001 - running from a source tree
        return "0"


def prepare_case(request: CaseRequest, overwrite: bool = False) -> dict[str, Any]:
    plan, body, mask = plan_case(request)
    run_dir = Path(plan.run_dir)
    if run_dir.exists() and any(run_dir.iterdir()) and not overwrite:
        if not (run_dir / "case.json").is_file():
            raise CaseError(
                "run_dir_in_use",
                f"{run_dir} has files that are not an xcompact3d-mcp case; pick another run_dir.",
            )
        if (run_dir / "data.bp5").exists():
            raise CaseError(
                "run_dir_has_results",
                f"{run_dir} already holds results (data.bp5). Pass overwrite to replace them.",
            )
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "input.i3d").write_text(input_i3d(plan, request), encoding="utf-8")
    config = resources.files("xcompact3d_mcp").joinpath("data/adios2_config.xml")
    (run_dir / "adios2_config.xml").write_text(config.read_text("utf-8"), encoding="utf-8")
    (run_dir / "clio-vtk.xml").write_text(vtk_xml(plan), encoding="utf-8")
    mask.write(run_dir)
    write_stl(body, run_dir / "body.stl")
    script = run_dir / "run.sh"
    script.write_text(run_script(plan, request), encoding="utf-8")
    script.chmod(0o755)
    descriptor = view_descriptor(plan)
    (run_dir / "view.json").write_text(json.dumps(descriptor, indent=2), encoding="utf-8")
    plan.files = sorted({p.name for p in run_dir.iterdir()} | {"case.json"})
    record = {
        "plan": asdict(plan),
        "request": asdict(request),
        "view": descriptor,
        "relay_job": relay_job(plan),
    }
    (run_dir / "case.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    return record


def load_case(run_dir: str) -> dict[str, Any]:
    path = Path(run_dir).expanduser() / "case.json"
    if not path.is_file():
        raise CaseError("not_a_case", f"{path.parent} has no case.json; run prepare_case first.")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data
