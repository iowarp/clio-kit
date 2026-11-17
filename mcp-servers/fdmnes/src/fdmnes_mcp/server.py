"""FDMNES MCP server: XANES spectra of structure files with FDMNES.

Adapted in part from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/xanes_core.py;
see :mod:`.inputs` and :mod:`.spectrum`. The LangChain/Parsl wrappers, the Materials
Project fetcher and the plotting were not ported: structures come as files (from the
ase or pubchem servers, or the person), and plots come from the spectrum CSVs.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Annotated, Any, Literal, TypeVar

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import compute, engine, materials, present
from .errors import FdmnesError
from .inputs import DEFAULT_RANGE, EDGES, XanesSpec

PATHS_META = "ai.iowarp.clio/paths"
READ = {"machine": "local", "access": "read"}
WRITE = {"machine": "local", "access": "write"}
INSTALL = (
    "FDMNES is free from the Institut Néel (CNRS). The fdmnes plugin's install skill "
    "builds it from source with Spack (spack install fdmnes); then set FDMNES_COMMAND to "
    "the executable or put fdmnes on PATH, and restart this server."
)

mcp = FastMCP(
    "fdmnes",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "X-ray absorption near-edge spectra (XANES) with FDMNES. run_xanes computes the "
        "spectrum of one absorbing element (or chosen sites) at an edge (K, L23, …) for a "
        "crystal (CIF) or molecule (XYZ, SDF, PDB) file and writes the spectrum as CSV "
        "(energy from the Fermi level, photon energy, raw and normalized absorption) "
        "with E0, white line and peaks. read_spectrum normalizes an existing spectrum (an "
        "FDMNES *_conv.txt or a measured two-column file); compare_spectra puts several "
        "normalized spectra on one grid and reports their differences. "
        "fetch_materials_project writes crystal structures (CIF) from the Materials Project "
        "by formula or id (needs MP_API_KEY). fdmnes_status says whether FDMNES is "
        "installed here and how to install it."
    ),
)

T = TypeVar("T")


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except FdmnesError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


OutputDir = Annotated[
    str | None,
    Field(
        description="Folder for outputs (a chemistry/fdmnes/... subfolder is made in it); "
        "default the workspace ($CLIO_KIT_ARTIFACTS) or the current folder."
    ),
]


@mcp.tool(
    name="run_xanes",
    title="Run XANES (FDMNES)",
    description=(
        "Compute a XANES spectrum with FDMNES for a structure file: a crystal (CIF, "
        "POSCAR; periodic) or a molecule/cluster (XYZ, SDF, PDB). absorber: an element "
        "('Fe'; every Fe site, FDMNES groups the equivalent ones) or a list of 0-based site "
        "indices (one spectrum each). edge: K, L1, L2, L3, L23, M45 …. method 'green' "
        "(multiple scattering, muffin-tin; fast, the usual first choice) or 'fdm' (finite "
        "differences, full potential; slow). radius: cluster radius in Å (6 is typical; "
        "convergence needs checking). scf: self-consistent potential (default on). "
        "energy_range: Emin, step, [E, step, …] Emax in eV from the Fermi level. Writes "
        "the FDMNES input and logs, spectrum CSVs (energy_eV, photon_energy_eV, mu, "
        "mu_norm) and result.json, and reports E0, white line and peaks. Minutes for a "
        "small crystal with green; much longer for fdm or large radii."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": False,
    },
    tags={"chemistry", "spectroscopy", "simulation"},
    meta={PATHS_META: {"structure": READ, "output_dir": WRITE}},
)
async def run_xanes(
    structure: Annotated[str, Field(description="Structure file (CIF, XYZ, SDF, PDB, POSCAR).")],
    absorber: Annotated[
        str | list[int],
        Field(description="Absorbing element ('Fe') or 0-based site indices ([0, 4])."),
    ],
    edge: Annotated[str, Field(description=f"Edge: {', '.join(EDGES)}.")] = "K",
    method: Annotated[
        Literal["green", "fdm"], Field(description="green (fast) or fdm (full potential).")
    ] = "green",
    radius: Annotated[float, Field(ge=1, le=12, description="Cluster radius, Å.")] = 6.0,
    scf: Annotated[bool, Field(description="Self-consistent potential.")] = True,
    quadrupole: Annotated[bool, Field(description="Add E1E2/E2E2 (pre-edges).")] = False,
    magnetism: Annotated[bool, Field(description="Spin-polarized calculation.")] = False,
    energy_range: Annotated[
        list[float] | None,
        Field(description=f"Emin, step, [E, step, …] Emax in eV; default {list(DEFAULT_RANGE)}."),
    ] = None,
    extra_keywords: Annotated[
        list[str] | None,
        Field(description="Further FDMNES input blocks, each verbatim (e.g. 'Hubbard\\n 5.')."),
    ] = None,
    timeout_s: Annotated[
        float | None, Field(gt=0, description="Stop FDMNES after this many seconds.")
    ] = None,
    run_name: Annotated[str | None, Field(description="Name of the run folder.")] = None,
    output_dir: OutputDir = None,
    ctx: Context | None = None,
) -> ToolResult:
    spec = XanesSpec(
        absorber=absorber,
        edge=edge.strip().upper(),
        method=method,
        radius=radius,
        scf=scf,
        quadrupole=quadrupole,
        magnetism=magnetism,
        energy_range=tuple(energy_range) if energy_range else DEFAULT_RANGE,
        extra_keywords=list(extra_keywords or []),
    )

    async def report(progress: float, total: float | None, message: str) -> None:
        if ctx is not None:
            await ctx.report_progress(progress, total, message)

    try:
        result = await compute.run_xanes(structure, spec, run_name, output_dir, timeout_s, report)
    except FdmnesError as exc:
        raise ToolError(exc.text()) from exc
    return _result(present.run_text(result), result)


@mcp.tool(
    name="read_spectrum",
    title="Read Spectrum",
    description=(
        "Read a XANES spectrum (an FDMNES *_conv.txt, or a measured spectrum as two "
        "columns energy, absorption in text or CSV), shift its energies by energy_shift "
        "eV, normalize it (pre-edge line subtracted, divided by the edge step at E0, the "
        "steepest rise) and write a CSV with energy_eV, mu and mu_norm. Reports E0, white "
        "line and peaks; when the range is too short to normalize it says why."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "spectroscopy"},
    meta={PATHS_META: {"path": READ, "output_dir": WRITE}},
)
async def read_spectrum(
    path: Annotated[str, Field(description="Spectrum file.")],
    energy_shift: Annotated[float, Field(description="Added to every energy, eV.")] = 0.0,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: compute.read(path, energy_shift, output_dir))
    return _result(present.read_text(data), data)


@mcp.tool(
    name="compare_spectra",
    title="Compare Spectra",
    description=(
        "Compare two or more XANES spectra (FDMNES runs, spectrum CSVs from run_xanes, or "
        "measured two-column files): each is normalized; align true shifts each so its E0 "
        "matches the first (needed between a calculation, relative to the Fermi level, and "
        "a measurement, in photon energy); all are put on the first one's energy grid over "
        "the shared range. Writes one CSV (energy_eV and a column per spectrum) and reports "
        "each one's E0, shift, R-factor and correlation against the first."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "spectroscopy"},
    meta={PATHS_META: {"paths": READ, "output_dir": WRITE}},
)
async def compare_spectra(
    paths: Annotated[list[str], Field(description="Spectrum files; the first is the reference.")],
    labels: Annotated[list[str] | None, Field(description="A label per spectrum.")] = None,
    align: Annotated[bool, Field(description="Shift each spectrum's E0 onto the first's.")] = False,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: compute.compare(paths, labels, align, output_dir))
    return _result(present.compare_text(data), data)


@mcp.tool(
    name="fetch_materials_project",
    title="Fetch from Materials Project",
    description=(
        "Crystal structures from the Materials Project by formula ('Fe3O4') or material id "
        "('mp-19306'), written as CIF files (with magnetic moments when the material has "
        "them) under output_dir/chemistry/fdmnes/structures/. A formula returns every "
        "matching material within energy_above_hull_max eV/atom of the convex hull, up to "
        "max_results, ordered by energy above the hull, each with its space group, "
        "magnetic ordering and CIF path; none is chosen for you. Needs a Materials Project "
        "API key (MP_API_KEY; mp_key_missing otherwise)."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    tags={"chemistry", "materials-project", "structure"},
    meta={PATHS_META: {"output_dir": WRITE}},
)
async def fetch_materials_project(
    query: Annotated[str, Field(description="A formula ('Fe3O4') or an id ('mp-19306').")],
    energy_above_hull_max: Annotated[
        float, Field(ge=0, description="Largest energy above the hull, eV/atom.")
    ] = 0.05,
    max_results: Annotated[int, Field(ge=1, le=25, description="Most materials.")] = 5,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: materials.fetch(query, energy_above_hull_max, max_results, output_dir))
    return _result(present.materials_text(data), data)


@mcp.tool(
    name="fdmnes_status",
    title="FDMNES Status",
    description=(
        "Whether the FDMNES executable is available here, where it was found "
        "(FDMNES_COMMAND, FDMNES_EXE or PATH), and how to install it when it is not."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry"},
)
async def fdmnes_status() -> ToolResult:
    data = {"executable": engine.find_executable(), "install": INSTALL}
    return _result(present.status_text(data), data)


@mcp.resource("fdmnes://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the fdmnes MCP server can do."""
    return {
        "tools": [
            "run_xanes",
            "read_spectrum",
            "compare_spectra",
            "fetch_materials_project",
            "fdmnes_status",
        ],
        "edges": list(EDGES),
        "methods": ["green", "fdm"],
        "structures": "files: CIF, POSCAR (crystals); XYZ, SDF/MOL, PDB (molecules)",
        "outputs": "chemistry/fdmnes/<run>/ under output_dir, $CLIO_KIT_ARTIFACTS or cwd",
        "needs": "the FDMNES executable (FDMNES_COMMAND, FDMNES_EXE or fdmnes on PATH); "
        "MP_API_KEY for fetch_materials_project",
    }


@mcp.prompt()
def site_xanes(structure: str, element: str) -> list[Message]:
    """Guided K-edge XANES for each inequivalent site."""
    return [
        Message(
            f"Compute the {element} K-edge XANES of {structure} for each inequivalent {element} site. "
            "Name each site by its coordination, say whether the potential was self-consistent, "
            "and compare the pre-edge and edge positions between sites."
        ),
    ]


def main() -> None:
    """Entry point for the fdmnes MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
