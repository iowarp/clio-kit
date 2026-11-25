"""ASE MCP server: molecular calculations with ASE, structures passed as files.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/ase_core.py and
tools/cheminformatics_core.py; the LangChain/MCP wrappers, AtomsData conversions and
runtime context are dropped. Every calculation runs in a worker process
(:mod:`.runner`, :mod:`.worker`) in the environment its calculator needs (:mod:`.envs`).
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Annotated, Any, Literal, TypeVar

from fastmcp import Context, FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import calculators, envs, present, runner, structures
from . import smiles as smiles_core
from .errors import AseError
from .records import existing_file, slug
from .schemas import RunInput, parse_calculator, parse_constraints

PATHS_META = "ai.iowarp.clio/paths"
READ = {"machine": "local", "access": "read"}
WRITE = {"machine": "local", "access": "write"}

mcp = FastMCP(
    "ase",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "Molecular calculations with ASE on structure files. smiles_to_structure builds a 3D "
        "structure (XYZ + SDF) from SMILES; inspect_structure reports formula, bonds and "
        "requested distances/angles/dihedrals; set_geometry sets a distance, angle or "
        "dihedral. run_ase computes an energy, optimization (opt), vibrations (vib), "
        "thermochemistry (thermo) or IR spectrum (ir) with EMT, xTB, MACE, UMA, NWChem or "
        "ORCA, optionally with constraints, and writes a result JSON, trajectory, per-step "
        "CSV, frequencies and normal-mode animations. displace_along_mode and "
        "trajectory_frame make new starting structures from a mode or a trajectory frame. "
        "calculators says which calculators are ready here and what a missing one needs; "
        "xTB, MACE and UMA build their environment on first use (minutes)."
    ),
)

T = TypeVar("T")


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except AseError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


OutputDir = Annotated[
    str | None,
    Field(
        description="Folder for outputs (a chemistry/ase/... subfolder is made in it); "
        "default the workspace ($CLIO_KIT_ARTIFACTS) or the current folder."
    ),
]


@mcp.tool(
    name="smiles_to_structure",
    title="SMILES to 3D Structure",
    description=(
        "Build a 3D structure from a SMILES string with RDKit: ETKDG embedding, hydrogens "
        "added, MMFF94 (else UFF) relaxation. Writes XYZ and SDF and returns the structure "
        "record, the canonical SMILES, formal charge and radical-electron count."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "structure"},
    meta={PATHS_META: {"output_dir": WRITE}},
)
async def smiles_to_structure(
    smiles: Annotated[str, Field(description="SMILES, e.g. 'CCO'.")],
    name: Annotated[str | None, Field(description="File name stem, e.g. 'ethanol'.")] = None,
    seed: Annotated[int, Field(description="Random seed of the embedding.")] = 2025,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: smiles_core.build(smiles, name, seed, output_dir))
    return _result(present.smiles_text(data), data)


@mcp.tool(
    name="inspect_structure",
    title="Inspect Structure",
    description=(
        "Report a structure file's formula, composition, bonds (covalent radii) with "
        "lengths, fragments and size, and measure distances (2 indices), angles (3) or "
        "dihedrals (4) for the index lists in measure. Indices are 0-based in file order. "
        "For a trajectory, frame picks the frame (default last)."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "structure"},
    meta={PATHS_META: {"structure": READ}},
)
async def inspect_structure(
    structure: Annotated[str, Field(description="Structure file (XYZ, SDF, PDB, CIF, traj).")],
    measure: Annotated[
        list[list[int]] | None,
        Field(description="Index lists to measure, e.g. [[0, 1], [2, 0, 1, 5]]."),
    ] = None,
    frame: Annotated[int, Field(description="Frame of a multi-frame file; -1 = last.")] = -1,
) -> ToolResult:
    data = _run(lambda: structures.inspect(structure, measure, frame))
    return _result(present.inspect_text(data), data)


@mcp.tool(
    name="set_geometry",
    title="Set Geometry",
    description=(
        "Write a new structure with a distance (Å), angle (degrees) and/or dihedral "
        "(degrees) set to a value, each as {indices, value}. The smaller of the two "
        "fragments the bond splits the molecule into moves rigidly; bonds in rings are "
        "refused. Reports the values before and after."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "structure"},
    meta={PATHS_META: {"structure": READ, "output_dir": WRITE}},
)
async def set_geometry(
    structure: Annotated[str, Field(description="Structure file to start from.")],
    dihedral: Annotated[
        dict[str, Any] | None,
        Field(description="{indices: [a, b, c, d], value: degrees}; b–c is turned."),
    ] = None,
    distance: Annotated[
        dict[str, Any] | None, Field(description="{indices: [a, b], value: Å}.")
    ] = None,
    angle: Annotated[
        dict[str, Any] | None, Field(description="{indices: [a, b, c], value: degrees}.")
    ] = None,
    name: Annotated[str | None, Field(description="File name stem of the new structure.")] = None,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(
        lambda: structures.set_internal(structure, distance, angle, dihedral, name, output_dir)
    )
    return _result(present.set_text(data), data)


@mcp.tool(
    name="run_ase",
    title="Run ASE Calculation",
    description=(
        "Run a calculation on a structure file. driver: energy (single point), opt "
        "(optimize), vib (optimize, then harmonic frequencies), thermo (vib + ideal-gas H, "
        "S, G at temperature/pressure), ir (vib + IR intensities; needs a dipole: xtb, "
        "nwchem, orca). calculator: {type: emt|xtb|mace|uma|nwchem|orca, model?, method?, "
        "basis?, ...settings}, e.g. {type: 'xtb'} (GFN2-xTB), {type: 'mace'} (MACE-OFF23 "
        "small), {type: 'nwchem', method: 'B3LYP', basis: '6-31G*'}. charge and "
        "multiplicity go to the calculator and the thermochemistry. constraints hold "
        "atoms or internal coordinates during the optimization: {type: 'fix_atoms', "
        "indices}, {type: 'bond', indices: [a, b], value_A?}, {type: 'angle', indices, "
        "value_deg?}, {type: 'dihedral', indices: [a, b, c, d], value_deg?}; frequencies "
        "always use the unconstrained Hessian. optimize false runs vib/thermo/ir on the "
        "structure as given. Writes result.json, final.xyz, the trajectory (extended XYZ) "
        "and a per-step CSV, frequencies.csv, one animation per mode, IR CSVs. The first "
        "xtb/mace/uma calculation builds that calculator's environment (minutes)."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True,
    },
    tags={"chemistry", "simulation"},
    meta={PATHS_META: {"structure": READ, "output_dir": WRITE}},
)
async def run_ase(
    structure: Annotated[str, Field(description="Structure file (XYZ, SDF, PDB, CIF, traj).")],
    calculator: Annotated[
        dict[str, Any] | str,
        Field(description="{type, model?, method?, basis?, ...settings} or just the type."),
    ],
    driver: Annotated[
        Literal["energy", "opt", "vib", "thermo", "ir"], Field(description="What to compute.")
    ] = "energy",
    charge: Annotated[int, Field(description="Total charge.")] = 0,
    multiplicity: Annotated[int, Field(ge=1, description="Spin multiplicity 2S+1.")] = 1,
    optimizer: Annotated[
        Literal["bfgs", "lbfgs", "fire", "gpmin", "mdmin"], Field(description="Optimizer.")
    ] = "bfgs",
    fmax: Annotated[float, Field(gt=0, description="Force convergence, eV/Å.")] = 0.05,
    steps: Annotated[int, Field(ge=0, description="Maximum optimization steps.")] = 200,
    optimize: Annotated[
        bool, Field(description="vib/thermo/ir: optimize first (default) or use as given.")
    ] = True,
    temperature: Annotated[float, Field(gt=0, description="Kelvin, for thermo.")] = 298.15,
    pressure: Annotated[float, Field(gt=0, description="Pa, for thermo.")] = 101325.0,
    constraints: Annotated[
        list[dict[str, Any]] | None, Field(description="Constraints for the optimization.")
    ] = None,
    run_name: Annotated[str | None, Field(description="Name of the run folder.")] = None,
    output_dir: OutputDir = None,
    ctx: Context | None = None,
) -> ToolResult:
    def build_job() -> RunInput:
        spec = parse_calculator(calculator)
        path = existing_file(structure)
        name = run_name or f"{slug(path.stem)}-{driver}-{spec.type}"
        return RunInput(
            structure=str(path),
            driver=driver,
            calculator=spec,
            charge=charge,
            multiplicity=multiplicity,
            optimizer=optimizer,
            fmax=fmax,
            steps=steps,
            optimize=optimize,
            temperature=temperature,
            pressure=pressure,
            constraints=parse_constraints(constraints),
            run_name=name,
        )

    job = _run(build_job)

    async def report(progress: float, total: float | None, message: str) -> None:
        if ctx is not None:  # a no-op when the client asked for no progress
            await ctx.report_progress(progress, total, message)

    try:
        result = await runner.run(job, output_dir, report)
    except AseError as exc:
        raise ToolError(exc.text()) from exc
    return _result(present.run_text(result), result)


@mcp.tool(
    name="displace_along_mode",
    title="Displace Along Mode",
    description=(
        "Write a structure displaced along a normal mode, e.g. to leave a saddle point "
        "along its imaginary mode before re-optimizing. source: a run_ase result JSON (with "
        "vibrations), its normal_modes.npz, or a mode_NNN.xyz animation; mode_index as in "
        "frequencies.csv; amplitude is the largest atomic displacement in Å. both=true "
        "also writes the opposite direction."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "structure"},
    meta={PATHS_META: {"source": READ, "output_dir": WRITE}},
)
async def displace_along_mode(
    source: Annotated[str, Field(description="Result JSON, normal_modes.npz or mode XYZ.")],
    mode_index: Annotated[int | None, Field(description="Mode, 0-based.")] = None,
    amplitude: Annotated[float, Field(description="Largest atomic displacement, Å.")] = 0.3,
    both: Annotated[bool, Field(description="Also write the minus direction.")] = False,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: structures.displace(source, mode_index, amplitude, both, output_dir))
    return _result(present.displace_text(data), data)


@mcp.tool(
    name="trajectory_frame",
    title="Trajectory Frame",
    description=(
        "Write one frame of a trajectory (e.g. opt_trajectory.xyz or an ASE .traj) as a "
        "structure file, to restart from it. frame is 0-based; -1 is the last."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "structure"},
    meta={PATHS_META: {"trajectory": READ, "output_dir": WRITE}},
)
async def trajectory_frame(
    trajectory: Annotated[str, Field(description="Multi-frame XYZ or ASE .traj file.")],
    frame: Annotated[int, Field(description="Frame index; -1 = last.")] = -1,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: structures.frame(trajectory, frame, output_dir))
    return _result(present.frame_text(data), data)


def calculator_status() -> list[dict[str, Any]]:
    """Each calculator: ready, needs setup (and what), or prepared."""
    out: list[dict[str, Any]] = []
    uv = None
    try:
        uv = envs.uv_executable()
    except AseError:
        uv = None
    for kind in ("emt", "xtb", "mace", "uma", "nwchem", "orca"):
        entry: dict[str, Any] = {"type": kind, "about": calculators.DESCRIPTION[kind]}
        extra = calculators.EXTRA[kind]
        needs: list[str] = []
        if extra:
            st = envs.status(extra)
            entry["environment"] = st
            if not st["prepared"]:
                if uv is None:
                    needs.append("uv (to build the environment) [uv_missing]")
                else:
                    needs.append(f"builds its environment on first use ({st['download']}, minutes)")
        if kind == "mace":
            cache = Path.home() / ".cache" / "mace"
            entry["model_cache"] = str(cache)
            entry["models_downloaded"] = (
                sorted(p.name for p in cache.glob("*")) if cache.is_dir() else []
            )
            if not entry["models_downloaded"]:
                needs.append(
                    "downloads the model weights on first use (MACE-OFF23 small: about 7 MB)"
                )
        if kind == "uma":
            tok = calculators.hf_token()
            entry["hf_token"] = tok
            if not tok["found"]:
                needs.append(
                    "HF_TOKEN from a Hugging Face account that accepted the UMA "
                    "licence (facebook/UMA) [hf_token_missing]"
                )
        if kind == "nwchem":
            found = calculators.nwchem_command()
            entry["command"] = found
            if not found["found"]:
                needs.append(f"{calculators.nwchem_missing()} [binary_missing]")
        if kind == "orca":
            found = calculators.orca_command()
            entry["command"] = found
            if not found["found"]:
                needs.append(f"{calculators.orca_missing()} [binary_missing]")
        blocking = [n for n in needs if "[" in n]
        if blocking:
            entry["state"] = "needs setup"
        elif needs:
            entry["state"] = "ready (first use prepares it)"
        else:
            entry["state"] = "ready"
        entry["needs"] = needs
        if needs:
            entry["detail"] = "; ".join(needs)
        elif kind in ("nwchem", "orca"):
            entry["detail"] = f"found {entry['command']['command']} ({entry['command']['source']})"
        elif extra:
            entry["detail"] = f"environment prepared at {entry['environment']['path']}"
        else:
            entry["detail"] = "in the server environment"
        out.append(entry)
    return out


@mcp.tool(
    name="calculators",
    title="Calculators",
    description=(
        "Which calculators are ready here, which need setup and what each missing one "
        "needs (NWChem/ORCA executables, a Hugging Face token for UMA, uv for the xTB/MACE/"
        "UMA environments), and which calculator environments are already prepared. "
        "prepare: calculators whose environment to build now (xtb, mace, uma), so the "
        "first calculation does not wait for it."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    tags={"chemistry"},
)
async def calculators_tool(
    prepare: Annotated[
        list[Literal["xtb", "mace", "uma"]] | None,
        Field(description="Build these calculator environments now (minutes each)."),
    ] = None,
    ctx: Context | None = None,
) -> ToolResult:
    async def report(progress: float, total: float | None, message: str) -> None:
        if ctx is not None:
            await ctx.report_progress(progress, total, message)

    prepared = []
    for extra in prepare or []:
        try:
            info = await runner.prepare_env(extra, report)
        except AseError as exc:
            raise ToolError(exc.text()) from exc
        if info and info.get("built_now"):
            prepared.append(f"{extra} (built in {info.get('build_seconds')} s)")
        elif info:
            prepared.append(f"{extra} (already prepared)")
    data = {
        "calculators": calculator_status(),
        "env_cache": str(envs.cache_root()),
        "prepared_now": prepared,
    }
    return _result(present.calculators_text(data), data)


@mcp.resource("ase://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the ase MCP server can do."""
    return {
        "tools": [
            "smiles_to_structure",
            "inspect_structure",
            "set_geometry",
            "run_ase",
            "displace_along_mode",
            "trajectory_frame",
            "calculators",
        ],
        "drivers": ["energy", "opt", "vib", "thermo", "ir"],
        "calculators": list(calculators.DESCRIPTION),
        "structures": "files: XYZ, extended XYZ, SDF/MOL, PDB, CIF, ASE .traj, POSCAR",
        "outputs": "chemistry/ase/<run>/ under output_dir, $CLIO_KIT_ARTIFACTS or cwd",
    }


@mcp.prompt()
def optimize_and_check(structure: str) -> list[Message]:
    """Guided optimization with a frequency check."""
    return [
        Message(
            f"Optimize {structure} with a calculator suited to it, then compute its frequencies. "
            "Report the method, the final energy and largest force, and any imaginary modes; "
            "say what an imaginary mode means for this structure."
        ),
    ]


def main() -> None:
    """Entry point for the ase MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
