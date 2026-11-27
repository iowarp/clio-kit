"""gRASPA MCP server: gas adsorption in porous crystals (GCMC) with gRASPA.

Adapted in part from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0),
tools/graspa_core.py and tools/graspa_analysis.py; see :mod:`.inputs`, :mod:`.parse`
and :mod:`.compute`. The LangChain/Parsl/Globus wrappers and the mock runner were not
ported; a run folder can be prepared here and run on a GPU node elsewhere.
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

from . import compute, engine, present
from .errors import GraspaError
from .inputs import ADSORBATES, GcmcSpec

PATHS_META = "ai.iowarp.clio/paths"
READ = {"machine": "local", "access": "read"}
WRITE = {"machine": "local", "access": "write"}
INSTALL = (
    "gRASPA needs a GPU (NVIDIA with the NVHPC compilers, or AMD with HIP). The graspa "
    "plugin's install skill builds it with Spack on a GPU node (spack install graspa); set "
    "GRASPA_COMMAND to the executable there. Without one here, prepare_graspa writes run "
    "folders to submit to a GPU node and read_graspa reads them back."
)

mcp = FastMCP(
    "graspa",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "Gas adsorption in porous crystals (MOFs, zeolites) by grand-canonical Monte Carlo "
        "with gRASPA. run_graspa computes the uptake (mol/kg, with its error bar) of one "
        "adsorbate (CO2, H2O or N2) at a temperature and pressure for a CIF; it needs gRASPA "
        "and a GPU here. prepare_graspa writes the same run folder without running it, to "
        "submit to a GPU node; read_graspa reads a finished folder. rank_frameworks ranks "
        "frameworks by uptake at one condition or by working capacity between an adsorption "
        "and a desorption condition. graspa_status says whether gRASPA is available here."
    ),
)

T = TypeVar("T")


def _run(fn: Callable[[], T]) -> T:
    try:
        return fn()
    except GraspaError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


OutputDir = Annotated[
    str | None,
    Field(
        description="Folder for outputs (a chemistry/graspa/... subfolder is made in it); "
        "default the workspace ($CLIO_KIT_ARTIFACTS) or the current folder."
    ),
]
Structure = Annotated[str, Field(description="Framework CIF file.")]
Adsorbate = Annotated[
    Literal["CO2", "H2O", "N2"], Field(description="The one adsorbate (no mixtures).")
]
Temperature = Annotated[float, Field(gt=0, description="Temperature, K.")]
Pressure = Annotated[float, Field(ge=0, description="Pressure, Pa.")]
InitCycles = Annotated[int, Field(ge=0, description="Initialization cycles.")]
ProdCycles = Annotated[int, Field(gt=0, description="Production cycles (averaged).")]
Charges = Annotated[
    Literal["none", "cif"],
    Field(description="'none': uncharged framework; 'cif': the CIF's _atom_site_charge."),
]
Cutoff = Annotated[float, Field(ge=6, le=20, description="VdW and Coulomb cutoff, Å.")]
RunName = Annotated[str | None, Field(description="Name of the run folder.")]


def _spec(adsorbate, temperature, pressure, init_cycles, prod_cycles, charges, cutoff, seed):
    return GcmcSpec(
        adsorbate=adsorbate,
        temperature_K=temperature,
        pressure_Pa=pressure,
        init_cycles=init_cycles,
        prod_cycles=prod_cycles,
        framework_charges=charges,
        cutoff_A=cutoff,
        seed=seed,
    )


RUN_ANNOTATIONS = {
    "readOnlyHint": False,
    "destructiveHint": False,
    "idempotentHint": False,
    "openWorldHint": False,
}


@mcp.tool(
    name="run_graspa",
    title="Run gRASPA (GCMC)",
    description=(
        "Compute the adsorption uptake of one gas (CO2, H2O or N2) in a framework CIF at a "
        "temperature (K) and pressure (Pa) by grand-canonical Monte Carlo with gRASPA, here. "
        "The cell is replicated so each width is at least twice the cutoff; the force field "
        "is UFF-like Lennard-Jones with TraPPE-style rigid adsorbates; framework charges "
        "'none' (default) or 'cif'. Writes the run folder (inputs, gRASPA output, run.json, "
        "result.json) and reports the uptake in mol/kg with its block-average error bar and "
        "molecules per unit cell. Needs gRASPA and a GPU here (binary_missing otherwise; "
        "use prepare_graspa)."
    ),
    annotations=RUN_ANNOTATIONS,
    tags={"chemistry", "adsorption", "simulation"},
    meta={PATHS_META: {"structure": READ, "output_dir": WRITE}},
)
async def run_graspa(
    structure: Structure,
    adsorbate: Adsorbate,
    temperature: Temperature = 298.15,
    pressure: Pressure = 101325.0,
    init_cycles: InitCycles = 10000,
    prod_cycles: ProdCycles = 10000,
    framework_charges: Charges = "none",
    cutoff: Cutoff = 12.8,
    seed: Annotated[int, Field(ge=0, description="Random seed.")] = 0,
    timeout_s: Annotated[float | None, Field(gt=0, description="Stop after, seconds.")] = None,
    run_name: RunName = None,
    output_dir: OutputDir = None,
    ctx: Context | None = None,
) -> ToolResult:
    spec = _spec(
        adsorbate, temperature, pressure, init_cycles, prod_cycles, framework_charges, cutoff, seed
    )

    async def report(progress: float, total: float | None, message: str) -> None:
        if ctx is not None:
            await ctx.report_progress(progress, total, message)

    try:
        result = await compute.run(structure, spec, run_name, output_dir, timeout_s, report)
    except GraspaError as exc:
        raise ToolError(exc.text()) from exc
    return _result(present.result_text(result), result)


@mcp.tool(
    name="prepare_graspa",
    title="Prepare gRASPA Run",
    description=(
        "Write a gRASPA run folder (framework CIF, force field, adsorbate, simulation.input "
        "with the unit cells, run.json) for one adsorbate at a temperature and pressure, "
        "without running it: for a GPU node elsewhere (a Slurm job that runs gRASPA in the "
        "folder with stdout saved to graspa_stdout.txt). Same settings as run_graspa."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": False},
    tags={"chemistry", "adsorption"},
    meta={PATHS_META: {"structure": READ, "output_dir": WRITE}},
)
async def prepare_graspa(
    structure: Structure,
    adsorbate: Adsorbate,
    temperature: Temperature = 298.15,
    pressure: Pressure = 101325.0,
    init_cycles: InitCycles = 10000,
    prod_cycles: ProdCycles = 10000,
    framework_charges: Charges = "none",
    cutoff: Cutoff = 12.8,
    seed: Annotated[int, Field(ge=0, description="Random seed.")] = 0,
    run_name: RunName = None,
    output_dir: OutputDir = None,
) -> ToolResult:
    spec = _spec(
        adsorbate, temperature, pressure, init_cycles, prod_cycles, framework_charges, cutoff, seed
    )
    data = _run(lambda: compute.prepare(structure, spec, run_name, output_dir))
    return _result(present.prepare_text(data), data)


@mcp.tool(
    name="read_graspa",
    title="Read gRASPA Run",
    description=(
        "Read a finished gRASPA run folder (made by prepare_graspa or run_graspa, run here "
        "or on another machine): the uptake in mol/kg with its error bar, molecules per "
        "unit cell, mg/g, g/L and the Henry coefficient when gRASPA printed them. Writes "
        "result.json in the folder."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "adsorption"},
    meta={PATHS_META: {"run_dir": READ}},
)
async def read_graspa(
    run_dir: Annotated[str, Field(description="The run folder (or its run.json).")],
) -> ToolResult:
    data = _run(lambda: compute.read(run_dir))
    return _result(present.result_text(data), data)


@mcp.tool(
    name="rank_frameworks",
    title="Rank Frameworks",
    description=(
        "Rank frameworks from gRASPA results (result.json files or folders holding them; one "
        "adsorbate): by uptake at the adsorption condition, or, with a desorption condition, "
        "by working capacity (uptake at adsorption minus at desorption). Repeats are "
        "averaged; a framework without a completed run at every requested condition is "
        "excluded with the reason. Writes ranking.csv; top_fraction selects the best share."
    ),
    annotations={"readOnlyHint": False, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry", "adsorption"},
    meta={PATHS_META: {"results": READ, "output_dir": WRITE}},
)
async def rank_frameworks(
    results: Annotated[list[str], Field(description="result.json files or folders.")],
    adsorption: Annotated[
        dict[str, float], Field(description="{temperature_K, pressure_Pa} of adsorption.")
    ],
    desorption: Annotated[
        dict[str, float] | None,
        Field(description="{temperature_K, pressure_Pa} of desorption, for working capacity."),
    ] = None,
    top_fraction: Annotated[float, Field(gt=0, le=1, description="Share to select.")] = 1.0,
    output_dir: OutputDir = None,
) -> ToolResult:
    data = _run(lambda: compute.rank(results, adsorption, desorption, top_fraction, output_dir))
    return _result(present.rank_text(data), data)


@mcp.tool(
    name="graspa_status",
    title="gRASPA Status",
    description=(
        "Whether the gRASPA executable is available here, where it was found "
        "(GRASPA_COMMAND, CHEMGRAPH_GRASPA_EXECUTABLE or PATH), and how to get it."
    ),
    annotations={"readOnlyHint": True, "destructiveHint": False, "idempotentHint": True},
    tags={"chemistry"},
)
async def graspa_status() -> ToolResult:
    data = {"executable": engine.find_executable(), "install": INSTALL}
    return _result(present.status_text(data), data)


@mcp.resource("graspa://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the graspa MCP server can do."""
    return {
        "tools": [
            "run_graspa",
            "prepare_graspa",
            "read_graspa",
            "rank_frameworks",
            "graspa_status",
        ],
        "adsorbates": list(ADSORBATES),
        "structures": "framework CIF files",
        "outputs": "chemistry/graspa/<run>/ under output_dir, $CLIO_KIT_ARTIFACTS or cwd",
        "needs": "gRASPA on a GPU node for run_graspa; nothing for prepare/read/rank",
    }


@mcp.prompt()
def adsorption_uptake(framework: str, adsorbate: str) -> list[Message]:
    """Guided adsorption uptake of a gas in a framework."""
    return [
        Message(
            f"Compute the {adsorbate} uptake in {framework} with grand canonical Monte Carlo at the "
            "conditions I give. Report the uptake with units and its error, the force field, and "
            "whether the framework carries charges."
        ),
    ]


def main() -> None:
    """Entry point for the graspa MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
