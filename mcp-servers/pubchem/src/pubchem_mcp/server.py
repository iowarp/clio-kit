"""PubChem MCP server: compound names to PubChem records and 3D structures.

Talks to PubChem's PUG REST API over HTTPS (no key), at most 5 requests per second.
Name lookups return candidates when a name is ambiguous, never a silent first match.
"""

from __future__ import annotations

from collections.abc import Awaitable
from typing import Annotated, Any, TypeVar

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from fastmcp.tools import ToolResult
from mcp.types import TextContent
from pydantic import Field

from . import compounds, present
from .client import PubChemClient
from .errors import PubChemError

PATHS_META = "ai.iowarp.clio/paths"

mcp = FastMCP(
    "pubchem",
    version="1.0.0",  # keep equal to mcp-server-versions.toml
    instructions=(
        "Look compounds up in PubChem by name and fetch their structures. resolve_name "
        "turns a name into a PubChem compound (CID, formula, SMILES, InChIKey, stereo "
        "centres); when the name is ambiguous ('xylene') or unknown it returns candidates "
        "and you must ask the person which one they mean. get_compound gives the record "
        "and synonyms of a CID. fetch_structure writes the compound's 3D conformer as SDF "
        "for a simulation; when PubChem has no 3D conformer it says so and writes the 2D "
        "record. Needs network access to pubchem.ncbi.nlm.nih.gov."
    ),
)

CLIENT = PubChemClient()
T = TypeVar("T")


async def _run(work: Awaitable[T]) -> T:
    try:
        return await work
    except PubChemError as exc:
        raise ToolError(exc.text()) from exc


def _result(text: str, data: dict[str, Any]) -> ToolResult:
    return ToolResult(content=[TextContent(type="text", text=text)], structured_content=data)


READ_ONLY = {
    "readOnlyHint": True,
    "destructiveHint": False,
    "idempotentHint": True,
    "openWorldHint": True,
}


@mcp.tool(
    name="resolve_name",
    title="Resolve Compound Name",
    description=(
        "Find the PubChem compound a name means. A name PubChem maps to exactly one "
        "compound returns status 'resolved' with its CID, title, IUPAC name, formula, "
        "molecular weight (g/mol), isomeric and canonical SMILES, InChIKey and stereo "
        "centre counts. A name that maps to several compounds or to none ('xylene') "
        "returns status 'ambiguous' with up to max_candidates candidates (found by "
        "PubChem's name dictionary and word search) for the person to choose from."
    ),
    annotations=READ_ONLY,
    tags={"chemistry", "pubchem"},
)
async def resolve_name(
    name: Annotated[str, Field(description="A compound name, e.g. 'p-xylene' or 'aspirin'.")],
    max_candidates: Annotated[
        int, Field(description="Most candidates to return when ambiguous.", ge=1, le=25)
    ] = 8,
) -> ToolResult:
    data = await _run(compounds.resolve_name(CLIENT, name, max_candidates))
    return _result(present.resolve_text(data), data)


@mcp.tool(
    name="get_compound",
    title="Get Compound",
    description=(
        "The PubChem record of one compound by CID: title, IUPAC name, formula, molecular "
        "weight (g/mol), isomeric and canonical SMILES, InChIKey, charge, stereo centre "
        "counts, components, and its first 20 synonyms (common names, CAS numbers)."
    ),
    annotations=READ_ONLY,
    tags={"chemistry", "pubchem"},
)
async def get_compound(
    cid: Annotated[int, Field(description="PubChem compound id (CID).", gt=0)],
) -> ToolResult:
    data = await _run(compounds.get_compound(CLIENT, cid))
    return _result(present.compound_text(data), data)


@mcp.tool(
    name="fetch_structure",
    title="Fetch Structure",
    description=(
        "Download a compound's 3D conformer (with hydrogens) from PubChem as an SDF file "
        "under output_dir/chemistry/pubchem/cid<CID>-<name>/, plus a JSON compound record. "
        "Returns the structure record {path, format, formula, n_atoms, sha256}. When "
        "PubChem has no 3D conformer the result says so (has_3d false, fallback "
        "'2d_record') and writes the flat 2D record; embed 3D coordinates from the SMILES "
        "with ase.smiles_to_structure instead."
    ),
    annotations={
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": True,
    },
    tags={"chemistry", "pubchem"},
    meta={PATHS_META: {"output_dir": {"machine": "local", "access": "write"}}},
)
async def fetch_structure(
    cid: Annotated[int, Field(description="PubChem compound id (CID).", gt=0)],
    output_dir: Annotated[
        str | None,
        Field(description="Folder for the files; default $CLIO_KIT_ARTIFACTS or the cwd."),
    ] = None,
) -> ToolResult:
    data = await _run(compounds.fetch_structure(CLIENT, cid, output_dir))
    return _result(present.structure_text(data), data)


@mcp.resource("pubchem://capabilities")
def capabilities() -> dict[str, Any]:
    """Describe what the pubchem MCP server can do."""
    return {
        "tools": ["resolve_name", "get_compound", "fetch_structure"],
        "source": "PubChem PUG REST and autocomplete (pubchem.ncbi.nlm.nih.gov), no key",
        "rate_limit": "at most 5 requests per second, retries with backoff on busy",
        "ambiguous_names": "returned as candidates for the person to choose; never a first match",
        "outputs": ["3D (or 2D) SDF", "compound record JSON"],
    }


@mcp.prompt()
def compound_lookup(name: str) -> list[Message]:
    """Guided lookup of a compound in PubChem by name."""
    return [
        Message(
            f"Find {name} in PubChem: resolve the name to its compound ID, report the formula, "
            "molecular weight and canonical SMILES, and fetch its 3D structure as a file."
        ),
    ]


def main() -> None:
    """Entry point for the pubchem MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
