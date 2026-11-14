# server.py

#  Created on: 2nd June, 2025
#      Contact: grc@illinoistech.edu


import os
from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from fastmcp.prompts import Message
from typing import Optional, Union
from . import mcp_handlers

# Initialize FastMCP server
mcp: FastMCP = FastMCP(
    "adios",
    version="2.3.2",
    instructions=(
        "Reads and inspects ADIOS2 BP files for scientific I/O. "
        "Use list_bp5 to find files, inspect_variables for every variable's type, "
        "shape, step count and range, and inspect_attributes for metadata. "
        "For questions over time (when does a value change, where is the peak) use "
        "variable_stats: one call gives per-step min/max/mean/std and the index of "
        "each extreme. read_variable_at_step returns raw values; read large arrays "
        "as start/count slices."
    ),
)

# ─── ADIOS BP5 TOOLS ─────────────────────────────────────────────────────────


# List BP5 Files Tool
@mcp.tool(
    name="list_bp5",
    title="List BP5",
    description="Lists the BP4/BP5 files (.bp, .bp5) in a directory, by default the server's working directory.",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"adios", "scientific-io"},
)
async def list_bp5_tool(directory: str = ".") -> dict:
    """List all BP5 files in a specified directory with file metadata."""
    return await mcp_handlers.list_bp5_files(directory)


# ─── INSPECT VARIABLES ───────────────────────────────────────────────────────
@mcp.tool(
    name="inspect_variables",
    title="Inspect Variables",
    description="Inspects variables in a BP5 file, returning type, shape, and steps. Optionally filters by variable name.",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"adios", "variables"},
)
async def inspect_variables_tool(
    filename: str, variable_name: Optional[str] = None
) -> dict:
    """Inspect all variables in a BP5 file or a specific variable by name."""
    return await mcp_handlers.inspect_variables_handler(filename, variable_name)


# ─── INSPECT VARIABLES AT STEP ─────────────────────────────────────────────────
@mcp.tool(
    name="inspect_variables_at_step",
    title="Inspect Step Variables",
    description="Inspects a specific variable at a given step in a BP5 file. All parameters are required.",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"adios", "variables"},
)
async def inspect_variables_at_step_tool(
    filename: str, variable_name: str, step: int
) -> dict:
    """Inspect a specific variable at a given step in a BP5 file."""
    return await mcp_handlers.inspect_variables_at_step_handler(
        filename, variable_name, step
    )


# ─── INSPECT ATTRIBUTES ──────────────────────────────────────────────────────
@mcp.tool(
    name="inspect_attributes",
    title="Inspect Attributes",
    description="Reads global or variable-specific attributes from a BP5 file. The 'variable_name' is optional.",
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"adios", "attributes"},
)
async def inspect_attributes_tool(
    filename: str, variable_name: Optional[str] = None
) -> dict:
    """Read global or variable-specific attributes from a BP5 file."""
    return await mcp_handlers.inspect_attributes_handler(filename, variable_name)


# ─── READ VARIABLE AT STEP ────────────────────────────────────────────────────
@mcp.tool(
    name="read_variable_at_step",
    title="Read Step Variable",
    description=(
        "Reads a variable at one step as a flattened row-major list with its shape. "
        "Pass start/count (one value per dimension) to read a slice; reads above "
        "max_elements (default 10000) are refused, so use variable_stats for summaries."
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"adios", "variables"},
)
async def read_variable_at_step_tool(
    filename: str,
    variable_name: str,
    target_step: int,
    start: Optional[Union[list[int], str]] = None,
    count: Optional[Union[list[int], str]] = None,
    max_elements: Optional[int] = None,
) -> dict:
    """Read a variable, or a start/count slice of it, at one step of a BP5 file.

    Args:
        filename: Path to the .bp file
        variable_name: Variable to read
        target_step: 0-based step
        start: Optional slice offset per dimension, e.g. [30, 10, 5]
        count: Optional slice extent per dimension, e.g. [4, 4, 3]
        max_elements: Largest read allowed (default 10000)
    """
    selection = _selection(start, count)
    if max_elements is not None:
        selection["max_elements"] = max_elements
    return await mcp_handlers.read_variable_at_step_handler(
        filename, variable_name, target_step, **selection
    )


def _selection(start: object, count: object) -> dict:
    from .implementation.selection import parse_indices

    try:
        parsed = {
            "start": parse_indices(start, "start"),
            "count": parse_indices(count, "count"),
        }
    except ValueError as exc:
        raise ToolError(str(exc)) from exc
    return {key: value for key, value in parsed.items() if value is not None}


# ─── VARIABLE STATISTICS OVER STEPS ──────────────────────────────────────────
@mcp.tool(
    name="variable_stats",
    title="Variable Statistics",
    description=(
        "Per-step min, max, mean and std of a variable over a step range, with the "
        "(i, j, k, ...) index of each step's min and max and the overall extremes. "
        "Optionally restricted to a start/count slice."
    ),
    annotations={
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
    },
    tags={"adios", "variables", "statistics"},
)
async def variable_stats_tool(
    filename: str,
    variable_name: str,
    step_start: int = 0,
    step_count: Optional[int] = None,
    start: Optional[Union[list[int], str]] = None,
    count: Optional[Union[list[int], str]] = None,
) -> dict:
    """Statistics of one variable for every step in a range (at most 1000 steps).

    Args:
        filename: Path to the .bp file
        variable_name: Variable to summarize
        step_start: First step (0-based, default 0)
        step_count: Number of steps (default all remaining)
        start: Optional slice offset per dimension
        count: Optional slice extent per dimension
    """
    return await mcp_handlers.variable_stats_handler(
        filename,
        variable_name,
        step_start=step_start,
        step_count=step_count,
        **_selection(start, count),
    )


# ─── RESOURCES ────────────────────────────────────────────────────────────────


@mcp.resource("adios://capabilities")
def adios_capabilities() -> dict:
    """ADIOS2 file format capabilities and supported engines."""
    return {
        "supported_formats": ["BP4", "BP5"],
        "operations": [
            "list files",
            "inspect variables",
            "read attributes",
            "read a variable or slice at a step",
            "per-step statistics with extreme locations",
        ],
    }


# ─── PROMPTS ──────────────────────────────────────────────────────────────────


@mcp.prompt()
def explore_bp_file(file_path: str) -> list[Message]:
    """Guided workflow for exploring an ADIOS2 BP file."""
    return [
        Message(
            f"I need to explore the ADIOS2 BP file at {file_path}. "
            "First run inspect_variables and inspect_attributes, then run "
            "variable_stats on the most important time-varying variables, and read "
            "small start/count slices with read_variable_at_step where detail matters."
        ),
    ]


def main() -> None:
    """Main entry point for the ADIOS MCP server."""
    import argparse

    parser = argparse.ArgumentParser(description="ADIOS MCP Server")
    parser.add_argument("--transport", choices=["stdio", "http"], default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    transport = args.transport or os.getenv("MCP_TRANSPORT", "stdio")
    if transport == "http":
        mcp.run(transport="http", host=args.host, port=args.port)
    else:
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
