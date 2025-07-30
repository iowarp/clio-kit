# server.py

#  Created on: 2nd June, 2025
#      Author: Soham Sonar ssonar2@hawk.illinoistech.edu



import os
import sys
import json
from fastmcp import FastMCP
from dotenv import load_dotenv

# Ensure parent directory is on PYTHONPATH so "capabilities" can be found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Load .env variables, if any
load_dotenv()

# Import our ADIOS BP5 capability and handler
import mcp_handlers

# Initialize FastMCP server
mcp = FastMCP("ADIOSMCP")

# ─── ADIOS BP5 TOOLS ─────────────────────────────────────────────────────────

# List BP5 Files Tool
@mcp.tool(
    name="list_bp5",
    description="Lists all BP5 files in a given directory, the bp5 files are actually directories so both file and directory words are correct. The 'directory' parameter must be an absolute path."
)
async def list_bp5_tool(directory: str = "data/") -> dict:
    """List all bp5 files in the specified directory."""
    try:
        return await mcp_handlers.list_bp5_files(directory)
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "list_bp5", "error": type(e).__name__},
            "isError": True
        }

# ─── INSPECT VARIABLES ─────────────────────────────────────────────────────── 
@mcp.tool(
    name="inspect_variables",
    description="Inspects variables in a BP5 file. If variable_name is provided, returns data for that specific variable. Otherwise, shows type, shape, and steps for all variables. The 'filename' parameter must be an absolute path to the BP5 file."
)
async def inspect_variables_tool(filename: str, variable_name: str = None) -> dict:
    try:
        return await mcp_handlers.inspect_variables_handler(filename, variable_name)
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "inspect_variables", "error": type(e).__name__},
            "isError": True
        }

# ─── INSPECT VARIABLES AT STEP ─────────────────────────────────────────────────
@mcp.tool(
    name="inspect_variables_at_step",
    description="Inspects a specific variable at a given step in a BP5 file. Shows variable type, shape, min, max. All parameters are required. The 'filename' must be an absolute path."
)
async def inspect_variables_at_step_tool(
    filename: str, variable_name: str, step: int
) -> dict:
    try:
        return await mcp_handlers.inspect_variables_at_step_handler(filename, variable_name, step)
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "inspect_variables_at_step", "error": type(e).__name__},
            "isError": True
        }

# ─── INSPECT ATTRIBUTES ──────────────────────────────────────────────────────
@mcp.tool(
    name="inspect_attributes",
    description="Reads global or variable-specific attributes from a BP5 file. The 'filename' parameter must be an absolute path. The 'variable_name' is optional."
)
async def inspect_attributes_tool(
    filename: str,
    variable_name: str = None
) -> dict:
    try:
        return await mcp_handlers.inspect_attributes_handler(filename, variable_name)
    except Exception as e:
        return {
            "content": [{"text": json.dumps({"error": str(e)})}],
            "_meta": {"tool": "inspect_attributes", "error": type(e).__name__},
            "isError": True
        }

import mcp_handlers

# ─── READ VARIABLE AT STEP ────────────────────────────────────────────────────
@mcp.tool(
    name="read_variable_at_step",
    description="Reads a named variable at a specific step from a BP5 file. " \
    "WARNING: This tool reads the entire variable into memory. For large variables (>1000 elements), " \
    "first use 'inspect_variables' to check the size, then use 'read_variable_chunk' instead to avoid memory/token issues. The 'filename' must be an absolute path."
)
async def read_variable_at_step_tool(
    filename: str, variable_name: str, target_step: int
) -> dict:
    return await mcp_handlers.read_variable_at_step_handler(
        filename, variable_name, target_step
    )

# ─── READ VARIABLE CHUNK ──────────────────────────────────────────────────────
@mcp.tool(
    name="read_variable_chunk",
    description="Reads a chunk of a large variable at a specific step from a BP5 file. This tool is designed for large variables (>1000 elements) to avoid memory/token issues. " \
    "First use 'inspect_variables' to check if a variable is large enough to need chunking. The chunk size is automatically calculated as roughly 1/5 of the total data. " \
    "Returns data chunk plus chunk_info with 'next_start_index' for subsequent calls. Continue calling with next_start_index until 'is_complete' is True. The 'filename' must be an absolute path."
)
async def read_variable_chunk_tool(
    filename: str, variable_name: str, target_step: int, start_index: int = 0
) -> dict:
    return await mcp_handlers.read_variable_chunk_handler(
        filename, variable_name, target_step, start_index
    )


def main():
    """
    Main entry point to run the ADIOS MCP server.
    Chooses between stdio or SSE transport based on MCP_TRANSPORT.
    """
    try:
        transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
        if transport == "sse":
            host = os.getenv("MCP_SSE_HOST", "0.0.0.0")
            port = int(os.getenv("MCP_SSE_PORT", "8000"))
            print(json.dumps({"message": f"Starting SSE on {host}:{port}"}), file=sys.stderr)
            mcp.run(transport="sse", host=host, port=port)
        else:
            #print(json.dumps({"message": "Starting stdio transport"}), file=sys.stderr)
            mcp.run(transport="stdio")
    except Exception as e:
        print(json.dumps({"error": str(e)}), file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
