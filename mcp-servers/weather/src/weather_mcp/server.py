"""Weather MCP server: fields, ensembles, soundings, verification, objects and climate.

Tools live in modules that register on the shared app; importing them here
puts them on the server.
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from . import (  # noqa: F401 - importing registers the tools
    tools_ai,
    tools_climate,
    tools_forecast,
    tools_verify,
)
from .app import mcp

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Entry point for the weather MCP server."""
    parser = argparse.ArgumentParser(description="Weather MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "http":
        mcp.run(transport="http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
