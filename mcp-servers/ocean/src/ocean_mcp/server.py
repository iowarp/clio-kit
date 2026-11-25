"""Ocean MCP server: observations, sections, circulation, events and model checks.

The tools live in five modules, and the resources and prompts in ``guide``; each
registers on the shared app, so importing them here is what puts them on the server.
"""

from __future__ import annotations

import argparse
import logging

from dotenv import load_dotenv

from . import (  # noqa: F401 - importing registers the tools, resources and prompts
    guide,
    tools_altimetry,
    tools_circulation,
    tools_events,
    tools_model,
    tools_profiles,
)
from .app import mcp

load_dotenv()
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def main() -> None:
    """Entry point for the ocean MCP server."""
    parser = argparse.ArgumentParser(description="Ocean MCP server")
    parser.add_argument("--transport", choices=["stdio", "http"], default="stdio")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    if args.transport == "http":
        mcp.run(transport="http", port=args.port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
