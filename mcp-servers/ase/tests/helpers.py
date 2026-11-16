"""Call the server's tools in memory (fastmcp Client)."""

from __future__ import annotations

from typing import Any

import pytest


async def call(name: str, args: dict[str, Any]) -> Any:
    from fastmcp import Client

    from ase_mcp.server import mcp

    async with Client(mcp) as client:
        return await client.call_tool(name, args)


async def call_error(name: str, args: dict[str, Any]) -> str:
    from fastmcp import Client
    from fastmcp.exceptions import ToolError

    from ase_mcp.server import mcp

    async with Client(mcp) as client:
        with pytest.raises(ToolError) as info:
            await client.call_tool(name, args)
    return str(info.value)
