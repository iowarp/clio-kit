"""Every parquet tool declares that it never reaches an external service.

The server only reads local Parquet files, so each tool's annotations must
carry ``openWorldHint: False`` on the wire.
"""

from __future__ import annotations

import pytest
from fastmcp import Client
from parquet_mcp.server import mcp


@pytest.mark.asyncio
async def test_every_tool_declares_closed_world() -> None:
    async with Client(mcp) as client:
        tools = await client.list_tools()
    assert tools, "server exposed no tools"
    for tool in tools:
        assert tool.annotations is not None, f"{tool.name} has no annotations"
        assert tool.annotations.open_world_hint is False, (
            f"{tool.name} openWorldHint is {tool.annotations.open_world_hint!r}"
        )
