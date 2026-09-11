"""Discover real stdio servers across both MCP protocol eras."""

import asyncio
from functools import partial
import sys

import mcp
import pytest

from clio_kit.protocol_probe import inspect_stdio


@pytest.mark.parametrize("mode", ["auto", "legacy"])
def test_probe_collects_paginated_tools_resources_and_prompts(tmp_path, monkeypatch, mode):
    server = tmp_path / "server.py"
    server.write_text('''
import asyncio
from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

async def tools(ctx, params):
    second = params is not None and params.cursor == "second"
    return types.ListToolsResult(
        tools=[types.Tool(name="second" if second else "first", input_schema={"type": "object"})],
        next_cursor=None if second else "second",
    )

async def resources(ctx, params):
    return types.ListResourcesResult(resources=[types.Resource(name="data", uri="test://data")])

async def prompts(ctx, params):
    return types.ListPromptsResult(prompts=[types.Prompt(name="explain")])

server = Server("probe-test", on_list_tools=tools, on_list_resources=resources, on_list_prompts=prompts)
async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())
asyncio.run(main())
''')
    monkeypatch.setattr(mcp, "Client", partial(mcp.Client, mode=mode))
    result = asyncio.run(inspect_stdio(sys.executable, [str(server)], timeout=20))
    assert result["protocol_version"] == ("2025-11-25" if mode == "legacy" else "2026-07-28")
    assert [tool["name"] for tool in result["tools"]] == ["first", "second"]
    assert result["tools"][0]["inputSchema"] == {"type": "object"}
    assert result["resources"][0]["uri"] == "test://data"
    assert result["prompts"][0]["name"] == "explain"
