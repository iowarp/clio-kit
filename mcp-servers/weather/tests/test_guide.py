"""The weather server's resources and prompts."""

from __future__ import annotations

import json

from fastmcp import Client

from weather_mcp.guide import SOURCES, TOOL_GROUPS
from weather_mcp.server import mcp


async def test_capabilities_group_every_registered_tool_once() -> None:
    async with Client(mcp) as client:
        tools = {tool.name for tool in await client.list_tools()}
        listed = {str(r.uri) for r in await client.list_resources()}
        content = await client.read_resource("weather://capabilities")
    grouped = [name for names in TOOL_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(tools)
    assert set(SOURCES) <= tools
    assert listed == {"weather://capabilities"}
    capabilities = json.loads(content[0].text)
    assert capabilities["tool_groups"] == TOOL_GROUPS
    assert capabilities["sources"] == SOURCES
    assert "provenance.json" in capabilities["outputs"]


async def test_ensemble_risk_prompt_chains_fetch_and_exceedance() -> None:
    async with Client(mcp) as client:
        prompts = {p.name for p in await client.list_prompts()}
        result = await client.get_prompt(
            "ensemble_risk", {"param": "tp", "threshold": "50 mm", "place": "Lisbon"}
        )
    assert prompts == {"ensemble_risk"}
    text = result.messages[0].content.text
    assert (
        "tp" in text and "50 mm" in text and "Lisbon" in text and "24, 48, 72" in text
    )
    assert text.index("ensemble_fetch") < text.index("exceedance")
    assert "station_fetch" in text
