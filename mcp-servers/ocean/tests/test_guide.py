"""The ocean server's resources and prompts."""

from __future__ import annotations

import json

from fastmcp import Client

from ocean_mcp.guide import TOOL_GROUPS
from ocean_mcp.implementation import hydrography
from ocean_mcp.server import mcp


async def test_capabilities_group_every_registered_tool_once() -> None:
    async with Client(mcp) as client:
        tools = {tool.name for tool in await client.list_tools()}
        content = await client.read_resource("ocean://capabilities")
    grouped = [name for names in TOOL_GROUPS.values() for name in names]
    assert sorted(grouped) == sorted(tools)
    capabilities = json.loads(content[0].text)
    assert capabilities["tool_groups"] == TOOL_GROUPS
    assert capabilities["water_mass_definitions"] == ["atlantic_emery_meincke_1986"]
    assert "provenance.json" in capabilities["outputs"]


async def test_water_mass_resource_is_the_built_in_library() -> None:
    async with Client(mcp) as client:
        listed = {str(r.uri) for r in await client.list_resources()}
        content = await client.read_resource("ocean://water-masses")
    assert listed == {"ocean://capabilities", "ocean://water-masses"}
    library = json.loads(content[0].text)
    assert library == hydrography.builtin_definitions()
    for definition in library.values():
        assert definition["citation"] and definition["masses"]


async def test_water_mass_census_prompt_chains_the_profile_tools() -> None:
    async with Client(mcp) as client:
        prompts = {p.name for p in await client.list_prompts()}
        result = await client.get_prompt(
            "water_mass_census",
            {
                "box": "[-40, -20, 30, 50, 0, 2000]",
                "start": "2020-01-01",
                "end": "2020-12-31",
            },
        )
    assert prompts == {"water_mass_census"}
    text = result.messages[0].content.text
    assert "[-40, -20, 30, 50, 0, 2000]" in text and "2020-12-31" in text
    assert "atlantic_emery_meincke_1986" in text
    order = ["argo_fetch", "teos10", "water_masses", "profile_metrics", "section"]
    positions = [text.index(name) for name in order]
    assert positions == sorted(positions)
