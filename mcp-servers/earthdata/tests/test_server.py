"""The wrapper runs the pinned earthdata-mcp-server and changes nothing it says."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import tomllib
from stdio_session import session

from earthdata_mcp import server

ROOT = Path(__file__).resolve().parents[1]
MODULE = "earthdata_mcp.server"
PINNED = "0.6.0"
ENV: dict[str, str] = {}


def test_it_pins_one_exact_upstream_release() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["project"]["dependencies"] == [f"{server.UPSTREAM}=={PINNED}"]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}
    assert locked[server.UPSTREAM] == PINNED


def test_the_command_is_the_upstream_script_with_arguments_passed_through() -> None:
    argv = server.command(["--extra", "value"])
    assert Path(argv[0]).is_file()
    assert Path(argv[0]).stem == server.SCRIPT
    assert argv[1:] == [*server.ARGUMENTS, "--extra", "value"]


def test_the_upstream_server_answers_with_its_own_name_and_tools() -> None:
    replies = session(MODULE, [{"method": "tools/list"}], env=ENV)
    assert replies[0]["result"]["serverInfo"]["name"] == "earthdata"
    names = {tool["name"] for tool in replies[1]["result"]["tools"]}
    assert {
        "search_earth_datasets",
        "search_earth_datagranules",
        "download_earth_data_granules",
    } <= names


def test_the_transport_is_always_stdio() -> None:
    # TRANSPORT in the environment would otherwise switch the upstream to HTTP.
    assert server.ARGUMENTS == ("start", "--transport", "stdio")


def test_mcp_describes_what_the_upstream_server_advertises() -> None:
    # scripts/extract_mcp_metadata.py reads server.mcp; it must agree with server.json.
    described = json.loads((ROOT / "server.json").read_text(encoding="utf-8"))
    mcp = server.mcp

    async def listing() -> tuple[list[str], list[str], list[str]]:
        tools = await mcp.list_tools()
        resources = await mcp.list_resources()
        prompts = await mcp.list_prompts()
        return (
            sorted(t.name for t in tools),
            sorted(r.uri for r in resources),
            sorted(p.name for p in prompts),
        )

    tools, resources, prompts = asyncio.run(listing())
    assert tools == sorted(t["name"] for t in described["tools"])
    assert resources == sorted(r["uri"] for r in described.get("resources", []))
    assert prompts == sorted(p["name"] for p in described.get("prompts", []))
