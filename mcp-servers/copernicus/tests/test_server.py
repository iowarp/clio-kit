"""The wrapper runs the pinned copernicus-mcp and changes nothing it says."""

from __future__ import annotations

import asyncio
import json
import tempfile
import tomllib
from pathlib import Path

from stdio_session import session

from copernicus_kit_mcp import server

ROOT = Path(__file__).resolve().parents[1]
MODULE = "copernicus_kit_mcp.server"
PINNED = "0.7.2"
#: Keep the upstream's job database and cache out of the home directory.
STATE = tempfile.mkdtemp(prefix="copernicus-mcp-test-")
ENV: dict[str, str] = {
    "COPERNICUS_MCP_CACHE_DIR": f"{STATE}/cache",
    "COPERNICUS_MCP_STATE_DB": f"{STATE}/state.sqlite",
}


def test_it_pins_one_exact_upstream_release_with_both_backends() -> None:
    manifest = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert manifest["project"]["dependencies"] == [
        f"{server.UPSTREAM}[cmems,cds]=={PINNED}"
    ]
    lock = tomllib.loads((ROOT / "uv.lock").read_text(encoding="utf-8"))
    locked = {package["name"]: package["version"] for package in lock["package"]}
    assert locked[server.UPSTREAM] == PINNED
    assert "copernicusmarine" in locked and "cdsapi" in locked


def test_the_command_is_the_upstream_script() -> None:
    argv = server.command([])
    assert Path(argv[0]).is_file()
    assert Path(argv[0]).stem == server.SCRIPT
    assert argv[1:] == ["serve"]


def tool_names(env: dict[str, str]) -> set[str]:
    replies = session(MODULE, [{"method": "tools/list"}], env={**ENV, **env})
    return {tool["name"] for tool in replies[1]["result"]["tools"]}


def test_the_upstream_server_answers_with_its_marine_tools() -> None:
    names = tool_names({})
    assert {
        "marine_search_datasets",
        "marine_describe_dataset",
        "marine_estimate_subset",
        "marine_subset_dataset",
        "marine_check_status",
        "marine_get_files",
    } <= names
    # Upstream offers CDS tools only once a CDS key resolves.
    assert not any(name.startswith("cds_") for name in names)


def test_cds_tools_appear_with_a_cds_key() -> None:
    # A placeholder: listing tools sends no request to the CDS.
    names = tool_names({"CDSAPI_KEY": "00000000-0000-0000-0000-000000000000"})
    assert {
        "cds_search_datasets",
        "cds_submit_request",
        "cds_download_request_result",
    } <= names


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
