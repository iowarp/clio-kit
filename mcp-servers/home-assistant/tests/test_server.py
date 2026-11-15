"""The wrapper runs the pinned ha-mcp and changes nothing it says."""

from __future__ import annotations

import tomllib
from pathlib import Path

from stdio_session import session

from home_assistant_mcp import server

ROOT = Path(__file__).resolve().parents[1]
MODULE = "home_assistant_mcp.server"
PINNED = "9.2.0"
ENV: dict[str, str] = {
    "HOMEASSISTANT_URL": "http://127.0.0.1:9",
    "HOMEASSISTANT_TOKEN": "listing-needs-no-home",
    # ha-mcp otherwise starts its detached settings page; a test leaves nothing behind.
    "HA_MCP_DISABLE_SETTINGS_UI": "1",
}


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
    assert replies[0]["result"]["serverInfo"]["name"] == "ha-mcp"
    names = {tool["name"] for tool in replies[1]["result"]["tools"]}
    assert {
        "ha_get_overview",
        "ha_search",
        "ha_get_state",
        "ha_get_history",
        "ha_call_service",
    } <= names
