"""The wrapper runs the pinned earthdata-mcp-server and changes nothing it says."""

from __future__ import annotations

import sys
from pathlib import Path

from stdio_session import session

if sys.version_info >= (3, 11):
    import tomllib
else:  # requires-python is >=3.10; pytest itself brings tomli there
    import tomli as tomllib

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
