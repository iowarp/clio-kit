"""The wrapper runs the pinned ltspice-mcp and changes nothing it says."""

from __future__ import annotations

import tomllib
from pathlib import Path

from stdio_session import session

from spice_mcp import server

ROOT = Path(__file__).resolve().parents[1]
MODULE = "spice_mcp.server"
PINNED = "0.8.0"
VIEW = "ui://ltspice-mcp/plot"
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
    assert replies[0]["result"]["serverInfo"]["name"] == "ltspice-mcp"
    names = {tool["name"] for tool in replies[1]["result"]["tools"]}
    assert {"plot_waveform", "run_experiments", "inspect", "analyze_results"} <= names


def test_the_waveform_view_passes_through() -> None:
    replies = session(
        MODULE,
        [{"method": "tools/list"}, {"method": "resources/read", "params": {"uri": VIEW}}],
        env=ENV,
    )
    plot = next(t for t in replies[1]["result"]["tools"] if t["name"] == "plot_waveform")
    assert plot["_meta"]["ui"]["resourceUri"] == VIEW
    (content,) = replies[2]["result"]["contents"]
    assert content["mimeType"] == "text/html;profile=mcp-app"
    assert "<html" in content["text"].lower()
