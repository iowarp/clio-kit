"""The slicing skill that ships beside this server (plugins/clio-slicer/skills/slicing).

It names every tool the agent sees and the view's own tools, never a tool only the
view calls, and the typed error it passes on; it carries no instance literals.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest
from fastmcp import Client

from slicer_mcp.server import mcp
from slicer_mcp.view import APP_TOOLS

SKILL = (
    Path(__file__).resolve().parents[3]
    / "plugins"
    / "clio-slicer"
    / "skills"
    / "slicing"
    / "SKILL.md"
)

pytestmark = pytest.mark.skipif(not SKILL.is_file(), reason="not in a clio-kit checkout")


async def _tools() -> tuple[set[str], set[str]]:
    """(tools the agent sees, tools only the view calls)."""

    async with Client(mcp) as client:
        tools = await client.list_tools()
    view_only = {
        tool.name
        for tool in tools
        if ((tool.meta or {}).get("ui") or {}).get("visibility") == ["app"]
    }
    return {tool.name for tool in tools} - view_only, view_only


async def test_the_skill_names_the_live_tools_and_the_typed_error() -> None:
    agent, view_only = await _tools()
    assert view_only == {"slice_layers", "remember_slice_view"}
    named = set(re.findall(r"`(?:clio-slicer:)?([a-z_]+)`", SKILL.read_text(encoding="utf-8")))
    assert agent | set(APP_TOOLS) <= named
    assert not named & view_only
    assert {"slicer_missing", "SLICER_BIN"} <= set(re.findall(r"`(\w+)`", SKILL.read_text()))


def test_the_skill_is_short_and_carries_no_instance_literals() -> None:
    text = SKILL.read_text(encoding="utf-8")
    assert len(text.split()) < 1100
    for literal in (
        r"(?i)whistle",
        r"\bs[0-9a-f]{10}\b",
        r"/home/",
        r"\d+ min \d+ s",
        r"\d+\.\d+ g\b",
        r"(?i)print-lab",
    ):
        assert re.search(literal, text) is None, literal
