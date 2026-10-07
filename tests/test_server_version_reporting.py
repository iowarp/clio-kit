"""Each server reports its own release, not the FastMCP library's version.

Without an explicit ``version=`` FastMCP fills ``serverInfo.version`` with its
own, so a client cannot tell which server release it is talking to. Parsed
statically: importing 22 servers would need 22 locked environments.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VERSIONS = tomllib.loads((ROOT / "mcp-server-versions.toml").read_text())["servers"]


def reported_versions(server: str) -> list[object]:
    """The ``version`` argument of every FastMCP(...) call in a server's source."""
    found: list[object] = []
    for path in sorted((ROOT / "mcp-servers" / server / "src").rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", None))
            if name != "FastMCP":
                continue
            version = next((k.value for k in node.keywords if k.arg == "version"), None)
            found.append(getattr(version, "value", None))
    return found


@pytest.mark.parametrize("server", sorted(VERSIONS))
def test_server_reports_its_inventory_version(server: str) -> None:
    if not (ROOT / "mcp-servers" / server / "pyproject.toml").is_file():
        return  # a hosted Node or Go server states its version in its own SDK
    versions = reported_versions(server)
    assert versions, f"{server}: no FastMCP(...) constructor found under src/"
    assert set(versions) == {VERSIONS[server]}, (
        f"{server}: every FastMCP(...) must pass version={VERSIONS[server]!r} "
        f"(mcp-server-versions.toml); found {versions}"
    )
