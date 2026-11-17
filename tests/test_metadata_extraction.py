"""Tests for publishing-metadata profile selection."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Protocol, cast

import pytest
from mcp.types import ToolAnnotations


class _ExtractorModule(Protocol):
    """Typed surface loaded from the repository maintenance script."""

    importlib: Any

    async def extract(self, module_path: str) -> dict[str, Any]: ...

    def extract_over_stdio(
        self, module_path: str, env: dict[str, str]
    ) -> dict[str, Any]: ...

    def upstream_entry(self, server_name: str) -> dict[str, Any] | None: ...


def _load_extractor() -> _ExtractorModule:
    path = Path(__file__).parents[1] / "scripts" / "extract_mcp_metadata.py"
    spec = importlib.util.spec_from_file_location("clio_kit_extract_mcp_metadata", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load metadata extractor: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return cast(_ExtractorModule, module)


extract_mcp_metadata = _load_extractor()


class _MetadataMcp:
    """Minimal async MCP metadata surface for extraction tests."""

    name = "profiled"
    instructions = "test"

    async def list_tools(self) -> list[Any]:
        return []

    async def list_resources(self) -> list[Any]:
        return []

    async def list_resource_templates(self) -> list[Any]:
        return []

    async def list_prompts(self) -> list[Any]:
        return []


def test_extract_selects_user_profile_before_listing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Registry manifests must describe the agent-facing user contract."""
    profiles: list[str] = []
    module = SimpleNamespace(
        mcp=_MetadataMcp(),
        MCP_METADATA_PROFILE="user",
        apply_tool_profile=profiles.append,
    )
    monkeypatch.setattr(
        extract_mcp_metadata.importlib, "import_module", lambda _: module
    )

    result = asyncio.run(extract_mcp_metadata.extract("example.server"))

    assert profiles == ["user"]
    assert result["name"] == "profiled"
    assert result["tools"] == []


def test_extract_preserves_tool_hints_for_readme_generation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class ToolMcp(_MetadataMcp):
        async def list_tools(self) -> list[Any]:
            return [
                SimpleNamespace(
                    name="inspect",
                    description="Inspect data.",
                    annotations=ToolAnnotations(
                        readOnlyHint=True, destructiveHint=False, idempotentHint=True
                    ),
                    tags={"inspection"},
                )
            ]

    monkeypatch.setattr(
        extract_mcp_metadata.importlib,
        "import_module",
        lambda _: SimpleNamespace(mcp=ToolMcp()),
    )
    result = asyncio.run(extract_mcp_metadata.extract("example.server"))
    hints = result["tools"][0]["annotations"]
    assert hints["readOnlyHint"] is True
    assert hints["destructiveHint"] is False
    assert hints["idempotentHint"] is True


FAKE_UPSTREAM = """
import json, os, sys

TOOLS = [
    {"name": "first", "description": "One.", "annotations": {"readOnlyHint": True}},
    {"name": "second", "_meta": {"fastmcp": {"tags": ["b", "a"]}}},
]


def main():
    for line in sys.stdin:
        message = json.loads(line)
        if "id" not in message:
            continue
        method, params = message["method"], message.get("params") or {}
        if method == "initialize":
            result = {
                "protocolVersion": params["protocolVersion"],
                "capabilities": {"tools": {}, "prompts": {}},
                "serverInfo": {"name": os.environ["FAKE_NAME"], "version": "9"},
                "instructions": "Upstream.",
            }
        elif method == "tools/list" and params.get("cursor") != "page-2":
            result = {"tools": TOOLS[:1], "nextCursor": "page-2"}
        elif method == "tools/list":
            result = {"tools": TOOLS[1:]}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "guide"}]}
        else:
            print(json.dumps({"jsonrpc": "2.0", "id": message["id"],
                              "error": {"code": -32601, "message": method}}), flush=True)
            continue
        print(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}),
              flush=True)
"""


def test_upstream_server_is_read_over_stdio(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A wrapper with no mcp object reports the tools its upstream lists, all pages."""
    (tmp_path / "fake_upstream.py").write_text(FAKE_UPSTREAM, encoding="utf-8")
    monkeypatch.setenv("PYTHONPATH", str(tmp_path))

    result = extract_mcp_metadata.extract_over_stdio(
        "fake_upstream", {"FAKE_NAME": "from-probe-environment"}
    )

    assert result["name"] == "from-probe-environment"
    assert result["instructions"] == "Upstream."
    assert [t["name"] for t in result["tools"]] == ["first", "second"]
    assert result["tools"][0]["annotations"] == {"readOnlyHint": True}
    assert result["tools"][1]["tags"] == ["a", "b"]
    assert result["resources"] == [] and result["resource_templates"] == []
    assert result["prompts"] == [{"name": "guide", "description": ""}]


def test_upstream_servers_come_from_the_inventory() -> None:
    entry = extract_mcp_metadata.upstream_entry("home-assistant")
    assert entry is not None and entry["package"] == "ha-mcp"
    assert "HOMEASSISTANT_URL" in entry["probe-environment"]
    assert extract_mcp_metadata.upstream_entry("hdf5") is None
