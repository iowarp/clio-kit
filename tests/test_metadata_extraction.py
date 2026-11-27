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
