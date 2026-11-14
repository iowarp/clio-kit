"""Without the ADIOS2 bindings, BP5 tools fail with a typed, plainly worded error."""

from __future__ import annotations

import importlib

import pytest
from fastmcp.exceptions import ToolError

from adios_mcp import mcp_handlers


def _missing(name: str):
    def import_module(module_name: str):
        raise ModuleNotFoundError(f"No module named {name!r}", name=name)

    return import_module


def test_windows_reports_adios2_unavailable(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_handlers.importlib, "import_module", _missing("adios2"))
    monkeypatch.setattr(mcp_handlers.sys, "platform", "win32")

    with pytest.raises(mcp_handlers.Adios2Unavailable) as raised:
        mcp_handlers.bp5_read_variable_at_step.read_variable_at_step("f.bp", "v", 0)

    assert raised.value.reason == "adios2_unavailable"
    assert str(raised.value) == mcp_handlers.ADIOS2_UNAVAILABLE_ON_WINDOWS


def test_other_platforms_report_missing_bindings(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_handlers.importlib, "import_module", _missing("adios2"))
    monkeypatch.setattr(mcp_handlers.sys, "platform", "linux")

    with pytest.raises(mcp_handlers.Adios2Unavailable, match="aren't installed"):
        mcp_handlers.bp5_attributes.inspect_attributes("f.bp", None)


async def test_tool_surfaces_the_message(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_handlers.importlib, "import_module", _missing("adios2"))
    monkeypatch.setattr(mcp_handlers.sys, "platform", "win32")

    with pytest.raises(ToolError, match="aren't available on Windows"):
        await mcp_handlers.read_variable_at_step_handler("f.bp", "v", 0)


def test_other_missing_modules_are_not_relabelled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(mcp_handlers.importlib, "import_module", _missing("numpy"))

    with pytest.raises(ModuleNotFoundError):
        mcp_handlers.bp5_read_variable_at_step.read_variable_at_step("f.bp", "v", 0)


def test_listing_does_not_need_adios2() -> None:
    module = importlib.import_module("adios_mcp.implementation.bp5_list")
    assert callable(module.list_bp5)
