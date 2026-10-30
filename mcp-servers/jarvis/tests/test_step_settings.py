"""Config keys that collide with the handler's own parameters (issue #385).

Each step tool splats its caller's config into ``append_pkg`` / ``configure_pkg``
beside their explicit parameters. Before the guard, a config key such as
``target`` reached Python as a duplicate keyword and surfaced as a raw
``TypeError`` ("got multiple values for keyword argument 'target'").
"""

from unittest.mock import AsyncMock, patch

import pytest
from fastmcp.exceptions import ToolError

from jarvis_mcp.server import (
    append_pkg_tool,
    configure_pkg_tool,
    jarvis_add_step_tool,
    jarvis_edit_step_tool,
)


@pytest.mark.asyncio
async def test_add_step_refuses_target_inside_config() -> None:
    """The #385 repro: a client that also carries 'target' inside config."""
    with patch("jarvis_mcp.server.append_pkg", new_callable=AsyncMock) as handler:
        with pytest.raises(ToolError) as exc_info:
            await jarvis_add_step_tool(
                "pipe",
                "darshan",
                target="app",
                config={"target": "app", "log_dir": "/tmp/darshan"},
            )

    message = str(exc_info.value)
    assert "'target' is not a package setting" in message
    assert "top-level 'target' parameter" in message
    handler.assert_not_called()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("key", "expected"),
    [
        ("pkg_id", "top-level 'step_id' parameter"),
        ("pkg_type", "top-level 'package_name' parameter"),
        ("pipeline_id", "top-level 'pipeline_id' parameter"),
        ("do_configure", "'do_configure' is not a package setting and cannot be set"),
        ("agent_visible_only", "'agent_visible_only' is not a package setting and"),
    ],
)
async def test_add_step_names_where_each_reserved_key_belongs(
    key: str, expected: str
) -> None:
    with patch("jarvis_mcp.server.append_pkg", new_callable=AsyncMock) as handler:
        with pytest.raises(ToolError, match=expected):
            await jarvis_add_step_tool("pipe", "builtin.ior", config={key: "x"})
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_add_step_reports_every_clash_at_once() -> None:
    with patch("jarvis_mcp.server.append_pkg", new_callable=AsyncMock):
        with pytest.raises(ToolError) as exc_info:
            await jarvis_add_step_tool(
                "pipe", "darshan", config={"target": "app", "pkg_id": "d"}
            )
    message = str(exc_info.value)
    assert "'pkg_id'" in message and "'target'" in message


@pytest.mark.asyncio
async def test_add_step_passes_ordinary_settings_through() -> None:
    with patch("jarvis_mcp.server.append_pkg", new_callable=AsyncMock) as handler:
        handler.return_value = {"pipeline_id": "pipe"}
        await jarvis_add_step_tool(
            "pipe", "darshan", target="app", config={"log_dir": "/tmp/d"}
        )
    handler.assert_awaited_once_with(
        "pipe",
        "darshan",
        pkg_id=None,
        do_configure=True,
        agent_visible_only=True,
        target="app",
        log_dir="/tmp/d",
    )


@pytest.mark.asyncio
async def test_edit_step_refuses_reserved_keys_and_keeps_its_own_checks() -> None:
    with patch("jarvis_mcp.server.configure_pkg", new_callable=AsyncMock) as handler:
        with pytest.raises(ToolError, match="top-level 'step_id' parameter"):
            await jarvis_edit_step_tool("pipe", "ior", config={"pkg_id": "other"})
        with pytest.raises(ToolError, match="config is required when operation='edit'"):
            await jarvis_edit_step_tool("pipe", "ior")
        with pytest.raises(ToolError, match="not accepted when operation='remove'"):
            await jarvis_edit_step_tool(
                "pipe", "ior", config={"nprocs": 2}, operation="remove"
            )
        handler.return_value = {"pipeline_id": "pipe"}
        await jarvis_edit_step_tool("pipe", "ior", config={"nprocs": 2})
    handler.assert_awaited_once_with("pipe", "ior", agent_visible_only=True, nprocs=2)


@pytest.mark.asyncio
async def test_legacy_append_pkg_names_its_own_parameters() -> None:
    with patch("jarvis_mcp.server.append_pkg", new_callable=AsyncMock) as handler:
        with pytest.raises(ToolError, match="top-level 'do_configure' parameter"):
            await append_pkg_tool("pipe", "ior", extra_args={"do_configure": False})
        with pytest.raises(ToolError, match="top-level 'pkg_id' parameter"):
            await append_pkg_tool("pipe", "ior", extra_args={"pkg_id": "x"})
    handler.assert_not_called()


@pytest.mark.asyncio
async def test_legacy_configure_pkg_refuses_reserved_keys() -> None:
    with patch("jarvis_mcp.server.configure_pkg", new_callable=AsyncMock) as handler:
        with pytest.raises(ToolError, match="top-level 'pkg_id' parameter"):
            await configure_pkg_tool("pipe", "ior", extra_args={"pkg_id": "x"})
        handler.return_value = {"pipeline_id": "pipe"}
        await configure_pkg_tool("pipe", "ior", extra_args={"nprocs": 4})
    handler.assert_awaited_once_with("pipe", "ior", nprocs=4)
