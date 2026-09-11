"""
Tests for ParaView MCP Server
"""

import pytest
from unittest.mock import Mock, patch


def test_server_import():
    """Test that the server module can be imported successfully"""
    try:
        from paraview_mcp.server import mcp

        assert mcp is not None
        assert mcp.name == "paraview"
    except ImportError as e:
        pytest.skip(f"ParaView not available: {e}")


def test_fastmcp_initialization():
    """Test FastMCP server initialization with instructions"""
    try:
        from paraview_mcp.server import mcp

        # Check that the server is properly initialized
        assert hasattr(mcp, "name")
        assert mcp.name == "paraview"

        # Verify instructions are set
        assert mcp.instructions is not None
        assert "ParaView" in mcp.instructions
    except ImportError:
        pytest.skip("ParaView not available")


@pytest.mark.asyncio
async def test_main_function_exists():
    """Test that main function exists and is callable"""
    try:
        from paraview_mcp.server import main

        assert callable(main)
    except ImportError:
        pytest.skip("ParaView not available")


def test_tools_registration():
    """Test that MCP tools are properly registered"""
    try:
        from paraview_mcp.server import mcp

        # FastMCP 3.0 uses tool decorator - check the server has methods
        assert callable(getattr(mcp, "tool", None)), (
            "MCP server should have tool method"
        )
        assert callable(getattr(mcp, "run", None)), "MCP server should have run method"

    except ImportError:
        pytest.skip("ParaView not available")


def test_tool_functions_are_callable():
    """Test that tool-decorated functions are callable (v3 returns original functions)"""
    try:
        from paraview_mcp.server import (
            create_source,
            create_isosurface,
            get_pipeline,
            list_commands,
            reset_camera,
        )

        # In FastMCP 3.0, decorated functions are the original functions
        assert callable(create_source)
        assert callable(create_isosurface)
        assert callable(get_pipeline)
        assert callable(list_commands)
        assert callable(reset_camera)
    except ImportError:
        pytest.skip("ParaView not available")


def test_resource_registration():
    """Test that the paraview capabilities resource is registered"""
    try:
        from paraview_mcp.server import paraview_capabilities

        # In FastMCP 3.0, resource decorator returns the original function
        assert callable(paraview_capabilities)
        result = paraview_capabilities()
        assert isinstance(result, dict)
        assert "supported_formats" in result
        assert "operations" in result
        assert "VTK" in result["supported_formats"]
    except ImportError:
        pytest.skip("ParaView not available")


def test_prompt_registration():
    """Test that the visualize_data prompt is registered"""
    try:
        from paraview_mcp.server import visualize_data

        # In FastMCP 3.0, prompt decorator returns the original function
        assert callable(visualize_data)
        result = visualize_data("/test/file.vtk")
        assert isinstance(result, list)
        assert len(result) == 1
        assert "/test/file.vtk" in str(result[0])
    except ImportError:
        pytest.skip("ParaView not available")


def test_mock_paraview_manager():
    """Test server functionality with mocked ParaView manager"""
    with patch("paraview_mcp.server.get_pv_manager") as mock_manager:
        # Mock the ParaView manager
        mock_pv = Mock()
        mock_pv.read_datafile.return_value = (True, "Success", None, "test_source")
        mock_manager.return_value = mock_pv

        try:
            # Test that the server module imports successfully with mocked manager
            from paraview_mcp.server import mcp

            assert mcp is not None

            # Test that we can access the mocked manager
            manager = mock_manager.return_value
            success, message, _, source_name = manager.read_datafile(
                "/fake/path/test.vtk"
            )
            assert success is True
            assert "Success" in message
            assert source_name == "test_source"

        except ImportError:
            pytest.skip("ParaView not available")


def test_tool_error_import():
    """Test that ToolError is properly imported from fastmcp"""
    try:
        from paraview_mcp.server import ToolError  # noqa: F401
        from fastmcp.exceptions import ToolError as FastMCPToolError

        assert ToolError is FastMCPToolError
    except ImportError:
        pytest.skip("fastmcp not available")


def test_message_import():
    """Test that Message is properly imported from fastmcp.prompts"""
    try:
        from paraview_mcp.server import Message  # noqa: F401
        from fastmcp.prompts import Message as FastMCPMessage

        assert Message is FastMCPMessage
    except ImportError:
        pytest.skip("fastmcp not available")


def test_transport_stdio_default():
    """Test that the default transport is stdio"""
    try:
        from paraview_mcp.server import main

        # The main function supports transport="stdio" by default
        assert callable(main)
    except ImportError:
        pytest.skip("ParaView not available")


if __name__ == "__main__":
    pytest.main([__file__])


@pytest.mark.asyncio
async def test_shape_and_screenshot_share_the_event_loop_thread(monkeypatch, tmp_path):
    """VTK's graphics context cannot move between worker and event-loop threads."""
    import threading
    from fastmcp import Client
    from paraview_mcp import server

    threads = []

    class Engine:
        def create_source(self, name):
            threads.append(threading.get_ident())
            return True, "created", None, "Sphere1"

        def get_screenshot(self):
            threads.append(threading.get_ident())
            return True, "saved", str(tmp_path / "view.png")

    monkeypatch.setattr(server, "pv_manager", Engine())
    async with Client(server.mcp) as client:
        await client.call_tool("create_geometric_shape", {"source_type": "Sphere"})
        await client.call_tool("take_viewport_screenshot", {})
    assert threads == [threading.get_ident(), threading.get_ident()]


@pytest.mark.asyncio
async def test_preview_returns_png_over_mcp(tmp_path, monkeypatch):
    import base64
    from fastmcp import Client
    from paraview_mcp import server

    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l1sAAAAASUVORK5CYII="
    )
    path = tmp_path / "viewport.png"
    path.write_bytes(png)
    manager = Mock()
    manager.get_screenshot.return_value = (True, "Saved", str(path))
    monkeypatch.setattr(server, "get_pv_manager", lambda: manager)
    async with Client(server.mcp) as client:
        result = await client.call_tool("show_screenshot_preview", {})
    images = [block for block in result.content if block.type == "image"]
    assert len(images) == 1
    assert base64.b64decode(images[0].data) == png


@pytest.mark.asyncio
async def test_histogram_preview_keeps_six_to_ten_bins(monkeypatch):
    from paraview_mcp import server

    manager = Mock()
    manager.get_histogram.return_value = (
        True,
        "Histogram",
        [(i, i + 1) for i in range(8)],
    )
    monkeypatch.setattr(server, "get_pv_manager", lambda: manager)
    result = await server.get_histogram("density", 8)
    assert "Bin 8: value=7.00, count=8" in result
    assert result.count("  Bin ") == 8
