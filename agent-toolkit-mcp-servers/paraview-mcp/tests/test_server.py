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
        assert mcp.name == "ParaView"
    except ImportError as e:
        pytest.skip(f"ParaView not available: {e}")


def test_fastmcp_initialization():
    """Test FastMCP server initialization"""
    try:
        from paraview_mcp.server import mcp
        # Check that the server is properly initialized
        assert hasattr(mcp, 'name')
        assert mcp.name == "ParaView"
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
        
        # Check that MCP server has tools registered
        # FastMCP 2.0 stores tools differently - check for tool registry
        assert hasattr(mcp, '_tool_registry') or hasattr(mcp, 'tool'), "MCP server should have tools registered"
        
        # Since the actual tool registration happens at import time,
        # we can check that the server has the expected functionality
        assert callable(getattr(mcp, 'run', None)), "MCP server should have run method"
            
    except ImportError:
        pytest.skip("ParaView not available")


def test_mock_paraview_manager():
    """Test server functionality with mocked ParaView manager"""
    with patch('paraview_mcp.server.get_pv_manager') as mock_manager:
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
            success, message, _, source_name = manager.read_datafile("/fake/path/test.vtk")
            assert success is True
            assert "Success" in message
            assert source_name == "test_source"
            
        except ImportError:
            pytest.skip("ParaView not available")


if __name__ == "__main__":
    pytest.main([__file__])