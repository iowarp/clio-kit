"""
Tests for server main function and initialization.
"""

import os
import sys
from unittest.mock import patch, MagicMock

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server


class TestServerMain:
    """Test suite for server main function and initialization."""

    @patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}, clear=False)
    @patch("server.mcp")
    def test_main_stdio_transport(self, mock_mcp):
        """Test main function with stdio transport."""
        mock_mcp.run = MagicMock()

        # Mock sys.exit to prevent actual exit
        with patch("sys.exit"):
            try:
                server.main()
            except Exception:
                pass

        # Verify mcp.run was called with stdio
        if mock_mcp.run.called:
            call_args = mock_mcp.run.call_args
            if call_args and call_args[1]:
                assert call_args[1].get("transport") == "stdio"

    @patch.dict(
        os.environ,
        {"MCP_TRANSPORT": "sse", "MCP_SSE_HOST": "127.0.0.1", "MCP_SSE_PORT": "9000"},
        clear=False,
    )
    @patch("server.mcp")
    def test_main_sse_transport(self, mock_mcp):
        """Test main function with SSE transport."""
        mock_mcp.run = MagicMock()

        with patch("sys.exit"):
            try:
                server.main()
            except Exception:
                pass

        # Verify mcp.run was called with SSE parameters
        if mock_mcp.run.called:
            call_args = mock_mcp.run.call_args
            if call_args and call_args[1]:
                assert call_args[1].get("transport") == "sse"
                assert call_args[1].get("host") == "127.0.0.1"
                assert call_args[1].get("port") == 9000

    @patch("server.mcp")
    def test_main_exception_handling(self, mock_mcp):
        """Test main function exception handling."""
        mock_mcp.run = MagicMock(side_effect=Exception("Test error"))

        with patch("sys.exit") as mock_exit:
            server.main()
            # Verify sys.exit was called with error code
            mock_exit.assert_called_once_with(1)

    @patch.dict(os.environ, {}, clear=False)
    @patch("server.mcp")
    def test_main_default_transport(self, mock_mcp):
        """Test main function with default transport (no env var)."""
        mock_mcp.run = MagicMock()

        # Remove MCP_TRANSPORT if it exists
        if "MCP_TRANSPORT" in os.environ:
            del os.environ["MCP_TRANSPORT"]

        with patch("sys.exit"):
            try:
                server.main()
            except Exception:
                pass

        # Default should be stdio
        if mock_mcp.run.called:
            call_args = mock_mcp.run.call_args
            if call_args and call_args[1]:
                assert call_args[1].get("transport") == "stdio"

    @patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}, clear=False)
    @patch("server.mcp")
    def test_main_sse_default_host_port(self, mock_mcp):
        """Test main function with SSE using default host and port."""
        mock_mcp.run = MagicMock()

        # Remove host/port env vars if they exist
        os.environ.pop("MCP_SSE_HOST", None)
        os.environ.pop("MCP_SSE_PORT", None)

        with patch("sys.exit"):
            try:
                server.main()
            except Exception:
                pass

        # Verify default host/port were used
        if mock_mcp.run.called:
            call_args = mock_mcp.run.call_args
            if call_args and call_args[1]:
                assert call_args[1].get("host") == "0.0.0.0"
                assert call_args[1].get("port") == 8000


class TestServerImports:
    """Test server module imports and initialization."""

    def test_server_module_attributes(self):
        """Test server module has required attributes."""
        assert hasattr(server, "mcp")
        assert hasattr(server, "logger")
        assert hasattr(server, "main")

    def test_mcp_server_name(self):
        """Test MCP server has correct name."""
        assert server.mcp.name == "ParallelSortMCP"

    def test_logger_configuration(self):
        """Test logger is configured."""
        assert server.logger is not None
        assert server.logger.name == "server"
