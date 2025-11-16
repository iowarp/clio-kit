"""Tests for the FastMCP server and tool integration."""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from unittest.mock import patch
from parquet_mcp import server


class TestServerToolsIntegration:
    """Test FastMCP tool integration with real files."""

    @pytest.fixture
    def simple_test_file(self, tmp_path):
        """Create a simple test file for integration testing."""
        file_path = tmp_path / "simple.parquet"
        table = pa.table(
            {
                "id": [1, 2, 3, 4, 5],
                "value": [10, 20, 30, 40, 50],
            }
        )
        pq.write_table(table, file_path)
        return str(file_path)

    def test_summarize_tool_exists(self):
        """Test that summarize_tool is registered."""
        assert hasattr(server, "summarize_tool")

    def test_read_slice_tool_exists(self):
        """Test that read_slice_tool is registered."""
        assert hasattr(server, "read_slice_tool")

    def test_get_column_preview_tool_exists(self):
        """Test that get_column_preview_tool is registered."""
        assert hasattr(server, "get_column_preview_tool")

    def test_aggregate_column_tool_exists(self):
        """Test that aggregate_column_tool is registered."""
        assert hasattr(server, "aggregate_column_tool")


class TestServerMain:
    """Test server main function."""

    def test_main_function_exists(self):
        """Test that main function is defined."""
        assert hasattr(server, "main")
        assert callable(server.main)

    def test_main_initializes_mcp_server(self):
        """Test that main function initializes the MCP server."""
        with patch.object(server.mcp, "run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            try:
                server.main()
            except (KeyboardInterrupt, SystemExit):
                pass

            mock_run.assert_called_once_with(transport="stdio")

    def test_main_handles_keyboard_interrupt(self):
        """Test that main handles KeyboardInterrupt gracefully."""
        with patch.object(server.mcp, "run") as mock_run:
            mock_run.side_effect = KeyboardInterrupt()

            # Should not raise an exception
            try:
                server.main()
            except SystemExit:
                pytest.fail("main() should not exit on KeyboardInterrupt")

    def test_main_handles_exception(self):
        """Test that main handles exceptions and exits with error code."""
        with patch.object(server.mcp, "run") as mock_run:
            mock_run.side_effect = RuntimeError("Test error")

            with pytest.raises(SystemExit) as exc_info:
                server.main()

            assert exc_info.value.code == 1

    def test_main_configures_logging(self):
        """Test that main configures logging."""
        with patch("logging.basicConfig") as mock_config:
            with patch.object(server.mcp, "run") as mock_run:
                mock_run.side_effect = KeyboardInterrupt()

                try:
                    server.main()
                except (KeyboardInterrupt, SystemExit):
                    pass

                # Check that basicConfig was called
                mock_config.assert_called_once()
                call_kwargs = mock_config.call_args[1]
                assert "level" in call_kwargs
                assert "format" in call_kwargs


class TestMCPServerInstance:
    """Test MCP server instance configuration."""

    def test_mcp_server_is_fastmcp_instance(self):
        """Test that mcp is a FastMCP instance."""
        from fastmcp import FastMCP

        assert isinstance(server.mcp, FastMCP)

    def test_mcp_server_name(self):
        """Test that MCP server has correct name."""
        assert server.mcp.name == "parquet-mcp"

    def test_mcp_server_has_tools_registered(self):
        """Test that MCP server has tools registered."""
        # Verify that our tool functions are defined
        assert hasattr(server, "summarize_tool")
        assert hasattr(server, "read_slice_tool")
        assert hasattr(server, "get_column_preview_tool")
        assert hasattr(server, "aggregate_column_tool")

    def test_tools_are_function_tool_objects(self):
        """Test that tools are wrapped in FunctionTool objects by FastMCP."""
        # FastMCP wraps decorated functions in FunctionTool objects
        from fastmcp.tools import FunctionTool

        assert isinstance(server.summarize_tool, FunctionTool)
        assert isinstance(server.read_slice_tool, FunctionTool)
        assert isinstance(server.get_column_preview_tool, FunctionTool)
        assert isinstance(server.aggregate_column_tool, FunctionTool)

    def test_tool_names(self):
        """Test that tools have correct names."""
        assert server.summarize_tool.name == "summarize_tool"
        assert server.read_slice_tool.name == "read_slice_tool"
        assert server.get_column_preview_tool.name == "get_column_preview_tool"
        assert server.aggregate_column_tool.name == "aggregate_column_tool"

    def test_tool_descriptions(self):
        """Test that tools have descriptions."""
        assert server.summarize_tool.description is not None
        assert len(server.summarize_tool.description) > 0
        assert server.read_slice_tool.description is not None
        assert len(server.read_slice_tool.description) > 0
        assert server.get_column_preview_tool.description is not None
        assert len(server.get_column_preview_tool.description) > 0
        assert server.aggregate_column_tool.description is not None
        assert len(server.aggregate_column_tool.description) > 0
