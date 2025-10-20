"""
Fixed server tests - basic functionality tests that work
"""

import os
import sys
import pytest
from unittest.mock import patch

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestServerFixed:
    """Fixed server tests that actually work"""

    def test_server_initialization(self):
        """Test server initialization"""
        import server

        assert hasattr(server, "mcp")
        assert server.mcp is not None
        assert hasattr(server.mcp, "name")
        assert server.mcp.name == "NodeHardware-MCP-SystemMonitoring"

    def test_server_tools_exist(self):
        """Test that all expected tools exist"""
        import server

        # Test that tools exist as attributes
        tools = [
            "get_cpu_info_tool",
            "get_memory_info_tool",
            "get_system_info_tool",
            "get_disk_info_tool",
            "get_network_info_tool",
            "get_gpu_info_tool",
            "get_sensor_info_tool",
            "get_process_info_tool",
            "get_performance_info_tool",
            "get_remote_node_info_tool",
        ]

        for tool_name in tools:
            assert hasattr(server, tool_name), f"Tool {tool_name} should exist"
            tool = getattr(server, tool_name)
            assert tool is not None, f"Tool {tool_name} should not be None"

    def test_server_logger(self):
        """Test server logger"""
        import server

        assert hasattr(server, "logger")
        assert server.logger is not None

    def test_server_exception(self):
        """Test custom exception"""
        import server

        # Test exception exists
        assert hasattr(server, "NodeHardwareMCPError")

        # Test exception works
        try:
            raise server.NodeHardwareMCPError("Test error")
        except server.NodeHardwareMCPError as e:
            assert str(e) == "Test error"

    def test_server_main_function(self):
        """Test server main function"""
        import server

        # Test main function exists
        assert hasattr(server, "main")

        # Test with stdio transport
        with patch.dict("os.environ", {"MCP_TRANSPORT": "stdio"}):
            with patch.object(server.mcp, "run") as mock_run:
                server.main()
                mock_run.assert_called_once_with(transport="stdio")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
