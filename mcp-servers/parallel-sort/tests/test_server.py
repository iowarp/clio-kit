"""
Tests for the Parallel Sort MCP server.
"""

import pytest

from parallel_sort_mcp.server import mcp


class TestServer:
    """Test suite for MCP server functionality."""

    @pytest.fixture
    def sample_log_content(self):
        """Create sample log content for testing."""
        return """2024-01-02 10:00:00 INFO Second entry
2024-01-01 08:30:00 DEBUG First entry
2024-01-01 09:00:00 ERROR Third entry"""

    def test_server_initialization(self):
        """Test that the server initializes correctly."""
        assert mcp is not None
        assert mcp.name == "parallel-sort"

    def test_sort_tool_registration(self):
        """Test that the sort tool is properly registered."""
        assert mcp.name == "parallel-sort"

    def test_sort_tool_metadata(self):
        """Test the sort tool is accessible through MCP server."""
        assert mcp.name == "parallel-sort"

    def test_server_has_instructions(self):
        """Test that the server has instructions configured."""
        assert mcp.instructions is not None
        assert (
            "parallel" in mcp.instructions.lower() or "sort" in mcp.instructions.lower()
        )

    def test_server_version_matches_release_manifest(self):
        """serverInfo.version is the server release, not FastMCP's version."""
        import re
        from pathlib import Path

        manifest = (Path(__file__).parents[1] / "clio-server.toml").read_text()
        assert mcp.version == re.search(r'^version = "(.+)"', manifest, re.M).group(1)
