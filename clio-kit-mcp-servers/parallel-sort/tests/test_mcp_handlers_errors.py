"""
Tests for MCP handlers error paths and exception handling.
"""

import pytest
import os
import sys
from unittest.mock import patch

# Add the src directory to the path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from mcp_handlers import (
    sort_log_handler,
    parallel_sort_handler,
    analyze_statistics_handler,
    detect_patterns_handler,
    filter_logs_handler,
    filter_time_range_handler,
    filter_level_handler,
    filter_keyword_handler,
    filter_preset_handler,
    export_json_handler,
    export_csv_handler,
    export_text_handler,
    summary_report_handler,
)


class TestMCPHandlersErrorPaths:
    """Test suite for MCP handlers error handling paths."""

    @pytest.mark.asyncio
    @patch("mcp_handlers.sort_log_by_timestamp")
    async def test_sort_log_handler_exception_path(self, mock_sort):
        """Test sort_log_handler exception handling."""
        mock_sort.side_effect = RuntimeError("Unexpected error")

        result = await sort_log_handler("test.log")

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "sort_log"
        assert result["_meta"]["error"] == "RuntimeError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.parallel_sort_large_file")
    async def test_parallel_sort_handler_exception_path(self, mock_sort):
        """Test parallel_sort_handler exception handling."""
        mock_sort.side_effect = ValueError("Invalid chunk size")

        result = await parallel_sort_handler("test.log", 100, 4)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "parallel_sort"
        assert result["_meta"]["error"] == "ValueError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.analyze_log_statistics")
    async def test_analyze_statistics_handler_exception_path(self, mock_analyze):
        """Test analyze_statistics_handler exception handling."""
        mock_analyze.side_effect = IOError("File read error")

        result = await analyze_statistics_handler("test.log")

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "analyze_statistics"
        assert result["_meta"]["error"] == "OSError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.detect_patterns")
    async def test_detect_patterns_handler_exception_path(self, mock_detect):
        """Test detect_patterns_handler exception handling."""
        mock_detect.side_effect = KeyError("Missing pattern config")

        result = await detect_patterns_handler("test.log", None)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "detect_patterns"
        assert result["_meta"]["error"] == "KeyError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.filter_logs")
    async def test_filter_logs_handler_exception_path(self, mock_filter):
        """Test filter_logs_handler exception handling."""
        mock_filter.side_effect = TypeError("Invalid filter condition")

        result = await filter_logs_handler("test.log", [], "and")

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "filter_logs"
        assert result["_meta"]["error"] == "TypeError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.filter_by_time_range")
    async def test_filter_time_range_handler_exception_path(self, mock_filter):
        """Test filter_time_range_handler exception handling."""
        mock_filter.side_effect = ValueError("Invalid time format")

        result = await filter_time_range_handler("test.log", "invalid", "invalid")

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "filter_time_range"
        assert result["_meta"]["error"] == "ValueError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.filter_by_log_level")
    async def test_filter_level_handler_exception_path(self, mock_filter):
        """Test filter_level_handler exception handling."""
        mock_filter.side_effect = AttributeError("Missing level attribute")

        result = await filter_level_handler("test.log", "ERROR", False)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "filter_level"
        assert result["_meta"]["error"] == "AttributeError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.filter_by_keyword")
    async def test_filter_keyword_handler_exception_path(self, mock_filter):
        """Test filter_keyword_handler exception handling."""
        mock_filter.side_effect = IndexError("Invalid keyword index")

        result = await filter_keyword_handler("test.log", "error", False, False)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "filter_keyword"
        assert result["_meta"]["error"] == "IndexError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.apply_filter_preset")
    async def test_filter_preset_handler_exception_path(self, mock_filter):
        """Test filter_preset_handler exception handling."""
        mock_filter.side_effect = LookupError("Preset not found")

        result = await filter_preset_handler("test.log", "invalid_preset")

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "filter_preset"
        assert result["_meta"]["error"] == "LookupError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.export_to_json")
    async def test_export_json_handler_exception_path(self, mock_export):
        """Test export_json_handler exception handling."""
        mock_export.side_effect = TypeError("Cannot serialize object")

        result = await export_json_handler({"data": "test"}, True)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "export_json"
        assert result["_meta"]["error"] == "TypeError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.export_to_csv")
    async def test_export_csv_handler_exception_path(self, mock_export):
        """Test export_csv_handler exception handling."""
        mock_export.side_effect = ValueError("Invalid CSV format")

        result = await export_csv_handler({"data": "test"}, True)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "export_csv"
        assert result["_meta"]["error"] == "ValueError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.export_to_text")
    async def test_export_text_handler_exception_path(self, mock_export):
        """Test export_text_handler exception handling."""
        mock_export.side_effect = UnicodeEncodeError("utf-8", "", 0, 1, "Cannot encode")

        result = await export_text_handler({"data": "test"}, True)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "export_text"
        assert result["_meta"]["error"] == "UnicodeEncodeError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.export_summary_report")
    async def test_summary_report_handler_exception_path(self, mock_export):
        """Test summary_report_handler exception handling."""
        mock_export.side_effect = RuntimeError("Report generation failed")

        result = await summary_report_handler({"data": "test"})

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["tool"] == "summary_report"
        assert result["_meta"]["error"] == "RuntimeError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.sort_log_by_timestamp")
    async def test_sort_log_handler_generic_exception(self, mock_sort):
        """Test sort_log_handler with generic Exception."""
        mock_sort.side_effect = Exception("Generic error")

        result = await sort_log_handler("test.log")

        assert "content" in result
        assert result["isError"] is True
        assert "error" in result["content"][0]["text"]

    @pytest.mark.asyncio
    @patch("mcp_handlers.parallel_sort_large_file")
    async def test_parallel_sort_handler_memory_error(self, mock_sort):
        """Test parallel_sort_handler with MemoryError."""
        mock_sort.side_effect = MemoryError("Out of memory")

        result = await parallel_sort_handler("test.log", 100, 4)

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["error"] == "MemoryError"

    @pytest.mark.asyncio
    @patch("mcp_handlers.filter_logs")
    async def test_filter_logs_handler_permission_error(self, mock_filter):
        """Test filter_logs_handler with PermissionError."""
        mock_filter.side_effect = PermissionError("Access denied")

        result = await filter_logs_handler("test.log", [], "and")

        assert "content" in result
        assert result["isError"] is True
        assert result["_meta"]["error"] == "PermissionError"
