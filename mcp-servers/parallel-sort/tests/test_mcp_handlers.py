"""
Tests for MCP handlers.
"""

import pytest
from fastmcp.exceptions import ToolError
import tempfile
import os
from parallel_sort_mcp.mcp_handlers import (
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


class TestMCPHandlers:
    """Test suite for MCP handler functionality."""

    @pytest.fixture
    def sample_log_content(self):
        """Create sample log content for testing."""
        return """2024-01-02 10:00:00 INFO Second entry
2024-01-01 08:30:00 DEBUG First entry
2024-01-01 09:00:00 ERROR Third entry"""

    @pytest.fixture
    def sample_log_file(self, sample_log_content):
        """Create a temporary log file for testing."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write(sample_log_content)
            temp_path = f.name
        yield temp_path
        if os.path.exists(temp_path):
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_sort_log_handler_success(self, sample_log_content):
        """Test successful log sorting through MCP handler."""
        test_content = """2024-01-02 10:00:00 INFO Second entry
2024-01-01 08:30:00 DEBUG First entry"""

        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            f.write(test_content)
            temp_path = f.name

        try:
            result = await sort_log_handler(temp_path)

            # Should return the actual sort result, not MCP error format
            assert "error" not in result or result.get("error") is None
            assert "sorted_lines" in result
            assert result["total_lines"] == 2
            assert result["valid_lines"] == 2

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_sort_log_handler_file_not_found(self):
        """Test MCP handler with non-existent file."""
        with pytest.raises(ToolError):
            await sort_log_handler("/nonexistent/file.log")

    @pytest.mark.asyncio
    async def test_sort_log_handler_empty_file(self):
        """Test MCP handler with empty file."""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".log") as f:
            temp_path = f.name

        try:
            result = await sort_log_handler(temp_path)

            assert "error" not in result or result.get("error") is None
            assert result["total_lines"] == 0
            assert result["sorted_lines"] == []
            assert "empty" in result["message"].lower()

        finally:
            os.unlink(temp_path)

    @pytest.mark.asyncio
    async def test_parallel_sort_handler_success(self, sample_log_file):
        """Test parallel sort handler with valid input."""
        output_file = tempfile.mktemp(suffix=".log")
        try:
            result = await parallel_sort_handler(
                sample_log_file, output_file, chunk_size_mb=1, num_workers=2
            )
            assert "error" not in result or result.get("error") is None
        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    @pytest.mark.asyncio
    async def test_parallel_sort_handler_file_not_found(self):
        """Test parallel sort handler with non-existent file."""
        with pytest.raises(ToolError):
            await parallel_sort_handler("/nonexistent/file.log", "/tmp/output.log")

    @pytest.mark.asyncio
    async def test_analyze_statistics_handler_success(self, sample_log_file):
        """Test analyze statistics handler."""
        result = await analyze_statistics_handler(sample_log_file)
        assert "error" not in result or result.get("error") is None
        assert "statistics" in result or "total_lines" in result

    @pytest.mark.asyncio
    async def test_analyze_statistics_handler_error(self):
        """Test analyze statistics handler with error."""
        with pytest.raises(ToolError):
            await analyze_statistics_handler("/nonexistent/file.log")

    @pytest.mark.asyncio
    async def test_detect_patterns_handler_success(self, sample_log_file):
        """Test detect patterns handler."""
        result = await detect_patterns_handler(sample_log_file, None)
        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_detect_patterns_handler_error(self):
        """Test detect patterns handler with error."""
        with pytest.raises(ToolError):
            await detect_patterns_handler("/nonexistent/file.log", None)

    @pytest.mark.asyncio
    async def test_filter_logs_handler_success(self, sample_log_file):
        """Test filter logs handler."""
        conditions = [{"field": "level", "operator": "equals", "value": "ERROR"}]
        result = await filter_logs_handler(sample_log_file, conditions, "and")
        assert "filtered_lines" in result
        assert len(result["filtered_lines"]) > 0

    @pytest.mark.asyncio
    async def test_filter_logs_handler_error(self):
        """Test filter logs handler with error."""
        conditions = [{"field": "level", "operator": "equals", "value": "ERROR"}]
        with pytest.raises(ToolError):
            await filter_logs_handler("/nonexistent/file.log", conditions, "and")

    @pytest.mark.asyncio
    async def test_filter_time_range_handler_success(self, sample_log_file):
        """Test filter time range handler."""
        result = await filter_time_range_handler(
            sample_log_file, "2024-01-01 08:00:00", "2024-01-01 10:00:00"
        )
        assert "filtered_lines" in result

    @pytest.mark.asyncio
    async def test_filter_time_range_handler_error(self):
        """Test filter time range handler with error."""
        with pytest.raises(ToolError):
            await filter_time_range_handler(
                "/nonexistent/file.log", "2024-01-01 08:00:00", "2024-01-01 10:00:00"
            )

    @pytest.mark.asyncio
    async def test_filter_level_handler_success(self, sample_log_file):
        """Test filter level handler."""
        result = await filter_level_handler(sample_log_file, "ERROR")
        assert "filtered_lines" in result

    @pytest.mark.asyncio
    async def test_filter_level_handler_error(self):
        """Test filter level handler with error."""
        with pytest.raises(ToolError):
            await filter_level_handler("/nonexistent/file.log", "ERROR")

    @pytest.mark.asyncio
    async def test_filter_keyword_handler_success(self, sample_log_file):
        """Test filter keyword handler."""
        result = await filter_keyword_handler(sample_log_file, "entry")
        assert "filtered_lines" in result

    @pytest.mark.asyncio
    async def test_filter_keyword_handler_error(self):
        """Test filter keyword handler with error."""
        with pytest.raises(ToolError):
            await filter_keyword_handler("/nonexistent/file.log", "entry")

    @pytest.mark.asyncio
    async def test_filter_preset_handler_success(self, sample_log_file):
        """Test filter preset handler."""
        result = await filter_preset_handler(sample_log_file, "errors_only")
        assert "filtered_lines" in result

    @pytest.mark.asyncio
    async def test_filter_preset_handler_error(self):
        """Test filter preset handler with error."""
        with pytest.raises(ToolError):
            await filter_preset_handler("/nonexistent/file.log", "errors_only")

    @pytest.mark.asyncio
    async def test_export_json_handler_success(self):
        """Test export JSON handler."""
        data = {"test": "data", "items": [1, 2, 3]}
        result = await export_json_handler(data, True)
        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_export_json_handler_error(self):
        """Test export JSON handler with invalid data."""
        # The implementation handles None gracefully and returns an error dict
        with pytest.raises(ToolError):
            await export_json_handler(None, True)

    @pytest.mark.asyncio
    async def test_export_csv_handler_success(self):
        """Test export CSV handler."""
        data = {
            "sorted_lines": [
                "2024-01-01 10:00:00 INFO Test message",
                "2024-01-02 11:00:00 ERROR Error message",
            ]
        }
        result = await export_csv_handler(data, True)
        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_export_csv_handler_error(self):
        """Test export CSV handler with error."""
        with pytest.raises(ToolError):
            await export_csv_handler(None, True)

    @pytest.mark.asyncio
    async def test_export_text_handler_success(self):
        """Test export text handler."""
        data = {
            "sorted_lines": [
                "2024-01-01 10:00:00 INFO Test message",
                "2024-01-02 11:00:00 ERROR Error message",
            ]
        }
        result = await export_text_handler(data, True)
        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_export_text_handler_error(self):
        """Test export text handler with error."""
        with pytest.raises(ToolError):
            await export_text_handler(None, True)

    @pytest.mark.asyncio
    async def test_summary_report_handler_success(self):
        """Test summary report handler."""
        data = {
            "total_lines": 100,
            "filtered_lines": ["test line"],
            "statistics": {"level_distribution": {"ERROR": 10}},
        }
        result = await summary_report_handler(data)
        assert "error" not in result or result.get("error") is None

    @pytest.mark.asyncio
    async def test_summary_report_handler_error(self):
        """Test summary report handler with error."""
        with pytest.raises(ToolError):
            await summary_report_handler(None)


# ---- Regression tests for the release acceptance findings ----

LOG = """2026-01-01 10:20:00 ERROR Disk FULL on node7
2026-01-01 10:00:00 [ERROR] disk full on node3
2026-01-01 10:01:00 INFO request served
2026-01-01 10:02:00 [ERROR] timeout
"""


@pytest.fixture
def log(tmp_path):
    path = tmp_path / "app.log"
    path.write_text(LOG)
    return str(path)


@pytest.mark.asyncio
async def test_output_file_gets_everything_and_reply_is_a_bounded_preview(tmp_path):
    source = tmp_path / "big.log"
    lines = [
        f"2026-01-01 {m // 60:02d}:{m % 60:02d}:00 INFO event {m}" for m in range(250)
    ]
    source.write_text("\n".join(reversed(lines)) + "\n")
    for handler, key, args in (
        (parallel_sort_handler, "sorted_lines", ()),
        (sort_log_handler, "sorted_lines", ()),
        (filter_level_handler, "filtered_lines", (["INFO"],)),
    ):
        out = tmp_path / f"{handler.__name__}.log"
        if handler is filter_level_handler:
            result = await handler(str(source), *args, str(out))
        else:
            result = await handler(str(source), str(out))
        assert (
            out.read_text().splitlines()
            == sorted(lines)[:: 1 if key == "sorted_lines" else -1]
        )
        assert result[key] == out.read_text().splitlines()[:100]
        assert result["truncated"] is True
        assert (result["lines_returned"], result["lines_written"]) == (100, 250)
        assert str(out) in result["truncation_note"]
    # No output file: the reply is the only copy, so it stays complete.
    result = await sort_log_handler(str(source))
    assert result["sorted_lines"] == lines and "truncated" not in result


@pytest.mark.asyncio
async def test_filter_by_keyword_honours_case_sensitive(log):
    insensitive = await filter_keyword_handler(log, ["FULL"])
    assert insensitive["matched_lines"] == 2
    sensitive = await filter_keyword_handler(log, ["FULL"], True)
    assert sensitive["filtered_lines"] == [
        "2026-01-01 10:20:00 ERROR Disk FULL on node7"
    ]


@pytest.mark.parametrize(
    "filters, operator, message",
    [
        (
            [{"field": "level", "operator": "bogus_op", "value": "ERROR"}],
            None,
            "operator",
        ),
        ([{"field": "nope", "operator": "equals", "value": "ERROR"}], None, "field"),
        (
            [{"field": "message", "operator": "regex", "value": "(unclosed"}],
            None,
            "regex",
        ),
        (["garbage"], None, "object"),
        ([], "XOR", "logical_operator"),
    ],
)
@pytest.mark.asyncio
async def test_invalid_filters_are_tool_errors(log, filters, operator, message):
    with pytest.raises(ToolError, match=message):
        await filter_logs_handler(log, filters, operator)


@pytest.mark.asyncio
async def test_bad_arguments_are_tool_errors_not_error_payloads(log):
    with pytest.raises(ToolError, match="Unknown preset"):
        await filter_preset_handler(log, "bogus")
    with pytest.raises(ToolError, match="Invalid time format"):
        await filter_time_range_handler(log, "yesterday", "2026-01-01 10:05:00")
    with pytest.raises(ToolError, match="logical_operator"):
        await filter_keyword_handler(log, ["disk"], False, "XOR")
    with pytest.raises(ToolError, match="No sorted_lines"):
        await export_csv_handler({"foo": 1})


@pytest.mark.asyncio
async def test_exports_normalise_bracketed_levels_like_filtering(log):
    data = await sort_log_handler(log)
    report = await summary_report_handler(data)
    assert report["structured_data"]["log_level_distribution"] == {
        "ERROR": 3,
        "INFO": 1,
    }
    csv_rows = (await export_csv_handler(data))["content"].splitlines()
    assert csv_rows[1].startswith("1,2026-01-01 10:00:00,ERROR,disk full on node3,")


@pytest.mark.asyncio
async def test_patterns_sort_before_clustering_and_honour_pattern_types(log):
    result = await detect_patterns_handler(log, ["error_clusters"])
    assert list(result["patterns"]) == ["error_clusters"]
    (cluster,) = result["patterns"]["error_clusters"]["clusters"]
    assert cluster["start_time"] == "2026-01-01T10:00:00"
    assert cluster["duration_seconds"] == 120.0 and cluster["error_count"] == 2
    assert len((await detect_patterns_handler(log))["patterns"]) == 6
    with pytest.raises(ToolError, match="bogus"):
        await detect_patterns_handler(log, ["bogus"])
    with pytest.raises(ToolError, match="sensitivity"):
        await detect_patterns_handler(log, None, "bogus")
