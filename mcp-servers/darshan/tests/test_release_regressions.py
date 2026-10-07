"""Regressions for the 2026-10 pre-release acceptance findings."""

from unittest.mock import patch

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from darshan_mcp.capabilities import darshan_parser
from darshan_mcp.capabilities.native_text import parse_native_text
from darshan_mcp.server import mcp

PARSE = "darshan_mcp.capabilities.darshan_parser._parse_darshan_json"
FILES = {
    "a": {"bytes_read": 100 << 20, "read_ops": 100, "read_time": 1.0},
    "b": {"bytes_written": 100 << 20, "write_ops": 100, "write_time": 4.0},
}
NATIVE = "\n".join(
    [
        "# darshan log version: 3.41",
        "# start_time: 1790951189",
        "# end_time: 1790951199",
        "# run time: 10.0",
        "POSIX\t0\t1\tPOSIX_F_WRITE_START_TIMESTAMP\t8.5\t/f\t/\text4",
        "POSIX\t0\t1\tPOSIX_F_WRITE_END_TIMESTAMP\t9.5\t/f\t/\text4",
        "POSIX\t0\t1\tPOSIX_F_READ_START_TIMESTAMP\t0.000000\t/f\t/\text4",
    ]
)


@pytest.mark.asyncio
async def test_total_bandwidth_has_one_definition_between_read_and_write():
    """200 MiB over 1 s reading + 4 s writing is 40 MiB/s in both tools."""
    with patch(PARSE, return_value={"success": True, "job": {}, "files": FILES}):
        metrics = await darshan_parser.get_io_performance_metrics("x")
        summary = await darshan_parser.get_job_summary("x")
    total = metrics["overall_metrics"]["total_bandwidth_mbps"]
    assert total == summary["total_bandwidth_mbps"] == 40.0
    read = metrics["read_metrics"]["bandwidth_mbps"]
    write = metrics["write_metrics"]["bandwidth_mbps"]
    assert write <= total <= read
    assert "MiB/s" in metrics["units"]["bandwidth_mbps"]


@pytest.mark.asyncio
async def test_timeline_is_truthful_and_validates_resolution():
    with patch(PARSE, return_value=parse_native_text(NATIVE)):
        result = await darshan_parser.get_timeline_analysis("x", "100ms")
        bad = await darshan_parser.get_timeline_analysis("x", "banana")
    assert result["timeline_available"] is False
    assert "DXT" in result["message"]
    assert result["analysis"]["total_duration"] == 10.0
    assert result["analysis"]["module_activity_windows"] == {
        "POSIX": {"write": {"first": 8.5, "last": 9.5}}
    }
    assert bad["success"] is False and "time_resolution" in bad["error"]


@pytest.mark.asyncio
async def test_compare_implements_file_count_and_rejects_unknown_metrics():
    one = {"success": True, "job": {}, "files": {"a": FILES["a"]}}
    two = {"success": True, "job": {}, "files": FILES}
    with patch(PARSE, side_effect=[one, two]):
        result = await darshan_parser.compare_darshan_logs("1", "2", ["file_count"])
    assert result["differences"]["file_count"] == {
        "log_1": 1,
        "log_2": 2,
        "difference": 1,
        "percent_change": 100.0,
    }
    assert result["summary"] == {"file_count": "higher in log_2"}
    bogus = await darshan_parser.compare_darshan_logs("1", "2", ["bogus_metric"])
    assert bogus["success"] is False and "bogus_metric" in bogus["error"]


@pytest.mark.asyncio
async def test_report_says_visualizations_are_not_available():
    with patch(PARSE, return_value={"success": True, "job": {}, "files": FILES}):
        plain = await darshan_parser.generate_io_summary_report("x")
        asked = await darshan_parser.generate_io_summary_report("x", True)
    assert "visualizations" not in plain
    assert asked["visualizations"]["available"] is False


@pytest.mark.asyncio
async def test_missing_darshan_parser_is_named_by_every_tool(tmp_path):
    """No tool may hide the missing prerequisite behind a generic failure."""
    log = tmp_path / "job.darshan"
    log.write_bytes(b"x")
    calls = {
        "analyze_posix_operations": {"log_file_path": str(log)},
        "analyze_mpiio_operations": {"log_file_path": str(log)},
        "identify_io_bottlenecks": {"log_file_path": str(log)},
        "compare_darshan_logs": {"log_file_1": str(log), "log_file_2": str(log)},
        "generate_io_summary_report": {"log_file_path": str(log)},
    }
    with patch(
        "darshan_mcp.capabilities.darshan_parser.asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError(),
    ):
        async with Client(mcp) as client:
            for tool, args in calls.items():
                with pytest.raises(ToolError, match="darshan-parser command not found"):
                    await client.call_tool(tool, args)


@pytest.mark.asyncio
async def test_server_info_version_and_glob_description():
    async with Client(mcp) as client:
        tools = {tool.name: tool for tool in await client.list_tools()}
        assert client.server_info.version == "2.2.5"
    schema = tools["analyze_file_access_patterns"].inputSchema
    assert "glob" in schema["properties"]["file_pattern"]["description"]
