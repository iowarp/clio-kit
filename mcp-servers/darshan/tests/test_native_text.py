"""Regressions based on real Darshan 3.5 --base output."""

import pytest
from darshan_mcp.capabilities.native_text import parse_native_text
from darshan_mcp.capabilities import darshan_parser

TRACE = """# darshan log version: 3.41
# uid: 1000
# jobid: 360
# start_time: 1789096172
# end_time: 1789096172
# run time: 0.0033
# nprocs: 2
# POSIX module: 125 bytes (compressed), ver=4
# MPI-IO module: 120 bytes (compressed), ver=3
POSIX\t0\t123\tPOSIX_READS\t4\t/data/a.bin\t/\text4
POSIX\t1\t123\tPOSIX_READS\t4\t/data/a.bin\t/\text4
POSIX\t0\t123\tPOSIX_BYTES_READ\t16384\t/data/a.bin\t/\text4
POSIX\t1\t123\tPOSIX_BYTES_READ\t16384\t/data/a.bin\t/\text4
MPI-IO\t-1\t123\tMPIIO_BYTES_READ\t32768\t/data/a.bin\t/\text4
"""


def test_native_counters_preserve_integers_and_do_not_double_count_layers():
    parsed = parse_native_text(TRACE)
    assert parsed["job"]["runtime"] == 0.0033
    assert parsed["job"]["job_id"] == 360
    file = parsed["files"]["/data/a.bin"]
    assert file["bytes_read"] == 32768
    assert file["read_ops"] == 8
    assert isinstance(file["read_ops"], int)
    assert file["counter_layer"] == "POSIX"


@pytest.mark.asyncio
async def test_real_parser_format_reaches_tool_summary(monkeypatch):
    async def parser(args, path):
        if args == ["--json"]:
            return "", "unsupported option", 1
        assert args == ["--base"]
        return TRACE, "", 0

    monkeypatch.setattr(darshan_parser, "_run_darshan_command", parser)
    result = await darshan_parser.get_job_summary("trace.darshan")
    assert result["runtime_seconds"] == 0.0033
    assert result["total_bytes_read"] == 32768
    assert result["total_read_operations"] == 8
    result = await darshan_parser.analyze_posix_operations("trace.darshan")
    assert result["operations"]["reads"] == 8
    assert result["operations"]["closes"] is None
    patterns = await darshan_parser.analyze_file_access_patterns("trace.darshan")
    assert patterns["files_analysis"][0]["file_size"] is None
    assert patterns["file_sizes"] == []
    assert "file_size_stats" not in patterns

    result = await darshan_parser.get_timeline_analysis("trace.darshan")
    assert result["analysis"]["total_duration"] == 0.0033


@pytest.mark.asyncio
async def test_billion_operation_metrics_stay_bounded_and_label_averages(monkeypatch):
    async def parsed(path):
        return {"files": {"large": {"bytes_read": 4096 * 10**9, "read_ops": 10**9}}}

    monkeypatch.setattr(darshan_parser, "_parse_darshan_json", parsed)
    result = await darshan_parser.get_io_performance_metrics("large.darshan")
    assert result["success"]
    stats = result["read_metrics"]["request_size_stats"]
    assert stats["avg"] == 4096
    assert stats["std"] == 0
    assert "not individual request sizes" in stats["basis"]
