"""Level-based operations must agree on standard log representations."""

import asyncio

import pytest

from parallel_sort_mcp.implementation import (
    filter_handler,
    statistics_handler,
    pattern_detection,
)


@pytest.mark.parametrize(
    "parser",
    [
        filter_handler.parse_log_entry,
        statistics_handler.parse_log_entry,
        pattern_detection.parse_log_entry,
    ],
)
@pytest.mark.parametrize(
    "line",
    [
        "2026-09-10 10:00:00 ERROR failed request",
        "2026-09-10 10:00:00 [ERROR] failed request",
        "2026-09-10 10:00:00\t[error]\tfailed request",
        "[2026-09-10 10:00:00] [ERROR] failed request",
    ],
)
def test_level_and_message_are_consistent(parser, line):
    result = parser(line)
    assert result["level"] == "ERROR"
    assert result["message"] == "failed request"
    assert result["original_line"] == line


def test_bracketed_level_filter_selects_actual_error_lines(tmp_path):
    source = tmp_path / "application.log"
    lines = [
        "2026-09-10 10:00:00 [INFO] Started",
        "2026-09-10 10:00:01 [ERROR] Failed",
        "2026-09-10 10:00:02 ERROR Failed again",
    ]
    source.write_text("\n".join(lines) + "\n")
    result = asyncio.run(filter_handler.filter_by_log_level(str(source), ["ERROR"]))
    assert result["filtered_lines"] == lines[1:]
    assert result["matched_lines"] == 2
