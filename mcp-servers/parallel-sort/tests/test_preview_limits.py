"""Inline previews are bounded independently of optional file output."""

from parallel_sort_mcp.mcp_handlers import _finish, PREVIEW_BYTES, PREVIEW_LINES


def test_inline_preview_and_complete_saved_result(tmp_path):
    lines = [f"event {i}" for i in range(5000)]
    for output in (None, str(tmp_path / "all.log")):
        result = _finish({"filtered_lines": lines}, "filtered_lines", output)
        assert result["filtered_lines"] == lines[:PREVIEW_LINES]
        assert result["truncated"] and result["total_result_lines"] == 5000
        if output:
            assert (tmp_path / "all.log").read_text().splitlines() == lines
            assert result["lines_written"] == 5000
        else:
            assert "not been saved" in result["truncation_note"]
            assert "lines_written" not in result


def test_one_large_unicode_line_is_bounded():
    result = _finish({"sorted_lines": ["🙂" * PREVIEW_BYTES]}, "sorted_lines")
    assert result["truncated"]
    assert len(result["sorted_lines"][0].encode("utf-8")) <= PREVIEW_BYTES
