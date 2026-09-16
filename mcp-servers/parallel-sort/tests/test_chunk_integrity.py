"""Chunking and the MCP sort must preserve every record, including duplicates."""

from collections import Counter
from pathlib import Path

from fastmcp import Client
import pytest

from parallel_sort_mcp.implementation.parallel_processor import (
    cleanup_temp_files,
    split_file_into_chunks,
)
from parallel_sort_mcp.server import mcp


@pytest.mark.asyncio
@pytest.mark.parametrize("chunk_size", [1, 8, 16, 70, 4096])
@pytest.mark.parametrize(
    "payload",
    [
        b"",
        b"one\ntwo\nthree\n",
        b"one\r\ntwo\r\nlast",
        "duplicate\nduplicate\n\nlong-line-é漢字-without-newline".encode(),
    ],
)
async def test_chunks_reconstruct_input_exactly(tmp_path, chunk_size, payload):
    source = tmp_path / "input.log"
    source.write_bytes(payload)
    chunks = await split_file_into_chunks(str(source), chunk_size)
    try:
        assert b"".join(Path(path).read_bytes() for path in chunks) == payload
    finally:
        await cleanup_temp_files(chunks)


@pytest.mark.asyncio
async def test_mcp_parallel_sort_preserves_all_records_across_real_chunks(tmp_path):
    # Cross the actual 1 MiB tool parameter boundary, not the small-file fallback.
    lines = [f"2026-01-01 00:00:{i % 60:02d} INFO record={i:05d}" for i in range(40000)]
    lines += [lines[0], lines[0]]
    source = tmp_path / "input.log"
    source.write_text("\n".join(reversed(lines)) + "\n")
    output = tmp_path / "sorted.log"
    assert source.stat().st_size > 1024 * 1024
    async with Client(mcp) as client:
        result = await client.call_tool(
            "parallel_sort_large_file",
            {
                "log_file": str(source),
                "output_file": str(output),
                "chunk_size_mb": 1,
                "num_workers": 2,
            },
        )
    assert not result.is_error
    saved = output.read_text().splitlines()
    assert Counter(saved) == Counter(lines)
    timestamps = [line[:19] for line in saved]
    assert timestamps == sorted(timestamps)
