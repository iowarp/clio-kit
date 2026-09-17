"""Operational errors must be MCP failures, with a usable session afterward."""

import asyncio

import h5py
import pytest
from fastmcp import Client

from hdf5_mcp.server import mcp


@pytest.mark.parametrize("mode", ["2026-07-28", "legacy"])
def test_failures_are_flagged_and_session_recovers(tmp_path, mode):
    source = tmp_path / "input.h5"
    with h5py.File(source, "w") as file:
        file.create_dataset("values", data=[2, 4, 8])
    original = source.read_bytes()

    async def exercise():
        async with Client(mcp, mode=mode) as client:
            await client.call_tool("close_file", {}, raise_on_error=False)

            async def failure(tool, arguments, message):
                result = await client.call_tool(tool, arguments, raise_on_error=False)
                assert result.is_error, result
                assert message in " ".join(
                    block.text for block in result.content if block.type == "text"
                )

            await failure("get_shape", {"path": "/values"}, "No file currently open")
            await failure(
                "open_file", {"path": str(tmp_path / "absent.h5")}, "absent.h5"
            )
            for tool, arguments in (
                ("hdf5_stream_data", {"path": "/values"}),
                ("analyze_dataset_structure", {}),
                ("identify_io_bottlenecks", {}),
            ):
                await failure(tool, arguments, "No file currently open")
            await client.call_tool("open_file", {"path": str(source)})
            try:
                await failure("get_shape", {"path": "/absent"}, "Dataset not found")
                await failure(
                    "hdf5_batch_read",
                    {"paths": "values", "slice_spec": "["},
                    "Invalid slice specification",
                )
                await failure(
                    "export_dataset",
                    {
                        "path": "/values",
                        "output_path": str(tmp_path / "missing-directory" / "out.json"),
                        "export_format": "json",
                    },
                    "Error writing",
                )
                result = await client.call_tool("get_shape", {"path": "/values"})
                assert not result.is_error
                assert "3" in str(result)
            finally:
                await client.call_tool("close_file", {})

    asyncio.run(exercise())
    assert source.read_bytes() == original
