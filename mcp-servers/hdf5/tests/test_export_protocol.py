"""Export format selection must work without a server-to-client back-channel."""

import asyncio
import json

import h5py
import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from hdf5_mcp.server import mcp


@pytest.mark.parametrize("mode", ["2026-07-28", "legacy"])
def test_explicit_csv_export_in_both_protocol_eras(tmp_path, mode):
    source = tmp_path / "data.h5"
    target = tmp_path / "data.csv"
    with h5py.File(source, "w") as file:
        file.create_dataset("values", data=[[1, 2], [3, 4]])

    async def exercise():
        async with Client(mcp, mode=mode) as client:
            await client.call_tool("open_file", {"path": str(source)})
            try:
                await client.call_tool(
                    "export_dataset",
                    {"path": "values", "output_path": str(target), "export_format": "csv"},
                )
            finally:
                await client.call_tool("close_file", {})

    asyncio.run(exercise())
    assert target.read_text().splitlines() == ["1,2", "3,4"]


def test_modern_export_defaults_to_json_and_rejects_unknown_format(tmp_path):
    source = tmp_path / "data.h5"
    target = tmp_path / "data.json"
    with h5py.File(source, "w") as file:
        file.create_dataset("values", data=[2, 4, 6])

    async def exercise():
        async with Client(mcp, mode="2026-07-28") as client:
            await client.call_tool("open_file", {"path": str(source)})
            try:
                with pytest.raises(ToolError):
                    await client.call_tool(
                        "export_dataset",
                        {"path": "values", "output_path": str(target), "export_format": "invalid"},
                    )
                assert not target.exists()
                await client.call_tool(
                    "export_dataset", {"path": "values", "output_path": str(target)}
                )
            finally:
                await client.call_tool("close_file", {})

    asyncio.run(exercise())
    assert json.loads(target.read_text())["data"] == [2, 4, 6]
