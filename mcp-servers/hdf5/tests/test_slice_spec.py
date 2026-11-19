"""hdf5_batch_read accepts the documented slice syntax and nothing else."""

import asyncio

import h5py
import numpy as np
import pytest
from fastmcp import Client

from hdf5_mcp.server import mcp
from hdf5_mcp.slice_spec import parse_slice_spec


@pytest.mark.parametrize(
    "spec, expected",
    [
        ("0:10", (slice(0, 10),)),
        ("0:10, :, 5", (slice(0, 10), slice(None), 5)),
        (" -3 ", (-3,)),
        ("::2", (slice(None, None, 2),)),
        ("1:-1:3", (slice(1, -1, 3),)),
        ("..., 0", (Ellipsis, 0)),
        ("[0:2, 1]", (slice(0, 2), 1)),
    ],
)
def test_parses_selection_syntax(spec, expected):
    assert parse_slice_spec(spec) == expected


@pytest.mark.parametrize(
    "spec",
    [
        "abs(-1)",
        "np.s_[0]",
        "0:3, len('ab')",
        "1.5",
        "0:1:2:3",
        "::0",
        "..., ...",
        "",
        " , ",
        "0x10",
    ],
)
def test_refuses_anything_else(spec):
    with pytest.raises(ValueError):
        parse_slice_spec(spec)


def test_batch_read_refuses_expressions_and_still_reads_slices(tmp_path):
    path = tmp_path / "data.h5"
    with h5py.File(path, "w") as file:
        file.create_dataset("x", data=np.arange(20).reshape(4, 5))

    async def exercise():
        async with Client(mcp) as client:
            await client.call_tool("open_file", {"path": str(path)})
            try:
                refused = await client.call_tool(
                    "hdf5_batch_read",
                    {"paths": "x", "slice_spec": "abs(-1)"},
                    raise_on_error=False,
                )
                read = await client.call_tool(
                    "hdf5_batch_read",
                    {"paths": "x", "slice_spec": "1:3, ::2"},
                )
            finally:
                await client.call_tool("close_file", {})
            return refused, read

    refused, read = asyncio.run(exercise())
    assert refused.is_error
    text = "\n".join(item.text for item in read.content if item.type == "text")
    # The summary reports the dataset shape; the preview shows the selection.
    assert "Preview: [[5, 7, 9], [10, 12, 14]]" in text
