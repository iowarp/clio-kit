"""Regression tests for the pre-release acceptance findings, on a real BP5 file."""

import adios2
import numpy as np
import pytest
from fastmcp import Client

from adios_mcp.server import mcp


@pytest.fixture
def bp_file(tmp_path):
    path = str(tmp_path / "s.bp")
    with adios2.Stream(path, "w") as writer:
        writer.write(
            "temperature",
            np.arange(12, dtype="f8").reshape(3, 4),
            shape=[3, 4],
            start=[0, 0],
            count=[3, 4],
        )
    return path


async def call(tool, arguments):
    async with Client(mcp) as client:
        assert client.server_info.version == "2.2.5"
        return await client.call_tool(tool, arguments, raise_on_error=False)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "tool, arguments, message",
    [
        ("inspect_attributes", {"variable_name": "nope"}, "Variable 'nope' not found"),
        ("inspect_variables", {"variable_name": "nope"}, "Variable 'nope' not found"),
        (
            "inspect_variables_at_step",
            {"variable_name": "temperature", "step": 9},
            "Step 9 not found",
        ),
    ],
)
async def test_unknown_names_are_mcp_errors_not_crashes(
    bp_file, tool, arguments, message
):
    result = await call(tool, {"filename": bp_file, **arguments})
    assert result.is_error, result
    assert message in result.content[0].text


@pytest.mark.asyncio
async def test_missing_file_error_has_no_ansi_escapes(tmp_path):
    result = await call("inspect_variables", {"filename": str(tmp_path / "nope.bp")})
    assert result.is_error, result
    assert "\x1b" not in result.content[0].text


@pytest.mark.asyncio
async def test_read_and_attributes_happy_path(bp_file):
    read = await call(
        "read_variable_at_step",
        {"filename": bp_file, "variable_name": "temperature", "target_step": 0},
    )
    assert read.structured_content == {
        "value": [float(i) for i in range(12)],
        "shape": [3, 4],
    }
    attributes = await call(
        "inspect_attributes", {"filename": bp_file, "variable_name": "temperature"}
    )
    assert not attributes.is_error and attributes.structured_content == {}
