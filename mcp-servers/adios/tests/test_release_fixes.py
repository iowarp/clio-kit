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
        assert client.server_info.version == "2.3.2"
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


@pytest.fixture
def series_file(tmp_path):
    """Five steps of a 4x5x3 field with a spike at (2, 3, 1) from step 3, plus a scalar."""
    path = str(tmp_path / "series.bp")
    with adios2.Stream(path, "w") as writer:
        for _ in writer.steps(5):
            step = writer.current_step()
            field = np.full((4, 5, 3), 300.0)
            if step >= 3:
                field[2, 3, 1] = 360.0 + step
            writer.write("T", field, field.shape, [0, 0, 0], field.shape)
            writer.write("time", np.float64(step * 0.5))
            if step == 0:
                writer.write_attribute("solver", "ftcs")
    return path


@pytest.mark.asyncio
async def test_variable_stats_finds_onset_and_location_in_one_call(series_file):
    result = await call(
        "variable_stats", {"filename": series_file, "variable_name": "T"}
    )
    assert not result.is_error, result
    stats = result.structured_content
    assert [row["max"] for row in stats["steps"]] == [300.0, 300.0, 300.0, 363.0, 364.0]
    assert stats["steps"][3]["argmax"] == [2, 3, 1]
    assert stats["overall"] == {
        "max": 364.0,
        "max_step": 4,
        "min": 300.0,
        "min_step": 0,
        "max_index": [2, 3, 1],
        "min_index": [0, 0, 0],
    }


@pytest.mark.asyncio
async def test_variable_stats_slice_steps_and_scalars(series_file):
    sliced = await call(
        "variable_stats",
        {
            "filename": series_file,
            "variable_name": "T",
            "step_start": 3,
            "step_count": 1,
            "start": "[2, 3, 0]",
            "count": [1, 1, 3],
        },
    )
    assert sliced.structured_content["steps"][0]["argmax"] == [2, 3, 1]
    assert sliced.structured_content["selection"] == {
        "start": [2, 3, 0],
        "count": [1, 1, 3],
    }
    scalar = await call(
        "variable_stats", {"filename": series_file, "variable_name": "time"}
    )
    assert [row["value"] for row in scalar.structured_content["steps"]] == [
        0.0,
        0.5,
        1.0,
        1.5,
        2.0,
    ]
    bad = await call(
        "variable_stats",
        {"filename": series_file, "variable_name": "T", "step_start": 9},
    )
    assert bad.is_error and "outside 0..4" in bad.content[0].text


@pytest.mark.asyncio
async def test_read_refuses_oversized_reads_and_reads_slices(series_file):
    refused = await call(
        "read_variable_at_step",
        {
            "filename": series_file,
            "variable_name": "T",
            "target_step": 3,
            "max_elements": 10,
        },
    )
    assert refused.is_error
    assert "selects 60 elements" in refused.content[0].text
    assert "variable_stats" in refused.content[0].text

    window = await call(
        "read_variable_at_step",
        {
            "filename": series_file,
            "variable_name": "T",
            "target_step": 3,
            "start": [2, 3, 0],
            "count": [1, 1, 3],
        },
    )
    assert window.structured_content == {
        "value": [300.0, 363.0, 300.0],
        "shape": [1, 1, 3],
        "start": [2, 3, 0],
    }
    outside = await call(
        "read_variable_at_step",
        {
            "filename": series_file,
            "variable_name": "T",
            "target_step": 0,
            "start": [3, 0, 0],
            "count": [2, 1, 1],
        },
    )
    assert outside.is_error and "axis 0" in outside.content[0].text


@pytest.mark.asyncio
async def test_list_defaults_to_working_directory_and_strings_unwrap(
    series_file, tmp_path, monkeypatch
):
    monkeypatch.chdir(tmp_path)
    listed = await call("list_bp5", {})
    assert listed.structured_content == {"files": ["series.bp"]}
    attributes = await call("inspect_attributes", {"filename": series_file})
    assert attributes.structured_content["solver"]["value"] == "ftcs"
