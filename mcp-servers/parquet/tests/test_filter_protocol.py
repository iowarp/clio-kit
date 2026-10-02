"""Invalid filters must fail through MCP instead of returning unfiltered data."""

import json

from fastmcp import Client
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from parquet_mcp.server import mcp


@pytest.fixture
def source(tmp_path):
    path = tmp_path / "data.parquet"
    pq.write_table(pa.table({"x": [1, 2, 3], "label": ["a", "b", "c"]}), path)
    return path


@pytest.mark.asyncio
@pytest.mark.parametrize("tool", ["read_slice_tool", "aggregate_column_tool"])
@pytest.mark.parametrize(
    "spec",
    [
        {},
        None,
        [],
        False,
        "text",
        {"column": "x", "op": "typo", "value": 2},
        {"column": "missing", "op": "equal", "value": 2},
        {"column": "x", "op": "equal", "value": "bad"},
        {"column": "x", "op": "equal"},
        {"column": "x", "op": "in", "values": "123"},
        {"column": "x", "op": "equal", "value": 2, "extra": 3},
        {"and": []},
        {"or": {}},
        {"not": None},
        {"and": [{"column": "x", "op": "equal", "value": 2}, {}]},
        {"not": {"column": "missing", "op": "equal", "value": 2}},
        {"and": [{"column": "x", "op": "equal", "value": 2}], "or": []},
    ],
)
async def test_invalid_filter_is_an_mcp_error(source, tool, spec, capsys):
    args = {"file_path": str(source), "filter_json": json.dumps(spec)}
    if tool == "read_slice_tool":
        args.update(start_row=0, end_row=3)
    else:
        args.update(column_name="x", operation="sum")
    async with Client(mcp) as client:
        result = await client.call_tool(tool, args, raise_on_error=False)
    assert result.is_error
    assert "Invalid filter" in str(result.content)
    assert capsys.readouterr().out == ""


@pytest.mark.asyncio
async def test_valid_nested_filter_and_unfiltered_calls_return_exact_values(source):
    spec = {
        "and": [
            {"column": "x", "op": "greater", "value": 1},
            {"not": {"column": "label", "op": "is_in", "values": ["c"]}},
        ]
    }
    async with Client(mcp) as client:
        filtered = await client.call_tool(
            "read_slice_tool",
            {
                "file_path": str(source),
                "start_row": 0,
                "end_row": 3,
                "filter_json": json.dumps(spec),
            },
        )
        total = await client.call_tool(
            "aggregate_column_tool",
            {
                "file_path": str(source),
                "column_name": "x",
                "operation": "sum",
                "filter_json": json.dumps(spec),
            },
        )
        unfiltered = await client.call_tool(
            "read_slice_tool",
            {
                "file_path": str(source),
                "start_row": 0,
                "end_row": 3,
            },
        )
    assert json.loads(filtered.content[0].text)["data"] == [{"x": 2, "label": "b"}]
    assert json.loads(total.content[0].text)["result"] == 2
    assert [row["x"] for row in json.loads(unfiltered.content[0].text)["data"]] == [
        1,
        2,
        3,
    ]


@pytest.mark.asyncio
async def test_filter_can_use_a_column_excluded_from_the_projection(source):
    async with Client(mcp) as client:
        result = await client.call_tool(
            "read_slice_tool",
            {
                "file_path": str(source),
                "start_row": 0,
                "end_row": 3,
                "columns": ["label"],
                "filter_json": json.dumps({"column": "x", "op": "equal", "value": 2}),
            },
        )
    assert json.loads(result.content[0].text)["data"] == [{"label": "b"}]
