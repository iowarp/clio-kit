"""Regressions from the 2026-10-02 pre-release acceptance test, driven over MCP."""

import asyncio

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib
from pathlib import Path

import pandas as pd
import pytest
from fastmcp import Client

from pandas_mcp.server import mcp

WRITERS = {
    "clean_data": "_cleaned.csv",
    "groupby_operations": "_grouped.csv",
    "merge_datasets": "_merged.csv",
    "pivot_table": "_pivot.csv",
    "time_series_operations": "_<operation>.csv",
    "filter_data": "_filtered.csv",
    "optimize_memory": "_optimized.csv",
}


def call(tool, arguments):
    async def exchange():
        async with Client(mcp) as client:
            return await client.call_tool(tool, arguments, raise_on_error=False)

    return asyncio.run(exchange())


@pytest.fixture
def csv(tmp_path):
    path = tmp_path / "w.csv"
    pd.DataFrame(
        {
            "day": pd.date_range("2026-01-01", periods=8).astype(str),
            "g": list("AABBAACC"),
            "c": list("uuvvuuvu"),
            "x": [10.0, 11.0, 9.5, 10.5, 12.0, 8.0, 10.0, 100.0],
            "y": [12.5, 13.0, 12.0, 14.5, 15.0, 11.0, 13.5, 14.0],
        }
    ).to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize(
    "tool,arguments",
    [
        ("load_data", {"file_path": "/nope.csv"}),
        ("save_data", {"data": {"k": [1, 2], "v": [1]}, "file_path": "/tmp/x.csv"}),
        ("statistical_summary", {"file_path": "/nope.csv"}),
        ("correlation_analysis", {"file_path": "/nope.csv"}),
        ("clean_data", {"file_path": "/nope.csv"}),
        (
            "groupby_operations",
            {"file_path": "/nope.csv", "group_by": ["g"], "operations": {"y": "sum"}},
        ),
        ("merge_datasets", {"left_file": "/nope.csv", "right_file": "/nope.csv"}),
        ("pivot_table", {"file_path": "/nope.csv", "index": ["g"]}),
        (
            "time_series_operations",
            {"file_path": "/nope.csv", "date_column": "d", "operation": "diff"},
        ),
        ("validate_data", {"file_path": "/nope.csv", "validation_rules": {}}),
        ("optimize_memory", {"file_path": "/nope.csv"}),
        ("profile_data", {"file_path": "/nope.csv"}),
    ],
)
def test_failures_are_tool_errors_with_the_real_message(tool, arguments):
    result = call(tool, arguments)
    assert result.is_error
    text = result.content[0].text
    assert "not found" in text or "same length" in text, text


def test_distribution_summary_is_structured(csv):
    result = call(
        "statistical_summary",
        {"file_path": csv, "columns": ["x"], "include_distributions": True},
    )
    normality = result.structured_content["additional_statistics"]["x"]
    assert normality["normality_test"]["is_normal"] is False


def test_groupby_rejects_unknown_operation(csv):
    result = call(
        "groupby_operations",
        {"file_path": csv, "group_by": ["g"], "operations": {"y": "bogus"}},
    )
    assert result.is_error
    assert "bogus" in result.content[0].text and "median" in result.content[0].text


@pytest.mark.parametrize("operation", ["rolling_mean", "trend", "seasonality"])
def test_time_series_only_advertises_what_it_implements(csv, operation):
    async def schema():
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            return tools["time_series_operations"].input_schema

    advertised = asyncio.run(schema())["properties"]["operation"]["description"]
    assert operation not in advertised
    result = call(
        "time_series_operations",
        {"file_path": csv, "date_column": "day", "operation": operation},
    )
    assert result.is_error and "Valid operations" in result.content[0].text


def test_pivot_defaults_to_numeric_values_and_keeps_empty_cells_missing(csv):
    result = call("pivot_table", {"file_path": csv, "index": ["g"], "columns": ["c"]})
    assert not result.is_error
    rows = {row["g"]: row for row in result.structured_content["pivot_table"]}
    assert rows["A"]["y_u"] == pytest.approx(12.875)
    assert rows["A"]["y_v"] is None  # no (A, v) rows: missing, not 0
    assert rows["B"]["x_u"] is None


def test_tools_that_write_side_files_say_so():
    async def listing():
        async with Client(mcp) as client:
            return {tool.name: tool for tool in await client.list_tools()}

    tools = asyncio.run(listing())
    for name, suffix in WRITERS.items():
        assert tools[name].annotations.read_only_hint is False, name
        assert tools[name].annotations.destructive_hint is True, name
        assert suffix in tools[name].description, name
        if name == "groupby_operations":
            assert "unless overwrite=true" in tools[name].description
        else:
            assert "overwriting" in tools[name].description, name
    for name in ("save_data", "handle_missing_data"):
        assert tools[name].annotations.destructive_hint is True, name


def test_optimize_memory_column_shares_sum_to_100(csv):
    usage = call("optimize_memory", {"file_path": csv}).structured_content
    shares = [
        col["percentage_of_total"] for col in usage["column_memory_usage"].values()
    ]
    assert sum(shares) == pytest.approx(100, abs=0.1)


def test_profile_csv_rejects_unknown_columns(csv):
    result = call("profile_csv", {"data_path": csv, "columns": ["nope"]})
    assert result.is_error and "nope" in result.content[0].text


def test_validate_data_names_the_missing_column(csv):
    result = call(
        "validate_data",
        {"file_path": csv, "validation_rules": {"nope": {"min_value": 0}}},
    )
    assert result.is_error and "'nope' not found" in result.content[0].text


def test_hypothesis_tests_name_their_variant(csv):
    arguments = {"file_path": csv, "column1": "x", "column2": "y"}
    t_test = call("hypothesis_testing", arguments | {"test_type": "t_test"})
    assert "pooled" in t_test.structured_content["test_info"]["variance_assumption"]
    chi = call(
        "hypothesis_testing",
        {"file_path": csv, "test_type": "chi_square", "column1": "g", "column2": "c"},
    )
    info = chi.structured_content["test_info"]
    assert info["yates_correction"] is (info["degrees_of_freedom"] == 1)


def test_server_reports_the_packaged_version():
    async def version():
        async with Client(mcp) as client:
            return client.server_info.version

    descriptor = Path(__file__).parents[1] / "clio-server.toml"
    expected = tomllib.loads(descriptor.read_text())["version"]
    assert asyncio.run(version()) == expected


def test_side_file_never_overwrites_a_non_csv_input(tmp_path):
    content = "g,y\nA,1\nA,3\nB,5\n"
    arguments = {"group_by": ["g"], "operations": {"y": "mean"}}
    for name in ("t.txt", "t.csv"):
        source = tmp_path / name
        source.write_text(content)
        result = call("groupby_operations", arguments | {"file_path": str(source)})
        if name == "t.csv":
            assert result.is_error  # Both inputs map to the same side file.
            result = call(
                "groupby_operations",
                arguments | {"file_path": str(source), "overwrite": True},
            )
        assert result.structured_content["output_file"] == str(
            tmp_path / "t_grouped.csv"
        )
        assert source.read_text() == content
    assert (tmp_path / "t_grouped.csv").read_text() == "g,y\nA,2.0\nB,5.0\n"


@pytest.fixture
def orders(tmp_path):
    path = tmp_path / "orders.csv"
    pd.DataFrame(
        {
            "date": ["2026-01-05", "2026-01-20", "2026-02-03", "2026-02-03", "oops"],
            "region": ["east", "east", "east", "east", None],
            "revenue": [10.0, 30.0, 99999.0, 99999.0, 5.0],
        }
    ).to_csv(path, index=False)
    return str(path)


def test_aggregate_is_read_only_and_buckets_dedupes_and_filters(orders, tmp_path):
    before = sorted(p.name for p in tmp_path.iterdir())
    result = call(
        "aggregate",
        {
            "file_path": orders,
            "group_by": ["region", "date:month"],
            "operations": {"revenue": "sum,count"},
            "drop_duplicates": True,
        },
    )
    assert not result.is_error, result
    data = result.structured_content
    assert data["duplicates_dropped"] == 1
    assert data["results"] == [
        {
            "region": "east",
            "date_month": "2026-01",
            "rows": 2,
            "revenue_sum": 40.0,
            "revenue_count": 2,
        },
        {
            "region": "east",
            "date_month": "2026-02",
            "rows": 1,
            "revenue_sum": 99999.0,
            "revenue_count": 1,
        },
        {
            "region": None,
            "date_month": None,
            "rows": 1,
            "revenue_sum": 5.0,
            "revenue_count": 1,
        },
    ]
    assert data["notes"] == ["1 'date' values are not dates; grouped as null"]
    filtered = call(
        "aggregate",
        {
            "file_path": orders,
            "group_by": ["region"],
            "operations": {"revenue": "max"},
            "filter_condition": "revenue < 50000",
            "sort_by": "revenue_max",
            "descending": True,
        },
    ).structured_content
    assert filtered["rows_filtered_out"] == 2
    assert [row["revenue_max"] for row in filtered["results"]] == [30.0, 5.0]
    assert sorted(p.name for p in tmp_path.iterdir()) == before

    async def annotations():
        async with Client(mcp) as client:
            tools = {tool.name: tool for tool in await client.list_tools()}
            return tools["aggregate"].annotations

    assert asyncio.run(annotations()).read_only_hint is True


@pytest.mark.parametrize(
    "arguments,message",
    [
        ({"group_by": ["date:fortnight"]}, "Unknown period 'fortnight'"),
        ({"operations": {"revenue": "total"}}, "Unknown operation(s) ['total']"),
        ({"sort_by": "revenue"}, "not an output column"),
    ],
)
def test_aggregate_rejects_bad_arguments_with_the_fix(orders, arguments, message):
    base = {
        "file_path": orders,
        "group_by": ["region"],
        "operations": {"revenue": "sum"},
    }
    result = call("aggregate", {**base, **arguments})
    assert result.is_error and message in result.content[0].text


def test_profile_and_validation_point_at_offending_rows(orders):
    profile = call("profile_data", {"file_path": orders}).structured_content
    outliers = profile["column_analysis"]["revenue"]["outliers"]
    assert [item["row"] for item in outliers["most_extreme"][:2]] == [2, 3]
    validation = call(
        "validate_data",
        {"file_path": orders, "validation_rules": {"revenue": {"max_value": 25}}},
    ).structured_content
    violation = validation["validation_results"]["revenue"]["violations"][0]
    assert violation["examples"] == [
        {"row": 2, "value": 99999.0},
        {"row": 3, "value": 99999.0},
        {"row": 1, "value": 30.0},
    ]
