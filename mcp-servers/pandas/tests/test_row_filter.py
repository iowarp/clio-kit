"""Row filters support column comparisons and refuse other expressions."""

import asyncio

import pandas as pd
import pytest
from fastmcp import Client

from pandas_mcp.implementation.row_filter import filter_rows
from pandas_mcp.server import mcp


@pytest.fixture
def frame():
    return pd.DataFrame(
        {
            "age": [22, 35, 47, 61],
            "city": ["Paris", "Rome", "Paris", "Oslo"],
            "salary": [30000, 52000, 61000, 45000],
            "unit price": [1.5, 2.0, 2.5, 3.0],
        }
    )


@pytest.mark.parametrize(
    "condition",
    [
        "age > 30",
        "age > 30 and salary > 50000",
        "age < 25 or age > 55",
        "not age > 30",
        "(age > 30) & (city == 'Paris')",
        "(age > 30) | ~(salary > 40000)",
        "city in ['Paris', 'Oslo']",
        "city not in ('Paris',)",
        "30 < age <= 47",
        "`unit price` * 2 >= 5",
        "salary / 1000 > age",
        "age != 35",
    ],
)
def test_matches_dataframe_query(frame, condition):
    expected = frame.query(condition)
    pd.testing.assert_frame_equal(filter_rows(frame, condition), expected)


@pytest.mark.parametrize(
    "condition",
    [
        "@limit == 0",
        "@pd.__version__ == ''",
        "age.__class__ == 0",
        "age.abs() > 3",
        "abs(age) > 3",
        "os == 0",
        "'x' * 10**12 == 'x'",
        "lambda: 0",
        "[c for c in age] == 0",
        "age",
        "age >",
    ],
)
def test_refuses_anything_else(frame, condition):
    with pytest.raises(ValueError):
        filter_rows(frame, condition)


def test_nullable_columns_treat_missing_as_false():
    frame = pd.DataFrame({"n": pd.array([1, None, 3], dtype="Int64")})
    assert filter_rows(frame, "n > 1")["n"].tolist() == [3]


def test_aggregate_refuses_expressions_and_still_filters(tmp_path):
    csv = tmp_path / "data.csv"
    pd.DataFrame({"g": ["a", "a", "b"], "v": [1, 2, 3]}).to_csv(csv, index=False)

    async def exercise():
        async with Client(mcp) as client:
            refused = await client.call_tool(
                "aggregate",
                {
                    "file_path": str(csv),
                    "group_by": ["g"],
                    "operations": {"v": "sum"},
                    "filter_condition": "@pd.__version__ != ''",
                },
                raise_on_error=False,
            )
            filtered = await client.call_tool(
                "aggregate",
                {
                    "file_path": str(csv),
                    "group_by": ["g"],
                    "operations": {"v": "sum"},
                    "filter_condition": "v > 1",
                },
            )
            return refused, filtered

    refused, filtered = asyncio.run(exercise())
    assert refused.is_error
    result = filtered.structured_content
    assert result["rows_filtered_out"] == 1
    assert result["results"] == [
        {"g": "a", "rows": 1, "v_sum": 2},
        {"g": "b", "rows": 1, "v_sum": 3},
    ]
