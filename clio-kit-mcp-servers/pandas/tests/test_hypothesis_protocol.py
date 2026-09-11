"""Verify statistical results survive a real MCP serialization round trip."""

import asyncio

import pytest
from fastmcp import Client
from scipy import stats
from pandas_mcp.server import mcp


@pytest.mark.parametrize("kind", ["t_test", "normality", "mann_whitney"])
def test_hypothesis_results_are_structured_and_match_scipy(tmp_path, kind):
    path = tmp_path / "samples.csv"
    path.write_text("x,y\n1,2\n2,4\n3,6\n4,8\n")
    expected = {
        "t_test": stats.ttest_ind([1, 2, 3, 4], [2, 4, 6, 8]),
        "normality": stats.shapiro([1, 2, 3, 4]),
        "mann_whitney": stats.mannwhitneyu([1, 2, 3, 4], [2, 4, 6, 8]),
    }[kind]
    result = call(
        {"file_path": str(path), "test_type": kind, "column1": "x", "column2": "y"}
    )
    data = result.structured_content
    assert data["results"]["p_value"] == pytest.approx(expected.pvalue)
    assert type(data["results"]["is_significant"]) is bool
    assert data["results"]["effect_size"] == "not_computed"


def test_invalid_hypothesis_is_a_tool_error(tmp_path):
    path = tmp_path / "samples.csv"
    path.write_text("x\n1\n2\n3\n")
    result = call({"file_path": str(path), "test_type": "unknown", "column1": "x"})
    assert result.is_error
    assert "Unknown test type" in result.content[0].text


def call(arguments):
    async def exchange():
        async with Client(mcp) as client:
            return await client.call_tool(
                "hypothesis_testing", arguments, raise_on_error=False
            )

    return asyncio.run(exchange())
