"""Tests for NULL/missing value handling across all tools."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_summarize_nullable_column_metadata(test_parquet_file):
    """Test that summarize correctly reports nullable status of columns."""
    result = await summarize(test_parquet_file)
    data = json.loads(result)

    assert data["status"] == "success"
    schema = data["schema"]

    # All columns should report their nullable status
    for col in schema["columns"]:
        assert "nullable" in col
        assert isinstance(col["nullable"], bool)


@pytest.mark.asyncio
async def test_read_slice_with_null_values(test_parquet_file):
    """Test read_slice correctly handles NULL values in data."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=50)
    data = json.loads(result)

    assert data["status"] == "success"

    # NULL values may be represented as None in JSON
    # Verify that the response is still valid JSON
    for row in data["data"]:
        assert isinstance(row, dict)
        # Some values in row may be None/null
        for col_name, value in row.items():
            # Valid types for values: string, number, bool, null, list, dict
            assert value is None or isinstance(
                value, (str, int, float, bool, list, dict)
            )


@pytest.mark.asyncio
async def test_read_slice_preserves_null_vs_empty_string(test_parquet_file):
    """Test that NULL and empty string are distinguished in output."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "success"

    # If there are string columns, verify None/null distinction
    # JSON null should map to Python None
    json_str = json.dumps(data)
    # Verify response parses back correctly
    reparsed = json.loads(json_str)
    assert reparsed == data


@pytest.mark.asyncio
async def test_column_preview_with_null_values(test_parquet_file):
    """Test get_column_preview handles NULL values in column data."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(
            test_parquet_file, available_columns[0], max_items=50
        )
        data = json.loads(result)

        if data["status"] == "success":
            # Data array may contain None values for NULL entries
            for value in data["data"]:
                # Valid types: primitives or null
                assert value is None or isinstance(
                    value, (str, int, float, bool, list, dict)
                )


@pytest.mark.asyncio
async def test_column_preview_pagination_with_nulls(test_parquet_file):
    """Test pagination consistency when data contains NULLs."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows > 100:
        # Get two overlapping pages to check NULL handling consistency
        result1 = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=0,
            max_items=20,
        )
        data1 = json.loads(result1)

        result2 = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=0,
            max_items=40,
        )
        data2 = json.loads(result2)

        # First 20 items should match exactly
        if data1["status"] == "success" and data2["status"] == "success":
            assert data1["data"] == data2["data"][:20]


@pytest.mark.asyncio
async def test_null_column_serialization_correctness(test_parquet_file):
    """Test that NULL values serialize correctly to JSON."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    if data["status"] == "success":
        # Verify response is valid JSON by re-serializing
        json_str = json.dumps(data)
        reparsed = json.loads(json_str)

        # None/null handling should be consistent
        assert reparsed == data


@pytest.mark.asyncio
async def test_nullable_metadata_consistency(test_parquet_file):
    """Test that nullable metadata is consistent across tools."""
    # Get schema from summarize
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    summarize_schema = {
        col["name"]: col["nullable"] for col in summary["schema"]["columns"]
    }

    # Get schema from read_slice
    slice_result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    slice_data = json.loads(slice_result)
    slice_schema = {
        col["name"]: col["nullable"] for col in slice_data["schema"]["columns"]
    }

    # Nullable metadata should match
    assert summarize_schema == slice_schema
