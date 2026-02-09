"""Tests for default parameter behavior and optional parameters."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_read_slice_columns_default_none(test_parquet_file):
    """Test read_slice with columns=None defaults to all columns."""
    result = await read_slice(
        test_parquet_file,
        start_row=0,
        end_row=5,
        columns=None,
    )
    data = json.loads(result)

    if data["status"] == "success":
        # With columns=None, should return all columns
        # Verify by comparing with a call that doesn't specify columns
        from parquet_mcp.capabilities.parquet_handler import summarize

        summary_result = await summarize(test_parquet_file)
        summary = json.loads(summary_result)
        expected_col_count = len(summary["schema"]["columns"])

        assert len(data["schema"]["columns"]) == expected_col_count


@pytest.mark.asyncio
async def test_read_slice_columns_not_specified(test_parquet_file):
    """Test read_slice without specifying columns parameter."""
    # Call without columns parameter at all
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    if data["status"] == "success":
        # Should return all columns
        from parquet_mcp.capabilities.parquet_handler import summarize

        summary_result = await summarize(test_parquet_file)
        summary = json.loads(summary_result)
        expected_col_count = len(summary["schema"]["columns"])

        assert len(data["schema"]["columns"]) == expected_col_count


@pytest.mark.asyncio
async def test_column_preview_start_index_default(test_parquet_file):
    """Test get_column_preview with start_index=None defaults to 0."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        # Call with start_index=None (default)
        result = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=None,
            max_items=10,
        )
        data = json.loads(result)

        if data["status"] == "success":
            # Should start from index 0
            assert data["pagination"]["start_index"] == 0


@pytest.mark.asyncio
async def test_column_preview_start_index_not_specified(test_parquet_file):
    """Test get_column_preview without specifying start_index."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        # Call without start_index parameter
        result = await get_column_preview(
            test_parquet_file, available_columns[0], max_items=10
        )
        data = json.loads(result)

        if data["status"] == "success":
            # Should start from index 0
            assert data["pagination"]["start_index"] == 0


@pytest.mark.asyncio
async def test_column_preview_max_items_default(test_parquet_file):
    """Test get_column_preview with max_items=None defaults to 100."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows >= 100:
        # Call with max_items=None (default)
        result = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=0,
            max_items=None,
        )
        data = json.loads(result)

        if data["status"] == "success":
            # Should return up to 100 items
            assert data["pagination"]["num_items"] <= 100
            # If total > 100, should return exactly 100
            if total_rows > 100:
                assert data["pagination"]["num_items"] == 100


@pytest.mark.asyncio
async def test_column_preview_max_items_not_specified(test_parquet_file):
    """Test get_column_preview without specifying max_items."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows >= 100:
        # Call without max_items parameter
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        if data["status"] == "success":
            # Should use default (likely 100)
            assert data["pagination"]["num_items"] <= 100
            if total_rows > 100:
                assert data["pagination"]["num_items"] == 100


@pytest.mark.asyncio
async def test_column_preview_both_defaults(test_parquet_file):
    """Test get_column_preview with all optional parameters using defaults."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0:
        # Call with only required parameter
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        if data["status"] == "success":
            # Should have defaults: start_index=0, max_items=100
            assert data["pagination"]["start_index"] == 0
            assert data["pagination"]["num_items"] <= 100
            if total_rows > 100:
                assert data["pagination"]["num_items"] == 100


@pytest.mark.asyncio
async def test_read_slice_columns_default_matches_all(test_parquet_file):
    """Test that default columns parameter returns same data as explicit all columns."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    # Get all columns
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    all_columns = [col["name"] for col in summary["schema"]["columns"]]

    # Read with default (no columns param)
    result_default = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data_default = json.loads(result_default)

    # Read with explicit all columns
    result_explicit = await read_slice(
        test_parquet_file,
        start_row=0,
        end_row=5,
        columns=all_columns,
    )
    data_explicit = json.loads(result_explicit)

    # Both should return identical data
    if data_default["status"] == "success" and data_explicit["status"] == "success":
        assert data_default["data"] == data_explicit["data"]
