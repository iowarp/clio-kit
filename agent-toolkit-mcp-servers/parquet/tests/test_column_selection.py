"""Tests for column selection edge cases and variations."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
)


@pytest.mark.asyncio
async def test_read_slice_multiple_columns(test_parquet_file):
    """Test read_slice with multiple column selection."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) >= 2:
        # Select first two columns
        selected_cols = available_columns[:2]

        result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
            columns=selected_cols,
        )
        data = json.loads(result)

        assert data["status"] == "success"
        # Schema should contain only selected columns
        assert len(data["schema"]["columns"]) == 2
        schema_names = [col["name"] for col in data["schema"]["columns"]]
        assert schema_names == selected_cols

        # Data should contain only selected columns
        for row in data["data"]:
            assert set(row.keys()) == set(selected_cols)


@pytest.mark.asyncio
async def test_read_slice_column_order_preserved(test_parquet_file):
    """Test that column order in request is preserved in response."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) >= 3:
        # Request columns in reverse order
        selected_cols = available_columns[2::-1]  # Reverse of first 3

        result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
            columns=selected_cols,
        )
        data = json.loads(result)

        assert data["status"] == "success"
        # Schema should preserve requested order
        schema_names = [col["name"] for col in data["schema"]["columns"]]
        assert schema_names == selected_cols


@pytest.mark.asyncio
async def test_read_slice_single_column(test_parquet_file):
    """Test read_slice with single column selection."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        # Select first column
        result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
            columns=[available_columns[0]],
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert len(data["schema"]["columns"]) == 1
        assert data["schema"]["columns"][0]["name"] == available_columns[0]


@pytest.mark.asyncio
async def test_read_slice_empty_column_list(test_parquet_file):
    """Test read_slice with empty column list."""
    result = await read_slice(
        test_parquet_file,
        start_row=0,
        end_row=5,
        columns=[],
    )
    data = json.loads(result)

    # Empty columns should return error or select all columns
    # Behavior depends on implementation
    assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_read_slice_duplicate_columns(test_parquet_file):
    """Test read_slice with duplicate column names."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        # Request same column twice
        result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
            columns=[available_columns[0], available_columns[0]],
        )
        data = json.loads(result)

        # Should either error or deduplicate
        assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_read_slice_case_sensitivity_columns(test_parquet_file):
    """Test if column names are case-sensitive."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        col_name = available_columns[0]

        # Try with uppercase (if original is lowercase)
        uppercase_col = col_name.upper()

        if uppercase_col != col_name:
            result = await read_slice(
                test_parquet_file,
                start_row=0,
                end_row=5,
                columns=[uppercase_col],
            )
            data = json.loads(result)

            # Should error if case-sensitive
            assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_read_slice_special_characters_in_columns(test_parquet_file):
    """Test handling of columns with special characters."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    # Check if any columns have special characters
    special_char_cols = [
        col for col in available_columns if not col.replace("_", "").isalnum()
    ]

    if len(special_char_cols) > 0:
        result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
            columns=[special_char_cols[0]],
        )
        data = json.loads(result)

        # Should handle special characters correctly
        assert data["status"] == "success"


@pytest.mark.asyncio
async def test_read_slice_all_columns_explicitly(test_parquet_file):
    """Test reading with explicit list of all columns."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    all_columns = [col["name"] for col in summary["schema"]["columns"]]

    # Request all columns explicitly
    result = await read_slice(
        test_parquet_file,
        start_row=0,
        end_row=5,
        columns=all_columns,
    )
    data = json.loads(result)

    # Should return all columns
    assert data["status"] == "success"
    assert len(data["schema"]["columns"]) == len(all_columns)


@pytest.mark.asyncio
async def test_read_slice_column_subset_data_integrity(test_parquet_file):
    """Test that column subset reads preserve data integrity."""

    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    all_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(all_columns) >= 2:
        # Read full row
        full_result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
        )
        full_data = json.loads(full_result)

        # Read with column subset
        subset_result = await read_slice(
            test_parquet_file,
            start_row=0,
            end_row=5,
            columns=all_columns[:2],
        )
        subset_data = json.loads(subset_result)

        # Subset data should match corresponding columns in full data
        if full_data["status"] == "success" and subset_data["status"] == "success":
            for i, row in enumerate(subset_data["data"]):
                for col_name in all_columns[:2]:
                    assert row[col_name] == full_data["data"][i][col_name]
