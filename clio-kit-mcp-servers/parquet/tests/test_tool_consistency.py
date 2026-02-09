"""Tests for consistency and integration between MCP tools."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_summarize_schema_matches_read_slice_schema(test_parquet_file):
    """Test that schema is identical between summarize and read_slice."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)

    slice_result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    slice_data = json.loads(slice_result)

    # Extract schemas
    summary_schema = {
        col["name"]: (col["type"], col["nullable"])
        for col in summary["schema"]["columns"]
    }
    slice_schema = {
        col["name"]: (col["type"], col["nullable"])
        for col in slice_data["schema"]["columns"]
    }

    # Schemas should match exactly
    assert summary_schema == slice_schema


@pytest.mark.asyncio
async def test_summarize_num_rows_matches_column_preview_total(test_parquet_file):
    """Test that summarize num_rows matches column_preview total_values."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    total_rows_from_summary = summary["num_rows"]

    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        preview_result = await get_column_preview(
            test_parquet_file, available_columns[0]
        )
        preview_data = json.loads(preview_result)

        if preview_data["status"] == "success":
            total_values_from_preview = preview_data["pagination"]["total_values"]
            assert total_rows_from_summary == total_values_from_preview


@pytest.mark.asyncio
async def test_column_names_consistency_across_tools(test_parquet_file):
    """Test that column names are consistent across all tools."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    summary_columns = [col["name"] for col in summary["schema"]["columns"]]

    slice_result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    slice_data = json.loads(slice_result)
    slice_columns = [col["name"] for col in slice_data["schema"]["columns"]]

    # Column names should be identical and in same order
    assert summary_columns == slice_columns

    # Test each column with get_column_preview
    for col_name in summary_columns:
        preview_result = await get_column_preview(test_parquet_file, col_name)
        preview_data = json.loads(preview_result)

        if preview_data["status"] == "success":
            assert preview_data["column_name"] == col_name


@pytest.mark.asyncio
async def test_column_order_preserved_across_tools(test_parquet_file):
    """Test that column order is preserved across tools."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    summary_order = [col["name"] for col in summary["schema"]["columns"]]

    # Read multiple slices and verify column order
    for start in [0, 5, 10]:
        slice_result = await read_slice(
            test_parquet_file, start_row=start, end_row=start + 5
        )
        slice_data = json.loads(slice_result)

        if slice_data["status"] == "success":
            slice_order = [col["name"] for col in slice_data["schema"]["columns"]]
            assert slice_order == summary_order


@pytest.mark.asyncio
async def test_read_slice_data_matches_column_preview_subset(test_parquet_file):
    """Test that read_slice data matches corresponding column_preview data."""
    # Get a slice of data
    slice_result = await read_slice(test_parquet_file, start_row=0, end_row=20)
    slice_data = json.loads(slice_result)

    assert slice_data["status"] == "success"

    # Get column preview for same range
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        col_name = available_columns[0]

        preview_result = await get_column_preview(
            test_parquet_file, col_name, start_index=0, max_items=20
        )
        preview_data = json.loads(preview_result)

        if preview_data["status"] == "success":
            # Extract column data from slice
            slice_column_data = [row[col_name] for row in slice_data["data"]]

            # Should match preview data
            assert slice_column_data == preview_data["data"]


@pytest.mark.asyncio
async def test_column_types_consistency(test_parquet_file):
    """Test that column types are consistent when reading different slices."""
    # Get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    schema_types = {col["name"]: col["type"] for col in summary["schema"]["columns"]}

    # Read multiple slices and verify types don't change
    for start in [0, 10, 20]:
        slice_result = await read_slice(
            test_parquet_file, start_row=start, end_row=start + 5
        )
        slice_data = json.loads(slice_result)

        if slice_data["status"] == "success":
            slice_types = {
                col["name"]: col["type"] for col in slice_data["schema"]["columns"]
            }
            assert slice_types == schema_types


@pytest.mark.asyncio
async def test_file_metadata_consistency(test_parquet_file):
    """Test that file metadata is consistent across multiple calls."""
    # Call summarize multiple times
    result1 = await summarize(test_parquet_file)
    data1 = json.loads(result1)

    result2 = await summarize(test_parquet_file)
    data2 = json.loads(result2)

    # Key metadata should be identical
    assert data1["num_rows"] == data2["num_rows"]
    assert data1["num_row_groups"] == data2["num_row_groups"]
    assert data1["file_size_bytes"] == data2["file_size_bytes"]


@pytest.mark.asyncio
async def test_nullable_consistency_across_tools(test_parquet_file):
    """Test that nullable information is consistent across all tools."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    summary_nullable = {
        col["name"]: col["nullable"] for col in summary["schema"]["columns"]
    }

    # Check in read_slice
    slice_result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    slice_data = json.loads(slice_result)
    slice_nullable = {
        col["name"]: col["nullable"] for col in slice_data["schema"]["columns"]
    }

    assert summary_nullable == slice_nullable


@pytest.mark.asyncio
async def test_column_preview_column_type_matches_schema(test_parquet_file):
    """Test that column_type in preview matches schema type."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    schema_dict = {col["name"]: col["type"] for col in summary["schema"]["columns"]}
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        col_name = available_columns[0]
        schema_type = schema_dict[col_name]

        preview_result = await get_column_preview(test_parquet_file, col_name)
        preview_data = json.loads(preview_result)

        if preview_data["status"] == "success":
            # column_type in preview should match schema type
            assert preview_data["column_type"] == schema_type


@pytest.mark.asyncio
async def test_row_count_matches_across_slices(test_parquet_file):
    """Test that reading slices gives correct row counts that sum properly."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    total_rows = summary["num_rows"]

    # Test with a small sample of the file (first 1000 rows or less)
    # Don't iterate through entire 32M+ row file
    sample_size = min(1000, total_rows)
    chunk_size = 100
    total_read = 0

    for start in range(0, sample_size, chunk_size):
        end = min(start + chunk_size, sample_size)
        slice_result = await read_slice(test_parquet_file, start_row=start, end_row=end)
        slice_data = json.loads(slice_result)

        if slice_data["status"] == "success":
            total_read += len(slice_data["data"])

    # For the sample we read, should match the slice sizes
    assert total_read == sample_size


@pytest.mark.asyncio
async def test_summarize_column_preview_read_slice_workflow(test_parquet_file):
    """Test realistic workflow: summarize → column_preview → read_slice on that column.

    This validates the user workflow where:
    1. summarize provides schema/metadata
    2. column_preview shows sample values for inspection
    3. read_slice retrieves full data for the selected column
    """
    # Step 1: summarize to get schema
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)

    assert summary["status"] == "success"
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    assert len(available_columns) > 0

    # Step 2: get_column_preview to inspect sample data
    col_name = available_columns[0]
    preview_result = await get_column_preview(test_parquet_file, col_name)
    preview_data = json.loads(preview_result)

    assert preview_data["status"] == "success"
    assert preview_data["column_name"] == col_name
    assert len(preview_data["data"]) > 0

    # Step 3: read_slice to get full data for that column
    slice_result = await read_slice(
        test_parquet_file,
        start_row=0,
        end_row=min(100, summary["num_rows"]),
        columns=[col_name],
    )
    slice_data = json.loads(slice_result)

    assert slice_data["status"] == "success"
    # Should have exactly one column in response
    assert len(slice_data["schema"]["columns"]) == 1
    assert slice_data["schema"]["columns"][0]["name"] == col_name

    # Data consistency: first values from read_slice should match preview data
    slice_column_values = [row[col_name] for row in slice_data["data"]]
    # Preview should contain at least some of the values from read_slice
    preview_values = preview_data["data"]

    # Check that at least the first few preview values appear in the slice
    for i in range(min(5, len(preview_values), len(slice_column_values))):
        assert preview_values[i] == slice_column_values[i]
