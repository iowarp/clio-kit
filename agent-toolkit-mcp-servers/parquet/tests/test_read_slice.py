"""Tests for the read_slice tool."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import read_slice


@pytest.mark.asyncio
async def test_read_slice_valid_range(test_parquet_file):
    """Test read_slice with a valid row range."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "success"
    assert "file_path" in data
    assert "slice_info" in data
    assert "schema" in data
    assert "data" in data
    assert "shape" in data
    assert data["slice_info"]["start_row"] == 0
    assert data["slice_info"]["end_row"] == 10
    assert data["slice_info"]["rows_after_filter"] == 10
    assert len(data["data"]) == 10
    assert data["shape"]["rows"] == 10


@pytest.mark.asyncio
async def test_read_slice_with_columns(test_parquet_file):
    """Test read_slice with column filtering."""
    # First get the schema to know what columns are available
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        # Test with a single column
        result = await read_slice(
            test_parquet_file, start_row=0, end_row=5, columns=[available_columns[0]]
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert len(data["schema"]["columns"]) == 1
        assert data["schema"]["columns"][0]["name"] == available_columns[0]


@pytest.mark.asyncio
async def test_read_slice_invalid_column(test_parquet_file):
    """Test read_slice with an invalid column name."""
    result = await read_slice(
        test_parquet_file, start_row=0, end_row=10, columns=["nonexistent_column"]
    )
    data = json.loads(result)

    assert data["status"] == "error"
    assert "Invalid columns" in data["message"]
    assert "available_columns" in data


@pytest.mark.asyncio
async def test_read_slice_negative_start_row(test_parquet_file):
    """Test read_slice with negative start_row."""
    result = await read_slice(test_parquet_file, start_row=-1, end_row=10)
    data = json.loads(result)

    assert data["status"] == "error"
    assert "start_row must be >= 0" in data["message"]


@pytest.mark.asyncio
async def test_read_slice_end_row_exceeds_total(test_parquet_file):
    """Test read_slice with end_row exceeding total rows."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    total_rows = summary["num_rows"]

    result = await read_slice(test_parquet_file, start_row=0, end_row=total_rows + 100)
    data = json.loads(result)

    assert data["status"] == "error"
    assert "exceeds total rows" in data["message"]


@pytest.mark.asyncio
async def test_read_slice_invalid_range(test_parquet_file):
    """Test read_slice with start_row >= end_row."""
    result = await read_slice(test_parquet_file, start_row=10, end_row=10)
    data = json.loads(result)

    assert data["status"] == "error"
    assert "start_row must be less than end_row" in data["message"]


@pytest.mark.asyncio
async def test_read_slice_nonexistent_file(nonexistent_file):
    """Test read_slice with a file that doesn't exist."""
    result = await read_slice(nonexistent_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "error"
    assert "File not found" in data["message"]


@pytest.mark.asyncio
async def test_read_slice_schema_structure(test_parquet_file):
    """Test that schema in response has proper structure."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    schema = data["schema"]
    assert "columns" in schema
    assert isinstance(schema["columns"], list)
    assert len(schema["columns"]) > 0

    # Check first column structure
    col = schema["columns"][0]
    assert "name" in col
    assert "type" in col
    assert "nullable" in col


@pytest.mark.asyncio
async def test_read_slice_payload_size_within_limit(test_parquet_file):
    """Test that successful responses have payload size within 16KB limit."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    if data["status"] == "success":
        assert "metadata" in data
        assert "payload_size_bytes" in data["metadata"]
        assert data["metadata"]["payload_size_bytes"] <= 16384


@pytest.mark.asyncio
async def test_read_slice_shape_matches_data(test_parquet_file):
    """Test that shape metadata matches actual data."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    assert data["status"] == "success"
    assert data["shape"]["rows"] == len(data["data"])
    assert data["shape"]["columns"] == len(data["schema"]["columns"])
