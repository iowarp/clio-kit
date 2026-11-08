"""Tests for the summarize tool."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import summarize


@pytest.mark.asyncio
async def test_summarize_valid_file(test_parquet_file):
    """Test summarize with a valid Parquet file."""
    result = await summarize(test_parquet_file)
    data = json.loads(result)

    assert data["status"] == "success"
    assert "filename" in data
    assert "schema" in data
    assert "num_rows" in data
    assert "num_row_groups" in data
    assert "file_size_bytes" in data
    assert isinstance(data["num_rows"], int)
    assert data["num_rows"] > 0
    assert isinstance(data["num_row_groups"], int)
    assert data["num_row_groups"] > 0


@pytest.mark.asyncio
async def test_summarize_nonexistent_file(nonexistent_file):
    """Test summarize with a file that doesn't exist."""
    result = await summarize(nonexistent_file)
    data = json.loads(result)

    assert data["status"] == "error"
    assert "message" in data


@pytest.mark.asyncio
async def test_summarize_schema_structure(test_parquet_file):
    """Test that schema has proper structure."""
    result = await summarize(test_parquet_file)
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
    assert isinstance(col["name"], str)
    assert isinstance(col["type"], str)
    assert isinstance(col["nullable"], bool)
