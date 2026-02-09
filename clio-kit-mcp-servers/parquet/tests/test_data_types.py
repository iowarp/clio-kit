"""Tests for data type specific handling and serialization."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_summarize_reports_all_column_types(test_parquet_file):
    """Test that summarize reports actual column types."""
    result = await summarize(test_parquet_file)
    data = json.loads(result)

    assert data["status"] == "success"
    schema = data["schema"]

    # All columns should have a type
    for col in schema["columns"]:
        assert "type" in col
        assert isinstance(col["type"], str)
        assert len(col["type"]) > 0
        # Type should be a PyArrow or common type name (int32, int64, string, double, etc.)


@pytest.mark.asyncio
async def test_read_slice_numeric_precision(test_parquet_file):
    """Test that numeric values maintain precision in JSON serialization."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    assert data["status"] == "success"

    # Get schema to identify numeric columns
    schema = data["schema"]
    numeric_types = {"int32", "int64", "float", "double", "decimal"}

    for row in data["data"]:
        for col in schema["columns"]:
            if (
                col["type"] in numeric_types
                or "int" in col["type"]
                or "float" in col["type"]
            ):
                value = row.get(col["name"])
                if value is not None:
                    # Numeric values should be numbers (int or float)
                    assert isinstance(value, (int, float))


@pytest.mark.asyncio
async def test_read_slice_string_encoding(test_parquet_file):
    """Test that string values are properly UTF-8 encoded."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "success"

    # Get schema to identify string columns
    schema = data["schema"]
    string_types = {"string", "utf8", "large_string"}

    for row in data["data"]:
        for col in schema["columns"]:
            if col["type"] in string_types or "string" in col["type"].lower():
                value = row.get(col["name"])
                if value is not None:
                    assert isinstance(value, str)
                    # Verify it's valid UTF-8 by re-encoding
                    assert value.encode("utf-8").decode("utf-8") == value


@pytest.mark.asyncio
async def test_read_slice_boolean_serialization(test_parquet_file):
    """Test that boolean values serialize correctly."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "success"

    # Get schema to identify boolean columns
    schema = data["schema"]

    for row in data["data"]:
        for col in schema["columns"]:
            if "bool" in col["type"].lower():
                value = row.get(col["name"])
                if value is not None:
                    # Booleans should be Python bool (true/false in JSON)
                    assert isinstance(value, bool)


@pytest.mark.asyncio
async def test_column_preview_type_consistency(test_parquet_file):
    """Test that column_preview maintains type consistency."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    schema_dict = {col["name"]: col["type"] for col in summary["schema"]["columns"]}
    available_columns = list(schema_dict.keys())

    if len(available_columns) > 0:
        col_name = available_columns[0]
        col_type = schema_dict[col_name]

        result = await get_column_preview(test_parquet_file, col_name, max_items=20)
        data = json.loads(result)

        if data["status"] == "success":
            # All values in the column should match the expected type
            for value in data["data"]:
                if value is None:
                    continue

                if any(t in col_type.lower() for t in ["string"]):
                    assert isinstance(value, str)
                elif any(t in col_type.lower() for t in ["int", "int32", "int64"]):
                    assert isinstance(value, int)
                elif any(t in col_type.lower() for t in ["float", "double", "decimal"]):
                    assert isinstance(value, (int, float))
                elif any(t in col_type.lower() for t in ["bool"]):
                    assert isinstance(value, bool)


@pytest.mark.asyncio
async def test_column_preview_special_float_values(test_parquet_file):
    """Test handling of special float values (NaN, Inf, -Inf)."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [
        col
        for col in summary["schema"]["columns"]
        if "float" in col["type"].lower() or "double" in col["type"].lower()
    ]

    if len(available_columns) > 0:
        col_name = available_columns[0]["name"]
        result = await get_column_preview(test_parquet_file, col_name, max_items=50)
        data = json.loads(result)

        if data["status"] == "success":
            # JSON doesn't support NaN/Inf, so they should be handled specially
            # Either as null, or as special string representations
            for value in data["data"]:
                if value is not None:
                    # Valid representations: number, "NaN", "Infinity", "-Infinity", null
                    assert isinstance(value, (int, float)) or value in [
                        "NaN",
                        "Infinity",
                        "-Infinity",
                    ]


@pytest.mark.asyncio
async def test_read_slice_timestamp_serialization(test_parquet_file):
    """Test that timestamp/datetime values are properly serialized."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "success"

    # Get schema to identify timestamp/date columns
    schema = data["schema"]
    time_types = {"timestamp", "date", "time"}

    for row in data["data"]:
        for col in schema["columns"]:
            if any(t in col["type"].lower() for t in time_types):
                value = row.get(col["name"])
                if value is not None:
                    # Timestamps should serialize to ISO strings or numbers
                    assert isinstance(value, (str, int, float))
                    # If string, verify it's ISO format or standard datetime format
                    if isinstance(value, str):
                        # Should contain date-like characters
                        assert any(c.isdigit() for c in value)


@pytest.mark.asyncio
async def test_read_slice_list_array_serialization(test_parquet_file):
    """Test serialization of list/array columns."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "success"

    # Get schema to identify list/array columns
    schema = data["schema"]

    for row in data["data"]:
        for col in schema["columns"]:
            if "list" in col["type"].lower() or "array" in col["type"].lower():
                value = row.get(col["name"])
                if value is not None:
                    # List/array values should serialize to JSON arrays
                    assert isinstance(value, list)


@pytest.mark.asyncio
async def test_column_type_in_response(test_parquet_file):
    """Test that column_type is reported in get_column_preview response."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        if data["status"] == "success":
            assert "column_type" in data
            assert isinstance(data["column_type"], str)
            # column_type should match what's in schema
            assert len(data["column_type"]) > 0
