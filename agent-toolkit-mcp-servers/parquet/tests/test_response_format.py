"""Tests for response format, JSON validity, and encoding."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_summarize_response_is_valid_json(test_parquet_file):
    """Test that summarize always returns valid JSON."""
    result = await summarize(test_parquet_file)

    # Should be parseable as JSON (no exception)
    data = json.loads(result)
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_read_slice_response_is_valid_json(test_parquet_file):
    """Test that read_slice always returns valid JSON."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)

    # Should be parseable as JSON
    data = json.loads(result)
    assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_column_preview_response_is_valid_json(test_parquet_file):
    """Test that get_column_preview always returns valid JSON."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])

        # Should be parseable as JSON
        data = json.loads(result)
        assert isinstance(data, dict)


@pytest.mark.asyncio
async def test_summarize_response_roundtrip(test_parquet_file):
    """Test that summarize response can be serialized and deserialized without loss."""
    result = await summarize(test_parquet_file)
    data1 = json.loads(result)

    # Serialize again
    reserialized = json.dumps(data1)
    data2 = json.loads(reserialized)

    # Should be identical
    assert data1 == data2


@pytest.mark.asyncio
async def test_read_slice_response_roundtrip(test_parquet_file):
    """Test that read_slice response survives roundtrip serialization."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data1 = json.loads(result)

    # Serialize again
    reserialized = json.dumps(data1)
    data2 = json.loads(reserialized)

    # Should be identical
    assert data1 == data2


@pytest.mark.asyncio
async def test_column_preview_response_roundtrip(test_parquet_file):
    """Test that get_column_preview response survives roundtrip serialization."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data1 = json.loads(result)

        # Serialize again
        reserialized = json.dumps(data1)
        data2 = json.loads(reserialized)

        # Should be identical
        assert data1 == data2


@pytest.mark.asyncio
async def test_summarize_response_has_status_field(test_parquet_file):
    """Test that all responses have a status field."""
    result = await summarize(test_parquet_file)
    data = json.loads(result)

    assert "status" in data
    assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_read_slice_response_has_status_field(test_parquet_file):
    """Test that read_slice response has status field."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    assert "status" in data
    assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_column_preview_response_has_status_field(test_parquet_file):
    """Test that column_preview response has status field."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        assert "status" in data
        assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_error_responses_have_message(test_parquet_file):
    """Test that error responses always include a message field."""
    # Force an error
    result = await read_slice(test_parquet_file, start_row=-1, end_row=10)
    data = json.loads(result)

    if data["status"] == "error":
        assert "message" in data
        assert isinstance(data["message"], str)
        assert len(data["message"]) > 0


@pytest.mark.asyncio
async def test_summarize_response_utf8_encoding(test_parquet_file):
    """Test that responses are properly UTF-8 encoded."""
    result = await summarize(test_parquet_file)

    # Should be valid UTF-8
    result.encode("utf-8")  # Should not raise

    # Should parse
    data = json.loads(result)
    assert data is not None


@pytest.mark.asyncio
async def test_read_slice_string_values_utf8(test_parquet_file):
    """Test that string values in responses are valid UTF-8."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=10)
    data = json.loads(result)

    if data["status"] == "success":
        for row in data["data"]:
            for col_name, value in row.items():
                if isinstance(value, str):
                    # Should be valid UTF-8
                    value.encode("utf-8")


@pytest.mark.asyncio
async def test_summarize_response_no_nan_inf(test_parquet_file):
    """Test that responses don't contain NaN or Infinity (invalid JSON)."""
    result = await summarize(test_parquet_file)

    # These should not appear in valid JSON
    assert "NaN" not in result or "NaN" in json.dumps(json.loads(result))
    assert "Infinity" not in result or "Infinity" in json.dumps(json.loads(result))
    assert "-Infinity" not in result or "-Infinity" in json.dumps(json.loads(result))


@pytest.mark.asyncio
async def test_read_slice_response_contains_required_fields(test_parquet_file):
    """Test that success responses contain all required fields."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    if data["status"] == "success":
        required_fields = [
            "status",
            "file_path",
            "slice_info",
            "schema",
            "data",
            "shape",
        ]
        for field in required_fields:
            assert field in data, f"Missing required field: {field}"


@pytest.mark.asyncio
async def test_column_preview_response_contains_required_fields(test_parquet_file):
    """Test that success responses contain all required fields."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        if data["status"] == "success":
            required_fields = [
                "status",
                "column_name",
                "column_type",
                "data",
                "pagination",
            ]
            for field in required_fields:
                assert field in data, f"Missing required field: {field}"


@pytest.mark.asyncio
async def test_summarize_schema_field_structure(test_parquet_file):
    """Test that schema field has proper structure."""
    result = await summarize(test_parquet_file)
    data = json.loads(result)

    if data["status"] == "success":
        assert "schema" in data
        schema = data["schema"]

        # Should have columns array
        assert "columns" in schema
        assert isinstance(schema["columns"], list)

        # Each column should have required fields
        for col in schema["columns"]:
            assert "name" in col
            assert "type" in col
            assert "nullable" in col
            assert isinstance(col["name"], str)
            assert isinstance(col["type"], str)
            assert isinstance(col["nullable"], bool)


@pytest.mark.asyncio
async def test_read_slice_numeric_field_types(test_parquet_file):
    """Test that numeric fields in responses are correct types."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    if data["status"] == "success":
        # shape should have integer fields
        assert isinstance(data["shape"]["rows"], int)
        assert isinstance(data["shape"]["columns"], int)

        # slice_info should have integer fields
        assert isinstance(data["slice_info"]["start_row"], int)
        assert isinstance(data["slice_info"]["end_row"], int)
        assert isinstance(data["slice_info"]["requested_rows"], int)
        assert isinstance(data["slice_info"]["rows_after_filter"], int)
