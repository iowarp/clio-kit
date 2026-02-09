"""Tests for the get_column_preview tool."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import get_column_preview, summarize


@pytest.mark.asyncio
async def test_get_column_preview_valid_column(test_parquet_file):
    """Test get_column_preview with a valid column name."""
    # First get the schema to know what columns are available
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        assert data["status"] == "success"
        assert "column_name" in data
        assert data["column_name"] == available_columns[0]
        assert "data" in data
        assert "pagination" in data
        assert isinstance(data["data"], list)
        assert data["pagination"]["num_items"] <= 100


@pytest.mark.asyncio
async def test_get_column_preview_nonexistent_column(test_parquet_file):
    """Test get_column_preview with a nonexistent column."""
    result = await get_column_preview(test_parquet_file, "nonexistent_column")
    data = json.loads(result)

    assert data["status"] == "error"
    assert "Column not found" in data["message"]
    assert "available_columns" in data


@pytest.mark.asyncio
async def test_get_column_preview_negative_start_index(test_parquet_file):
    """Test get_column_preview with negative start_index."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(
            test_parquet_file, available_columns[0], start_index=-1
        )
        data = json.loads(result)

        assert data["status"] == "error"
        assert "start_index must be >= 0" in data["message"]


@pytest.mark.asyncio
async def test_get_column_preview_start_index_exceeds_total(test_parquet_file):
    """Test get_column_preview with start_index exceeding total rows."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0:
        result = await get_column_preview(
            test_parquet_file, available_columns[0], start_index=total_rows + 100
        )
        data = json.loads(result)

        assert data["status"] == "error"
        assert "exceeds total rows" in data["message"]


@pytest.mark.asyncio
async def test_get_column_preview_pagination(test_parquet_file):
    """Test pagination with get_column_preview."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows > 50:
        # Get first page
        result1 = await get_column_preview(
            test_parquet_file, available_columns[0], start_index=0, max_items=20
        )
        data1 = json.loads(result1)

        assert data1["status"] == "success"
        assert len(data1["data"]) == 20
        assert data1["pagination"]["has_more"] is True
        assert data1["pagination"]["start_index"] == 0
        assert data1["pagination"]["end_index"] == 20

        # Get second page
        result2 = await get_column_preview(
            test_parquet_file, available_columns[0], start_index=20, max_items=20
        )
        data2 = json.loads(result2)

        assert data2["status"] == "success"
        assert data2["pagination"]["start_index"] == 20
        assert data2["pagination"]["end_index"] == 40
        # Data from page 1 and page 2 should be different (likely)
        # unless all values are the same
        assert len(data2["data"]) == 20


@pytest.mark.asyncio
async def test_get_column_preview_max_items_constraint(test_parquet_file):
    """Test that max_items is constrained to 100."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        # Request more than 100 items
        result = await get_column_preview(
            test_parquet_file, available_columns[0], max_items=500
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["pagination"]["num_items"] <= 100


@pytest.mark.asyncio
async def test_get_column_preview_payload_size_within_limit(test_parquet_file):
    """Test that successful responses have payload size within 16KB limit."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        if data["status"] == "success":
            assert "metadata" in data
            assert "payload_size_bytes" in data["metadata"]
            assert data["metadata"]["payload_size_bytes"] <= 16384


@pytest.mark.asyncio
async def test_get_column_preview_column_type_info(test_parquet_file):
    """Test that column type information is included."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(test_parquet_file, available_columns[0])
        data = json.loads(result)

        assert data["status"] == "success"
        assert "column_type" in data
        assert isinstance(data["column_type"], str)
        assert len(data["column_type"]) > 0


@pytest.mark.asyncio
async def test_get_column_preview_nonexistent_file(nonexistent_file):
    """Test get_column_preview with a file that doesn't exist."""
    result = await get_column_preview(nonexistent_file, "some_column")
    data = json.loads(result)

    assert data["status"] == "error"
    assert "File not found" in data["message"]


@pytest.mark.asyncio
async def test_get_column_preview_pagination_info_complete(test_parquet_file):
    """Test that pagination info is complete."""
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(
            test_parquet_file, available_columns[0], start_index=0, max_items=50
        )
        data = json.loads(result)

        assert data["status"] == "success"
        pagination = data["pagination"]
        assert "start_index" in pagination
        assert "end_index" in pagination
        assert "num_items" in pagination
        assert "total_values" in pagination
        assert "has_more" in pagination
        assert pagination["num_items"] <= 50
        assert (
            pagination["end_index"] - pagination["start_index"]
            == pagination["num_items"]
        )


@pytest.mark.asyncio
async def test_get_column_preview_payload_exceeds_limit():
    """Test get_column_preview when payload exceeds 16KB limit.

    Uses batch_large_strings.parquet which has a column with large strings
    (each ~200 bytes), so 100 items exceed 16KB.
    """
    large_file = "datasets/batch_large_strings.parquet"

    # Try to get 100 items (should fail due to size)
    result = await get_column_preview(
        large_file, "large_text", start_index=0, max_items=100
    )
    data = json.loads(result)

    assert data["status"] == "error"
    assert "Payload exceeds limit" in data["message"]
    assert "metadata" in data
    assert "recommended_max_items" in data["metadata"]
    assert "payload_size_bytes" in data["metadata"]
    assert "limit_bytes" in data["metadata"]

    # Verify recommended_max_items is less than requested
    recommended = data["metadata"]["recommended_max_items"]
    assert recommended < 100
    assert recommended > 0

    # Now try with the recommended size - should succeed
    result2 = await get_column_preview(
        large_file, "large_text", start_index=0, max_items=recommended
    )
    data2 = json.loads(result2)

    assert data2["status"] == "success"
    assert len(data2["data"]) == recommended
    assert data2["metadata"]["payload_size_bytes"] <= 16384
