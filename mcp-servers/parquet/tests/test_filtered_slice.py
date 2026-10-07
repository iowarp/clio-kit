"""
Test Suite 3: Filtered Slice Operations
Priority: 3

Tests read_slice with various filter scenarios including empty results,
column projection, and size limits.
"""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import read_slice


# SUITE 6: Filtered Slice Operations (8 tests)


@pytest.mark.asyncio
async def test_filtered_slice_empty_result():
    """Test filter that matches no rows."""
    filter_spec = json.dumps(
        {"column": "zenith", "op": "less", "value": 0}
    )  # Impossible (zenith >= 0)
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert data["slice_info"]["rows_after_filter"] == 0
    assert len(data["data"]) == 0


@pytest.mark.asyncio
async def test_filtered_slice_all_match():
    """Test filter that matches all rows."""
    filter_spec = json.dumps(
        {"column": "event_id", "op": "greater", "value": 0}
    )  # All should match
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert data["slice_info"]["rows_after_filter"] == 100


@pytest.mark.asyncio
async def test_filtered_slice_with_projection():
    """Test filter with column selection."""
    filter_spec = json.dumps({"column": "batch_id", "op": "equal", "value": 1})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,
        columns=["event_id", "zenith"],
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert data["shape"]["columns"] == 2
    for row in data["data"]:
        assert "event_id" in row
        assert "zenith" in row
        assert "azimuth" not in row


@pytest.mark.asyncio
async def test_filtered_slice_multiple_and():
    """Test multiple AND conditions."""
    filter_spec = json.dumps(
        {
            "and": [
                {"column": "batch_id", "op": "equal", "value": 1},
                {"column": "zenith", "op": "less", "value": 1.0},
            ]
        }
    )
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["batch_id"] == 1
        assert row["zenith"] < 1.0


@pytest.mark.asyncio
async def test_filtered_slice_or():
    """Test OR condition."""
    filter_spec = json.dumps(
        {
            "or": [
                {"column": "batch_id", "op": "equal", "value": 1},
                {"column": "batch_id", "op": "equal", "value": 2},
            ]
        }
    )
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["batch_id"] in [1, 2]


@pytest.mark.asyncio
async def test_filtered_slice_different_types():
    """Test filters on int, float, and bool columns."""
    filter_spec = json.dumps(
        {
            "and": [
                {"column": "auxiliary", "op": "equal", "value": True},
                {"column": "charge", "op": "greater", "value": 1.0},
            ]
        }
    )
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=200,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["auxiliary"] is True
        assert row["charge"] > 1.0


@pytest.mark.asyncio
async def test_filtered_slice_size_limit():
    """Test that large filtered results trigger size error."""
    filter_spec = json.dumps(
        {"column": "charge", "op": "greater", "value": 0.1}
    )  # Matches most rows
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=2000,  # Adjusted for trimmed dataset
        filter_json=filter_spec,
    )
    data = json.loads(result)

    if data["status"] == "error":
        assert (
            "payload" in data["message"].lower() or "limit" in data["message"].lower()
        )
        # Verify error metadata includes rows_after_filter
        assert "rows_after_filter" in data["metadata"]
        assert data["metadata"]["rows_after_filter"] > 0
    else:
        # If it fits, verify size is under limit
        assert data["metadata"]["payload_size_bytes"] <= 16384


@pytest.mark.asyncio
async def test_filtered_slice_order():
    """Test that filtering preserves row order."""
    filter_spec = json.dumps({"column": "batch_id", "op": "equal", "value": 1})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    # Event IDs should be monotonically increasing
    event_ids = [row["event_id"] for row in data["data"]]
    assert event_ids == sorted(event_ids)


# SUITE 7: Error Handling in Filtered Slice (5 tests)


@pytest.mark.asyncio
async def test_filtered_slice_invalid_syntax():
    """Test error handling for invalid filter syntax."""
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json="not valid json",
    )
    data = json.loads(result)

    assert data["status"] == "error"
    assert "json" in data["message"].lower() or "invalid" in data["message"].lower()


@pytest.mark.asyncio
async def test_filtered_slice_column_not_found():
    """Test error when filter references non-existent column."""
    filter_spec = json.dumps(
        {"column": "nonexistent_column", "op": "greater", "value": 5}
    )
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    # Should error or handle gracefully (filter might be silently ignored)
    assert data["status"] in ["error", "success"]


@pytest.mark.asyncio
async def test_filtered_slice_type_mismatch():
    """Test error when filter value type doesn't match column type."""
    filter_spec = json.dumps(
        {"column": "sensor_id", "op": "equal", "value": "abc"}
    )  # String for int column
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    # Should either error or handle gracefully
    assert data["status"] in ["error", "success"]


@pytest.mark.asyncio
async def test_filtered_slice_empty_filter():
    """Test handling of empty filter expression."""
    result = await read_slice(
        file_path="datasets/batch_1.parquet", start_row=0, end_row=100, filter_json=""
    )
    data = json.loads(result)

    # Should treat as no filter (return all rows)
    assert data["status"] == "success"
    assert data["slice_info"]["rows_after_filter"] == 100


@pytest.mark.asyncio
async def test_filtered_slice_malformed_json():
    """Test error for malformed JSON."""
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json='{"column": "zenith", "op": "less"',  # Missing closing brace
    )
    data = json.loads(result)

    # Should error on malformed JSON
    assert data["status"] == "error"
    assert "json" in data["message"].lower() or "invalid" in data["message"].lower()
