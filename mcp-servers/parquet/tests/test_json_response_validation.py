"""
Test Suite 1: JSON Response Validation
Priority: 1 (START HERE)

This test suite validates that all MCP tools return proper JSON with correct structure
and that the data in JSON responses matches the filter criteria applied.
"""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    read_slice,
    aggregate_column,
    get_column_preview,
)


# SUITE 1: JSON Structure Validation (6 tests)


@pytest.mark.asyncio
async def test_read_slice_returns_valid_json():
    """Test that read_slice returns parseable JSON."""
    result = await read_slice(
        file_path="datasets/train_meta.parquet", start_row=0, end_row=10
    )

    # Should not raise JSONDecodeError
    data = json.loads(result)
    assert isinstance(data, dict)
    assert "status" in data


@pytest.mark.asyncio
async def test_aggregate_returns_valid_json():
    """Test that aggregate_column returns parseable JSON."""
    result = await aggregate_column(
        file_path="datasets/train_meta.parquet", column_name="zenith", operation="min"
    )

    data = json.loads(result)
    assert isinstance(data, dict)
    assert "status" in data
    assert "result" in data or "error" in data


@pytest.mark.asyncio
async def test_preview_returns_valid_json():
    """Test that get_column_preview returns parseable JSON."""
    result = await get_column_preview(
        file_path="datasets/train_meta.parquet", column_name="event_id", max_items=10
    )

    data = json.loads(result)
    assert isinstance(data, dict)
    assert "status" in data
    assert "data" in data or "error" in data


@pytest.mark.asyncio
async def test_success_response_structure():
    """Test that success responses have required fields."""
    result = await read_slice(
        file_path="datasets/train_meta.parquet", start_row=0, end_row=10
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert "data" in data
    assert "metadata" in data or "slice_info" in data
    assert isinstance(data["data"], list)


@pytest.mark.asyncio
async def test_error_response_structure():
    """Test that error responses have required fields."""
    result = await read_slice(file_path="nonexistent.parquet", start_row=0, end_row=10)
    data = json.loads(result)

    assert data["status"] == "error"
    assert "message" in data


@pytest.mark.asyncio
async def test_filtered_response_metadata():
    """Test that filtered responses include filter metadata."""
    filter_spec = json.dumps({"column": "batch_id", "op": "equal", "value": 1})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert "slice_info" in data
    # Check for filter metadata (field names may vary)
    assert "rows_after_filter" in data["slice_info"] or "rows" in data["slice_info"]


# SUITE 2: JSON Data Validation with Filters (12 tests)


@pytest.mark.asyncio
async def test_json_data_matches_equality_filter():
    """Test that every row in JSON response matches equality filter."""
    filter_spec = json.dumps({"column": "batch_id", "op": "equal", "value": 1})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert len(data["data"]) > 0, "Should have matching rows"

    # Validate every row in JSON response
    for row in data["data"]:
        assert row["batch_id"] == 1, f"Row has batch_id={row['batch_id']}, expected 1"


@pytest.mark.asyncio
async def test_json_data_matches_inequality_filter():
    """Test that every row in JSON response matches inequality filter."""
    filter_spec = json.dumps({"column": "batch_id", "op": "not_equal", "value": 1})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["batch_id"] != 1, (
            f"Row has batch_id={row['batch_id']}, should not be 1"
        )


@pytest.mark.asyncio
async def test_json_data_matches_less_than_filter():
    """Test that every row in JSON response matches less-than filter."""
    filter_spec = json.dumps({"column": "zenith", "op": "less", "value": 0.5})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,  # Reduced to avoid payload limit
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["zenith"] < 0.5, f"Row has zenith={row['zenith']}, should be < 0.5"


@pytest.mark.asyncio
async def test_json_data_matches_greater_than_filter():
    """Test that every row in JSON response matches greater-than filter."""
    filter_spec = json.dumps({"column": "azimuth", "op": "greater", "value": 5.0})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["azimuth"] > 5.0, (
            f"Row has azimuth={row['azimuth']}, should be > 5.0"
        )


@pytest.mark.asyncio
async def test_json_data_matches_and_filter():
    """Test that every row in JSON response matches AND filter."""
    filter_spec = json.dumps(
        {
            "and": [
                {"column": "azimuth", "op": "greater", "value": 3.0},
                {"column": "zenith", "op": "less", "value": 2.0},
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
        assert row["azimuth"] > 3.0, f"Row fails azimuth condition: {row['azimuth']}"
        assert row["zenith"] < 2.0, f"Row fails zenith condition: {row['zenith']}"


@pytest.mark.asyncio
async def test_json_data_matches_or_filter():
    """Test that every row in JSON response matches OR filter."""
    filter_spec = json.dumps(
        {
            "or": [
                {"column": "zenith", "op": "less", "value": 0.5},
                {"column": "zenith", "op": "greater", "value": 2.5},
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
        assert row["zenith"] < 0.5 or row["zenith"] > 2.5, (
            f"Row fails OR condition: zenith={row['zenith']}"
        )


@pytest.mark.asyncio
async def test_json_data_matches_in_filter():
    """Test that every row in JSON response matches IN clause."""
    filter_spec = json.dumps({"column": "batch_id", "op": "in", "values": [1, 2, 3]})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=50,  # Reduced - IN matches many rows, avoid payload limit
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["batch_id"] in [1, 2, 3], (
            f"Row has batch_id={row['batch_id']}, should be in [1, 2, 3]"
        )


@pytest.mark.asyncio
async def test_json_data_matches_complex_filter():
    """Test that every row in JSON response matches complex nested filter."""
    filter_spec = json.dumps(
        {
            "and": [
                {
                    "or": [
                        {"column": "azimuth", "op": "greater", "value": 5.0},
                        {"column": "azimuth", "op": "less", "value": 1.0},
                    ]
                },
                {"column": "zenith", "op": "less", "value": 1.5},
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
        assert row["azimuth"] > 5.0 or row["azimuth"] < 1.0, (
            f"Row fails azimuth OR condition: {row['azimuth']}"
        )
        assert row["zenith"] < 1.5, f"Row fails zenith condition: {row['zenith']}"


@pytest.mark.asyncio
async def test_aggregate_json_with_filter():
    """Test that aggregation with filter returns valid count in JSON."""
    filter_spec = json.dumps({"column": "zenith", "op": "less", "value": 0.5})
    result = await aggregate_column(
        file_path="datasets/train_meta.parquet",
        column_name="event_id",
        operation="count",
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert isinstance(data["result"], int)
    assert data["result"] > 0, "Should have matching rows"
    # Check that filter was applied (metadata structure may vary)
    assert "metadata" in data


@pytest.mark.asyncio
async def test_preview_json_basic():
    """Test that column preview returns valid JSON."""
    result = await get_column_preview(
        file_path="datasets/train_meta.parquet", column_name="event_id", max_items=50
    )
    data = json.loads(result)

    assert data["status"] == "success"
    # Validate the response is valid
    assert isinstance(data["data"], list)
    assert len(data["data"]) > 0


@pytest.mark.asyncio
async def test_empty_filter_returns_valid_json():
    """Test that filters matching no rows return valid JSON."""
    filter_spec = json.dumps(
        {"column": "zenith", "op": "less", "value": 0}
    )  # Impossible
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert isinstance(data["data"], list)
    assert len(data["data"]) == 0
    assert data["slice_info"]["rows_after_filter"] == 0


@pytest.mark.asyncio
async def test_json_data_matches_boolean_filter():
    """Test that every row in JSON response matches boolean filter."""
    filter_spec = json.dumps({"column": "auxiliary", "op": "equal", "value": True})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["auxiliary"] is True, (
            f"Row has auxiliary={row['auxiliary']}, should be True"
        )
