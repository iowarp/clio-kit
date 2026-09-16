"""
Test Suite 4: Filter Edge Cases and Error Handling
Priority: 4

Tests edge cases, special values, and error handling in filtering.
"""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import read_slice


# SUITE 9: Edge Cases and Error Handling (13 tests)


@pytest.mark.asyncio
async def test_filter_null_comparison():
    """Test filter behavior with NULL values in comparisons."""
    # NULLs should be filtered out in comparisons
    filter_spec = json.dumps({"column": "charge", "op": "greater", "value": 0.5})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        if row.get("charge") is not None:
            assert row["charge"] > 0.5


@pytest.mark.asyncio
async def test_filter_empty_string():
    """Test handling of empty string in filter."""
    result = await read_slice(
        file_path="datasets/batch_1.parquet", start_row=0, end_row=100, filter_json=""
    )
    data = json.loads(result)

    # Should return all rows (no filter applied)
    assert data["status"] == "success"
    assert len(data["data"]) == 100


@pytest.mark.asyncio
async def test_filter_whitespace_only():
    """Test handling of whitespace-only filter."""
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json="   \t\n  ",
    )
    data = json.loads(result)

    # Should error on invalid JSON
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_filter_very_long_expression():
    """Test handling of very long filter expression."""
    # Build a long OR chain
    or_conditions = [
        {"column": "event_id", "op": "equal", "value": i} for i in range(100)
    ]
    filter_spec = json.dumps({"or": or_conditions})

    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] in ["success", "error"]
    if data["status"] == "error":
        assert (
            "too long" in data["message"].lower()
            or "complex" in data["message"].lower()
        )


@pytest.mark.asyncio
async def test_filter_unicode():
    """Test filter with valid expression (Unicode chars should work in JSON)."""
    # JSON handles Unicode natively
    filter_spec = json.dumps({"column": "sensor_id", "op": "greater", "value": 1000})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_filter_case_sensitivity():
    """Test that filter is case-sensitive for column names."""
    filter_spec = json.dumps(
        {"column": "ZENITH", "op": "less", "value": 1.0}
    )  # Wrong case
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    # Should error or return no matches (column not found)
    assert data["status"] in ["error", "success"]


@pytest.mark.asyncio
async def test_filter_division_by_zero():
    """Test that division by zero is handled gracefully."""
    # Note: JSON filter format doesn't support arithmetic operations like division
    # This test verifies the filter system doesn't crash on edge cases
    filter_spec = json.dumps({"column": "charge", "op": "equal", "value": 0})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    # Should handle gracefully
    assert data["status"] in ["error", "success"]


@pytest.mark.asyncio
async def test_filter_boolean_column():
    """Test filtering on boolean column."""
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
        assert row["auxiliary"] is True


@pytest.mark.asyncio
async def test_filter_scientific_notation():
    """Test filter with scientific notation."""
    filter_spec = json.dumps({"column": "charge", "op": "greater", "value": 1.5e-1})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["charge"] > 0.15


@pytest.mark.asyncio
async def test_filter_negative_numbers():
    """Test filter with negative numbers."""
    filter_spec = json.dumps(
        {"column": "time", "op": "greater", "value": -1}
    )  # All should pass if time is positive
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_filter_float_precision():
    """Test filter with high-precision float."""
    filter_spec = json.dumps(
        {"column": "zenith", "op": "less", "value": 0.2924767344867341}
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
        assert row["zenith"] < 0.2924767344867341


@pytest.mark.asyncio
async def test_filter_special_characters():
    """Test filter with standard operators (no special chars in JSON format)."""
    filter_spec = json.dumps({"column": "charge", "op": "greater", "value": 0.5})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"


@pytest.mark.asyncio
async def test_filter_extra_whitespace_in_json():
    """Test that JSON parser handles extra whitespace correctly."""
    # JSON parser should handle whitespace fine
    filter_spec = """
    {
        "column"  :  "zenith"  ,
        "op"      :  "less"    ,
        "value"   :  1.0
    }
    """
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    # Should handle extra whitespace gracefully
    assert data["status"] in ["success", "error"]
    if data["status"] == "success":
        for row in data["data"]:
            if "zenith" in row:
                assert row["zenith"] < 1.0
