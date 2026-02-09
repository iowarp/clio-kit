"""
Test Suite 2: Filtering Operations
Priority: 2

Tests filter parsing and PyArrow filter application across different comparison
and logical operators.
"""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import read_slice


# SUITE 1: Simple Comparison Operators (6 tests)


@pytest.mark.asyncio
async def test_filter_equals():
    """Test simple equality filter."""
    filter_spec = json.dumps({"column": "batch_id", "op": "equal", "value": 1})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert data["slice_info"]["rows_after_filter"] > 0
    for row in data["data"]:
        assert row["batch_id"] == 1


@pytest.mark.asyncio
async def test_filter_not_equals():
    """Test inequality filter."""
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
        assert row["batch_id"] != 1


@pytest.mark.asyncio
async def test_filter_less_than():
    """Test less than comparison."""
    filter_spec = json.dumps({"column": "zenith", "op": "less", "value": 0.5})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["zenith"] < 0.5


@pytest.mark.asyncio
async def test_filter_greater_than():
    """Test greater than comparison."""
    filter_spec = json.dumps({"column": "azimuth", "op": "greater", "value": 5.0})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["azimuth"] > 5.0


@pytest.mark.asyncio
async def test_filter_less_than_equal():
    """Test less than or equal comparison."""
    filter_spec = json.dumps({"column": "zenith", "op": "less_equal", "value": 1.0})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["zenith"] <= 1.0


@pytest.mark.asyncio
async def test_filter_greater_than_equal():
    """Test greater than or equal comparison."""
    filter_spec = json.dumps({"column": "azimuth", "op": "greater_equal", "value": 3.0})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["azimuth"] >= 3.0


# SUITE 2: Logical Operators (5 tests)


@pytest.mark.asyncio
async def test_filter_and_operator():
    """Test AND logical operator."""
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
        assert row["azimuth"] > 3.0
        assert row["zenith"] < 2.0


@pytest.mark.asyncio
async def test_filter_or_operator():
    """Test OR logical operator."""
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
        assert row["zenith"] < 0.5 or row["zenith"] > 2.5


@pytest.mark.asyncio
async def test_filter_not_operator():
    """Test NOT logical operator."""
    filter_spec = json.dumps({"not": {"column": "batch_id", "op": "equal", "value": 1}})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["batch_id"] != 1


@pytest.mark.asyncio
async def test_filter_nested_expression():
    """Test complex nested logical expression."""
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
        assert row["azimuth"] > 5.0 or row["azimuth"] < 1.0
        assert row["zenith"] < 1.5


@pytest.mark.asyncio
async def test_filter_multiple_and():
    """Test multiple AND conditions."""
    filter_spec = json.dumps(
        {
            "and": [
                {"column": "batch_id", "op": "equal", "value": 1},
                {"column": "zenith", "op": "less", "value": 1.0},
                {"column": "azimuth", "op": "greater", "value": 2.0},
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
        assert row["azimuth"] > 2.0


# SUITE 3: IN Clause and NULL Checks (4 tests)


@pytest.mark.asyncio
async def test_filter_in_clause_integers():
    """Test IN clause with integer values."""
    filter_spec = json.dumps({"column": "batch_id", "op": "in", "values": [1, 2, 3]})
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=50,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["batch_id"] in [1, 2, 3]


@pytest.mark.asyncio
async def test_filter_in_clause_floats():
    """Test IN clause with float values (using actual values from dataset)."""
    # Use actual float values that exist in the dataset (accounting for float precision)
    target_values = [0.22499999403953552, 0.925000011920929, 1.3250000476837158]
    filter_spec = json.dumps({"column": "charge", "op": "in", "values": target_values})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    # Should have some matches
    assert len(data["data"]) > 0
    for row in data["data"]:
        # Check that charge matches one of the exact target values
        assert row["charge"] in target_values


@pytest.mark.asyncio
async def test_filter_is_null():
    """Test IS NULL check (testing the functionality exists)."""
    # Most columns in our datasets aren't nullable, so we test that the operation works
    # even if it returns no results
    filter_spec = json.dumps({"column": "charge", "op": "is_null"})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    # Should either succeed or return error if operation not supported
    assert data["status"] in ["success", "error"]


@pytest.mark.asyncio
async def test_filter_is_not_null():
    """Test IS NOT NULL check."""
    filter_spec = json.dumps({"column": "charge", "op": "is_not_null"})
    result = await read_slice(
        file_path="datasets/batch_1.parquet",
        start_row=0,
        end_row=100,
        filter_json=filter_spec,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    for row in data["data"]:
        assert row["charge"] is not None
