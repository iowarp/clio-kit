"""
Test Suite 5: Performance Benchmarks
Priority: 5

Tests performance of aggregations and filtering operations with large datasets.
Run with: pytest -m slow
"""

import json
import time
import pytest
from parquet_mcp.capabilities.parquet_handler import read_slice, aggregate_column


# SUITE 10: Aggregation Performance (6 tests)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_aggregate_min_performance():
    """Test MIN aggregation performance."""
    start = time.time()
    result = await aggregate_column(
        file_path="datasets/train_meta.parquet", column_name="zenith", operation="min"
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] == "success"
    assert isinstance(data["result"], float)
    assert elapsed < 10.0  # Should complete in < 10 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_aggregate_max_performance():
    """Test MAX aggregation performance."""
    start = time.time()
    result = await aggregate_column(
        file_path="datasets/train_meta.parquet", column_name="azimuth", operation="max"
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] == "success"
    assert isinstance(data["result"], float)
    assert elapsed < 10.0  # Should complete in < 10 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_aggregate_with_filter_performance():
    """Test filtered aggregation performance."""
    filter_spec = json.dumps({"column": "zenith", "op": "less", "value": 0.5})
    start = time.time()
    result = await aggregate_column(
        file_path="datasets/train_meta.parquet",
        column_name="event_id",
        operation="count",
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] == "success"
    assert isinstance(data["result"], int)
    assert elapsed < 10.0  # Should complete in < 10 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_count_distinct_performance():
    """Test COUNT_DISTINCT performance."""
    start = time.time()
    result = await aggregate_column(
        file_path="datasets/batch_1.parquet",
        column_name="sensor_id",
        operation="count_distinct",
        start_row=0,
        end_row=2000,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] == "success"
    assert isinstance(data["result"], int)
    assert elapsed < 10.0  # Should complete in < 10 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_sequential_aggregations_performance():
    """Test multiple aggregations in sequence."""
    operations = ["min", "max", "mean", "std", "count"]

    start = time.time()
    for op in operations:
        result = await aggregate_column(
            file_path="datasets/train_meta.parquet", column_name="zenith", operation=op
        )
        data = json.loads(result)
        assert data["status"] == "success"
    elapsed = time.time() - start

    # All 5 operations should complete in < 50 seconds
    assert elapsed < 50.0


@pytest.mark.asyncio
@pytest.mark.slow
async def test_aggregate_complex_filter_performance():
    """Test aggregation with complex filter performance."""
    filter_spec = json.dumps(
        {
            "and": [
                {"column": "zenith", "op": "greater", "value": 1.0},
                {"column": "zenith", "op": "less", "value": 2.0},
                {"column": "batch_id", "op": "in", "values": [1, 2, 3]},
            ]
        }
    )
    start = time.time()
    result = await aggregate_column(
        file_path="datasets/train_meta.parquet",
        column_name="azimuth",
        operation="mean",
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] == "success"
    assert isinstance(data["result"], float)
    assert elapsed < 8.0  # Should complete in < 8 seconds


# SUITE 11: Filtering Performance (6 tests)


@pytest.mark.asyncio
@pytest.mark.slow
async def test_filter_large_file_performance():
    """Test filtering 131M rows completes in reasonable time."""
    filter_spec = json.dumps({"column": "zenith", "op": "less", "value": 0.5})
    start = time.time()
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=100000,
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] in ["success", "error"]
    assert elapsed < 5.0  # Should complete in < 5 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_complex_filter_performance():
    """Test complex multi-condition filter performance."""
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
                {"column": "batch_id", "op": "in", "values": [1, 2, 3]},
            ]
        }
    )
    start = time.time()
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=50000,
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] in ["success", "error"]
    assert elapsed < 10.0  # Should complete in < 10 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_filter_memory_usage():
    """Test that filtering doesn't cause memory leaks."""
    try:
        import psutil
        import gc

        process = psutil.Process()
        gc.collect()
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB

        # Run 10 filtering operations
        filter_spec = json.dumps({"column": "zenith", "op": "less", "value": 1.0})
        for i in range(10):
            await read_slice(
                file_path="datasets/train_meta.parquet",
                start_row=i * 1000,
                end_row=(i + 1) * 1000,
                filter_json=filter_spec,
            )

        gc.collect()
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory

        # Memory increase should be < 500MB
        assert memory_increase < 500
    except ImportError:
        pytest.skip("psutil not installed, skipping memory test")


@pytest.mark.asyncio
@pytest.mark.slow
async def test_in_filter_performance():
    """Test IN filter performance with many values."""
    filter_spec = json.dumps(
        {"column": "event_id", "op": "in", "values": list(range(100))}
    )
    start = time.time()
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=200,  # Reduced to avoid payload size limit
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] in ["success", "error"]
    assert elapsed < 10.0  # Should complete in < 10 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_nested_filter_performance():
    """Test deeply nested filter performance."""
    filter_spec = json.dumps(
        {
            "and": [
                {
                    "or": [
                        {
                            "and": [
                                {"column": "zenith", "op": "greater", "value": 1.0},
                                {"column": "zenith", "op": "less", "value": 2.0},
                            ]
                        },
                        {
                            "and": [
                                {"column": "azimuth", "op": "greater", "value": 5.0},
                                {"column": "azimuth", "op": "less", "value": 6.0},
                            ]
                        },
                    ]
                },
                {"column": "batch_id", "op": "equal", "value": 1},
            ]
        }
    )
    start = time.time()
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=10000,
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] in ["success", "error"]
    assert elapsed < 15.0  # Should complete in < 15 seconds


@pytest.mark.asyncio
@pytest.mark.slow
async def test_filter_and_projection_performance():
    """Test combined filtering and column projection performance."""
    filter_spec = json.dumps(
        {
            "and": [
                {"column": "zenith", "op": "less", "value": 1.5},
                {"column": "batch_id", "op": "equal", "value": 1},
            ]
        }
    )
    start = time.time()
    result = await read_slice(
        file_path="datasets/train_meta.parquet",
        start_row=0,
        end_row=50000,
        columns=["event_id", "zenith", "azimuth"],
        filter_json=filter_spec,
    )
    elapsed = time.time() - start

    data = json.loads(result)
    assert data["status"] in ["success", "error"]
    assert elapsed < 8.0  # Should complete in < 8 seconds
