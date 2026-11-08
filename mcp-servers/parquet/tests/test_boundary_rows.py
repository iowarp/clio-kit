"""Tests for boundary row conditions and edge cases."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_read_slice_first_row_only(test_parquet_file):
    """Test reading exactly the first row (start=0, end=1)."""
    result = await read_slice(test_parquet_file, start_row=0, end_row=1)
    data = json.loads(result)

    assert data["status"] == "success"
    assert len(data["data"]) == 1
    assert data["shape"]["rows"] == 1
    assert data["slice_info"]["rows_after_filter"] == 1


@pytest.mark.asyncio
async def test_read_slice_last_row(test_parquet_file):
    """Test reading the last row of the file."""
    # First get total rows
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    total_rows = summary["num_rows"]

    # Read last row
    result = await read_slice(
        test_parquet_file,
        start_row=total_rows - 1,
        end_row=total_rows,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert len(data["data"]) == 1
    assert data["shape"]["rows"] == 1


@pytest.mark.asyncio
async def test_read_slice_last_n_rows(test_parquet_file):
    """Test reading the last N rows (near end boundary)."""
    # First get total rows
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    total_rows = summary["num_rows"]

    # Read last 10 rows
    start = total_rows - 10
    result = await read_slice(
        test_parquet_file,
        start_row=start,
        end_row=total_rows,
    )
    data = json.loads(result)

    assert data["status"] == "success"
    assert len(data["data"]) == 10
    assert data["shape"]["rows"] == 10


@pytest.mark.asyncio
async def test_read_slice_middle_range(test_parquet_file):
    """Test reading from middle of file."""
    # First get total rows
    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    total_rows = summary["num_rows"]

    if total_rows > 200:
        # Read middle 50 rows
        start = total_rows // 2 - 25
        end = start + 50

        result = await read_slice(
            test_parquet_file,
            start_row=start,
            end_row=end,
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert len(data["data"]) == 50
        assert data["slice_info"]["start_row"] == start


@pytest.mark.asyncio
async def test_read_slice_row_ordering(test_parquet_file):
    """Test that rows maintain original order and are contiguous."""
    # Read first batch
    result1 = await read_slice(test_parquet_file, start_row=0, end_row=5)
    data1 = json.loads(result1)

    # Read overlapping second batch
    result2 = await read_slice(test_parquet_file, start_row=2, end_row=7)
    data2 = json.loads(result2)

    assert data1["status"] == "success"
    assert data2["status"] == "success"

    # Rows 2-4 should be identical in both results
    for i in range(3):
        assert data1["data"][2 + i] == data2["data"][i]


@pytest.mark.asyncio
async def test_column_preview_start_at_zero(test_parquet_file):
    """Test column preview starting from index 0."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(
            test_parquet_file, available_columns[0], start_index=0, max_items=10
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["pagination"]["start_index"] == 0
        assert data["pagination"]["end_index"] == 10


@pytest.mark.asyncio
async def test_column_preview_at_end_boundary(test_parquet_file):
    """Test column preview near end of column."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows > 10:
        # Read last 10 items
        result = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=total_rows - 10,
            max_items=10,
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert data["pagination"]["has_more"] is False
        assert data["pagination"]["num_items"] == 10


@pytest.mark.asyncio
async def test_column_preview_first_item_only(test_parquet_file):
    """Test column preview requesting exactly 1 item."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]

    if len(available_columns) > 0:
        result = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=0,
            max_items=1,
        )
        data = json.loads(result)

        assert data["status"] == "success"
        assert len(data["data"]) == 1
        assert data["pagination"]["num_items"] == 1


@pytest.mark.asyncio
async def test_column_preview_multiple_pages_to_end(test_parquet_file):
    """Test that pagination correctly identifies end of data."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows > 50:
        # Test pagination from the end backwards
        # Request items starting near the end
        near_end_start = max(0, total_rows - 100)

        result = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=near_end_start,
            max_items=50,
        )
        data = json.loads(result)

        assert data["status"] == "success"
        # When we request from near the end, has_more should eventually be False
        # or we should get fewer items than requested
        assert data["pagination"]["num_items"] <= 50

        # If file is small enough, should be able to reach actual end
        if total_rows < 200:
            # Small file: paginate to actual end
            last_result = await get_column_preview(
                test_parquet_file,
                available_columns[0],
                start_index=total_rows - 1,
                max_items=1,
            )
            last_data = json.loads(last_result)
            assert last_data["status"] == "success"
            assert last_data["pagination"]["has_more"] is False


@pytest.mark.asyncio
async def test_read_slice_exact_row_count(test_parquet_file):
    """Test that read_slice returns exactly the requested number of rows."""
    # Test various sizes
    for start in [0, 5, 10]:
        for count in [1, 5, 20]:
            result = await read_slice(
                test_parquet_file,
                start_row=start,
                end_row=start + count,
            )
            data = json.loads(result)

            if data["status"] == "success":
                assert len(data["data"]) == count
                assert data["shape"]["rows"] == count
                assert data["slice_info"]["rows_after_filter"] == count


@pytest.mark.asyncio
async def test_column_preview_consistency_across_ranges(test_parquet_file):
    """Test that overlapping column previews are consistent."""
    from parquet_mcp.capabilities.parquet_handler import summarize

    summary_result = await summarize(test_parquet_file)
    summary = json.loads(summary_result)
    available_columns = [col["name"] for col in summary["schema"]["columns"]]
    total_rows = summary["num_rows"]

    if len(available_columns) > 0 and total_rows > 30:
        # Get 0-10
        result1 = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=0,
            max_items=10,
        )
        data1 = json.loads(result1)

        # Get 5-15
        result2 = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=5,
            max_items=10,
        )
        data2 = json.loads(result2)

        # Get 0-20
        result3 = await get_column_preview(
            test_parquet_file,
            available_columns[0],
            start_index=0,
            max_items=20,
        )
        data3 = json.loads(result3)

        # Items 5-9 should be identical
        if (
            data1["status"] == "success"
            and data2["status"] == "success"
            and data3["status"] == "success"
        ):
            for i in range(5):
                assert data1["data"][5 + i] == data2["data"][i]
                assert data1["data"][5 + i] == data3["data"][5 + i]
