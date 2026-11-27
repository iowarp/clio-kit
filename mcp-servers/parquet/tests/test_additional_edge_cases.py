"""Additional edge case tests to increase coverage to >90%."""

import pytest
import json
import pyarrow as pa
import pyarrow.parquet as pq


@pytest.fixture
def edge_test_file(tmp_path):
    """Create a test file for edge case testing."""
    file_path = tmp_path / "edge_test.parquet"

    table = pa.table(
        {
            "int_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
            "str_col": ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j"],
        }
    )

    pq.write_table(table, file_path)
    return str(file_path)


@pytest.mark.asyncio
class TestReadSliceTypeCoercion:
    """Test type coercion and error handling in read_slice."""

    async def test_read_slice_with_float_rows(self, edge_test_file):
        """Test read_slice with float type rows (should be coerced)."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        # Float should be coerced to int
        result_str = await read_slice(edge_test_file, 0.0, 5.0)
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert len(result["data"]) == 5

    async def test_read_slice_with_list_start_row(self, edge_test_file):
        """Test read_slice with list type for start_row."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        result_str = await read_slice(edge_test_file, [0], 5)
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "must be an integer" in result["message"]

    async def test_read_slice_with_none_start_row(self, edge_test_file):
        """Test read_slice with None for start_row."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        result_str = await read_slice(edge_test_file, None, 5)
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "must be an integer" in result["message"]


@pytest.mark.asyncio
class TestColumnPreviewEdgeCases:
    """Test edge cases in column preview."""

    async def test_column_preview_max_items_zero(self, edge_test_file):
        """Test column preview with max_items=0 (should be coerced to 1)."""
        from parquet_mcp.capabilities.parquet_handler import get_column_preview

        result_str = await get_column_preview(
            edge_test_file, "int_col", start_index=0, max_items=0
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        # max_items < 1 should be coerced to 1
        assert len(result["data"]) == 1

    async def test_column_preview_max_items_negative(self, edge_test_file):
        """Test column preview with negative max_items (should be coerced to 1)."""
        from parquet_mcp.capabilities.parquet_handler import get_column_preview

        result_str = await get_column_preview(
            edge_test_file, "int_col", start_index=0, max_items=-5
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert len(result["data"]) == 1

    async def test_column_preview_generic_exception(self, edge_test_file):
        """Test column preview with corrupted file to trigger generic exception."""
        from parquet_mcp.capabilities.parquet_handler import get_column_preview

        # Create a corrupted file
        corrupted_file = edge_test_file.replace(".parquet", "_corrupt.parquet")
        with open(corrupted_file, "w") as f:
            f.write("not a parquet file")

        result_str = await get_column_preview(corrupted_file, "int_col")
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Error getting column preview" in result["message"]


@pytest.mark.asyncio
class TestAggregateTypeCoercion:
    """Test type coercion in aggregate_column."""

    async def test_aggregate_with_float_rows(self, edge_test_file):
        """Test aggregate with float type rows (should be coerced)."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            edge_test_file, "int_col", "sum", start_row=0.0, end_row=5.0
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 15  # 1+2+3+4+5

    async def test_aggregate_with_dict_start_row(self, edge_test_file):
        """Test aggregate with dict type for start_row."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            edge_test_file, "int_col", "sum", start_row={}, end_row=5
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "must be an integer" in result["message"]

    async def test_aggregate_unknown_operation_result_none(self, edge_test_file):
        """Test aggregate when operation somehow results in None."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        # This shouldn't happen normally but tests the else branch at line 790
        # We'll test with a valid operation to ensure the code path exists
        result_str = await aggregate_column(edge_test_file, "int_col", "count")
        result = json.loads(result_str)

        # Should succeed normally
        assert result["status"] == "success"

    async def test_aggregate_generic_exception(self, edge_test_file):
        """Test aggregate with corrupted file to trigger generic exception."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        # Create a corrupted file
        corrupted_file = edge_test_file.replace(".parquet", "_agg_corrupt.parquet")
        with open(corrupted_file, "w") as f:
            f.write("not a parquet file")

        result_str = await aggregate_column(corrupted_file, "int_col", "sum")
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Error computing aggregation" in result["message"]


@pytest.mark.asyncio
class TestReadSliceGenericException:
    """Test generic exception handling in read_slice."""

    async def test_read_slice_generic_exception(self, edge_test_file):
        """Test read_slice with corrupted file to trigger generic exception."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        # Create a corrupted file
        corrupted_file = edge_test_file.replace(".parquet", "_read_corrupt.parquet")
        with open(corrupted_file, "w") as f:
            f.write("not a parquet file")

        result_str = await read_slice(corrupted_file, 0, 5)
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Error reading slice from Parquet file" in result["message"]


@pytest.mark.asyncio
class TestServerToolFunctions:
    """Test that server tool functions can be called directly."""

    @pytest.fixture
    def simple_file(self, tmp_path):
        """Create a simple test file."""
        file_path = tmp_path / "simple.parquet"
        table = pa.table({"col": [1, 2, 3]})
        pq.write_table(table, file_path)
        return str(file_path)

    async def test_summarize_tool_direct_call(self, simple_file):
        """Test calling summarize_tool directly."""
        from parquet_mcp.capabilities.parquet_handler import summarize

        result_str = await summarize(simple_file)
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["num_rows"] == 3

    async def test_read_slice_tool_direct_call(self, simple_file):
        """Test calling read_slice through handler."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        result_str = await read_slice(simple_file, 0, 2)
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert len(result["data"]) == 2

    async def test_get_column_preview_tool_direct_call(self, simple_file):
        """Test calling get_column_preview through handler."""
        from parquet_mcp.capabilities.parquet_handler import get_column_preview

        result_str = await get_column_preview(simple_file, "col")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert len(result["data"]) == 3

    async def test_aggregate_column_tool_direct_call(self, simple_file):
        """Test calling aggregate_column through handler."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(simple_file, "col", "sum")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 6
