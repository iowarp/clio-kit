"""Comprehensive tests for aggregate_column functionality."""

import pytest
import json
import pyarrow as pa
import pyarrow.parquet as pq


@pytest.fixture
def aggregate_test_file(tmp_path):
    """Create a test file for aggregation operations."""
    file_path = tmp_path / "aggregate_test.parquet"

    # Create a table with various numeric types
    table = pa.table(
        {
            "int_col": [1, 2, 3, 4, 5, 10, 20, 30, 40, 50],
            "float_col": [1.5, 2.5, 3.5, 4.5, 5.5, 10.5, 20.5, 30.5, 40.5, 50.5],
            "str_col": ["a", "b", "a", "b", "a", "b", "a", "b", "a", "b"],
            "null_col": [1, None, 3, None, 5, None, 7, None, 9, None],
            "batch_id": [1, 1, 1, 1, 1, 2, 2, 2, 2, 2],
        }
    )

    pq.write_table(table, file_path)
    return str(file_path)


@pytest.mark.asyncio
class TestAggregateBasicOperations:
    """Test basic aggregation operations."""

    async def test_aggregate_min(self, aggregate_test_file):
        """Test min aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "min")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 1
        assert result["operation"] == "min"

    async def test_aggregate_max(self, aggregate_test_file):
        """Test max aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "max")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 50
        assert result["operation"] == "max"

    async def test_aggregate_sum(self, aggregate_test_file):
        """Test sum aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "sum")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 165  # 1+2+3+4+5+10+20+30+40+50

    async def test_aggregate_mean(self, aggregate_test_file):
        """Test mean aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "mean")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 16.5  # 165 / 10

    async def test_aggregate_count(self, aggregate_test_file):
        """Test count aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "count")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 10

    async def test_aggregate_std(self, aggregate_test_file):
        """Test standard deviation aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "std")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] > 0  # Should have some standard deviation

    async def test_aggregate_count_distinct(self, aggregate_test_file):
        """Test count_distinct aggregation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "str_col", "count_distinct"
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 2  # 'a' and 'b'


@pytest.mark.asyncio
class TestAggregateWithFilters:
    """Test aggregation with filters."""

    async def test_aggregate_with_equal_filter(self, aggregate_test_file):
        """Test aggregation with equality filter."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        filter_json = json.dumps({"column": "batch_id", "op": "equal", "value": 1})
        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", filter_json=filter_json
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 15  # 1+2+3+4+5
        assert result["metadata"]["filter_applied"] is True
        assert result["metadata"]["rows_processed"] == 5

    async def test_aggregate_with_greater_filter(self, aggregate_test_file):
        """Test aggregation with greater than filter."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        filter_json = json.dumps({"column": "int_col", "op": "greater", "value": 10})
        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "count", filter_json=filter_json
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 4  # 20, 30, 40, 50

    async def test_aggregate_with_and_filter(self, aggregate_test_file):
        """Test aggregation with AND filter."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        filter_json = json.dumps(
            {
                "and": [
                    {"column": "batch_id", "op": "equal", "value": 2},
                    {"column": "int_col", "op": "greater", "value": 20},
                ]
            }
        )
        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", filter_json=filter_json
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 120  # 30+40+50


@pytest.mark.asyncio
class TestAggregateWithRange:
    """Test aggregation with row range constraints."""

    async def test_aggregate_with_start_end_row(self, aggregate_test_file):
        """Test aggregation with start and end row."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", start_row=0, end_row=5
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 15  # 1+2+3+4+5
        assert "range_constraint" in result["metadata"]
        assert result["metadata"]["range_constraint"]["start_row"] == 0
        assert result["metadata"]["range_constraint"]["end_row"] == 5

    async def test_aggregate_with_start_end_row_float(self, aggregate_test_file):
        """Test aggregation with float start/end row (should convert to int)."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", start_row=0.0, end_row=5.0
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 15

    async def test_aggregate_with_start_end_row_string_valid(self, aggregate_test_file):
        """Test aggregation with string start/end row that can be converted."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", start_row="0", end_row="5"
        )
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["result"] == 15

    async def test_aggregate_with_start_end_row_string_invalid(
        self, aggregate_test_file
    ):
        """Test aggregation with invalid string start/end row."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", start_row="abc", end_row=5
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "must be an integer" in result["message"]

    async def test_aggregate_with_invalid_range(self, aggregate_test_file):
        """Test aggregation with invalid range."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", start_row=5, end_row=3
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Invalid range" in result["message"]


@pytest.mark.asyncio
class TestAggregateErrorCases:
    """Test error cases for aggregate_column."""

    async def test_aggregate_nonexistent_file(self):
        """Test aggregation on non-existent file."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column("/nonexistent/file.parquet", "col", "sum")
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "File not found" in result["message"]

    async def test_aggregate_nonexistent_column(self, aggregate_test_file):
        """Test aggregation on non-existent column."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "nonexistent_col", "sum"
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Column not found" in result["message"]
        assert "available_columns" in result

    async def test_aggregate_invalid_operation(self, aggregate_test_file):
        """Test aggregation with invalid operation."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "invalid_op"
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Invalid operation" in result["message"]
        assert "valid_operations" in result

    async def test_aggregate_no_rows_after_filter(self, aggregate_test_file):
        """Test aggregation when filter eliminates all rows."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        filter_json = json.dumps({"column": "int_col", "op": "greater", "value": 1000})
        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", filter_json=filter_json
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "No rows remain after filtering" in result["message"]

    async def test_aggregate_invalid_filter_json(self, aggregate_test_file):
        """Test aggregation with invalid filter JSON."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(
            aggregate_test_file, "int_col", "sum", filter_json="not valid json"
        )
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Invalid filter JSON format" in result["message"]

    async def test_aggregate_incompatible_type(self, aggregate_test_file):
        """Test aggregation on incompatible data type."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        # Try to compute mean on string column
        result_str = await aggregate_column(aggregate_test_file, "str_col", "mean")
        result = json.loads(result_str)

        assert result["status"] == "error"
        assert "Error computing mean" in result["message"]


@pytest.mark.asyncio
class TestAggregateMetadata:
    """Test metadata in aggregation results."""

    async def test_aggregate_includes_null_count(self, aggregate_test_file):
        """Test that aggregation includes null count in metadata."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "null_col", "count")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert "null_count" in result["metadata"]
        assert result["metadata"]["null_count"] == 5  # Half the values are null

    async def test_aggregate_includes_data_type(self, aggregate_test_file):
        """Test that aggregation includes data type in metadata."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "sum")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert "data_type" in result["metadata"]
        assert "int" in result["metadata"]["data_type"].lower()

    async def test_aggregate_includes_rows_processed(self, aggregate_test_file):
        """Test that aggregation includes rows processed count."""
        from parquet_mcp.capabilities.parquet_handler import aggregate_column

        result_str = await aggregate_column(aggregate_test_file, "int_col", "count")
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["metadata"]["rows_processed"] == 10
        assert result["metadata"]["rows_before_filter"] == 10
