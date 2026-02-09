"""Comprehensive tests for filter operations and edge cases."""

import pytest
import json
import pyarrow as pa
import pyarrow.parquet as pq
from parquet_mcp.capabilities.parquet_handler import _apply_filter, _build_filter_mask


@pytest.fixture
def filter_test_file(tmp_path):
    """Create a test file for filter operations."""
    file_path = tmp_path / "filter_test.parquet"

    table = pa.table(
        {
            "int_col": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "float_col": [1.1, 2.2, 3.3, 4.4, 5.5, 6.6, 7.7, 8.8, 9.9, 10.10],
            "str_col": ["a", "b", "c", "a", "b", "c", "a", "b", "c", "a"],
            "null_col": [1, None, 3, None, 5, None, 7, None, 9, None],
            "bool_col": [
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
                True,
                False,
            ],
        }
    )

    pq.write_table(table, file_path)
    return str(file_path)


class TestBuildFilterMask:
    """Test _build_filter_mask function."""

    def test_build_filter_equal(self, filter_test_file):
        """Test building equal filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "equal", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 1
        assert filtered["int_col"][0].as_py() == 5

    def test_build_filter_not_equal(self, filter_test_file):
        """Test building not_equal filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "not_equal", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 9

    def test_build_filter_less(self, filter_test_file):
        """Test building less than filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "less", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 4  # 1, 2, 3, 4

    def test_build_filter_less_equal(self, filter_test_file):
        """Test building less_equal filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "less_equal", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 5  # 1, 2, 3, 4, 5

    def test_build_filter_greater(self, filter_test_file):
        """Test building greater than filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "greater", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 5  # 6, 7, 8, 9, 10

    def test_build_filter_greater_equal(self, filter_test_file):
        """Test building greater_equal filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "greater_equal", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 6  # 5, 6, 7, 8, 9, 10

    def test_build_filter_is_null(self, filter_test_file):
        """Test building is_null filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "null_col", "op": "is_null"}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 5  # Half are null

    def test_build_filter_is_valid(self, filter_test_file):
        """Test building is_valid filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "null_col", "op": "is_valid"}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 5  # Half are valid (non-null)

    def test_build_filter_is_not_null(self, filter_test_file):
        """Test building is_not_null filter mask (alias for is_valid)."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "null_col", "op": "is_not_null"}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 5

    def test_build_filter_is_in(self, filter_test_file):
        """Test building is_in filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "str_col", "op": "is_in", "values": ["a", "b"]}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 7  # 'a' appears 4 times, 'b' appears 3 times

    def test_build_filter_in_alias(self, filter_test_file):
        """Test building 'in' filter mask (alias for is_in)."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "in", "values": [1, 3, 5, 7, 9]}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 5


class TestLogicalFilterOperators:
    """Test logical operators in filters (AND, OR, NOT)."""

    def test_build_filter_and(self, filter_test_file):
        """Test building AND filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {
            "and": [
                {"column": "int_col", "op": "greater", "value": 3},
                {"column": "int_col", "op": "less", "value": 8},
            ]
        }

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 4  # 4, 5, 6, 7

    def test_build_filter_or(self, filter_test_file):
        """Test building OR filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {
            "or": [
                {"column": "int_col", "op": "less", "value": 3},
                {"column": "int_col", "op": "greater", "value": 8},
            ]
        }

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 4  # 1, 2, 9, 10

    def test_build_filter_not(self, filter_test_file):
        """Test building NOT filter mask."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"not": {"column": "int_col", "op": "equal", "value": 5}}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        assert len(filtered) == 9  # All except 5

    def test_build_filter_nested_logical(self, filter_test_file):
        """Test building nested logical filter."""
        table = pq.read_table(filter_test_file)
        filter_dict = {
            "or": [
                {
                    "and": [
                        {"column": "int_col", "op": "greater", "value": 7},
                        {"column": "bool_col", "op": "equal", "value": True},
                    ]
                },
                {"column": "int_col", "op": "less", "value": 3},
            ]
        }

        mask = _build_filter_mask(table, filter_dict)

        assert mask is not None
        filtered = table.filter(mask)
        # Should include: (int>7 AND bool=True) OR int<3
        # int>7 with bool=True: rows 8(9) - row 9 has True
        # int<3: rows 0(1), 1(2)
        assert len(filtered) >= 2


class TestFilterErrorCases:
    """Test error cases for filter operations."""

    def test_build_filter_missing_column_key(self, filter_test_file):
        """Test filter without 'column' key."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"op": "equal", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_missing_op_key(self, filter_test_file):
        """Test filter without 'op' key."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_nonexistent_column(self, filter_test_file):
        """Test filter on non-existent column."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "nonexistent", "op": "equal", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_missing_value_for_comparison(self, filter_test_file):
        """Test comparison filter without 'value' key."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "equal"}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_missing_values_for_is_in(self, filter_test_file):
        """Test is_in filter without 'values' key."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "str_col", "op": "is_in"}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_unknown_operation(self, filter_test_file):
        """Test filter with unknown operation."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "unknown_op", "value": 5}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_and_with_invalid_subfilter(self, filter_test_file):
        """Test AND filter with invalid subfilter."""
        table = pq.read_table(filter_test_file)
        filter_dict = {
            "and": [
                {"column": "int_col", "op": "equal", "value": 5},
                {"column": "nonexistent", "op": "equal", "value": 1},  # Invalid
            ]
        }

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_or_with_invalid_subfilter(self, filter_test_file):
        """Test OR filter with invalid subfilter."""
        table = pq.read_table(filter_test_file)
        filter_dict = {
            "or": [
                {"column": "int_col", "op": "equal", "value": 5},
                {"op": "equal", "value": 1},  # Missing column
            ]
        }

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None

    def test_build_filter_not_with_invalid_subfilter(self, filter_test_file):
        """Test NOT filter with invalid subfilter."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"not": {"column": "nonexistent", "op": "equal", "value": 5}}

        mask = _build_filter_mask(table, filter_dict)

        assert mask is None


class TestApplyFilter:
    """Test _apply_filter function."""

    def test_apply_filter_none_returns_unfiltered(self, filter_test_file):
        """Test that None filter returns unfiltered table."""
        table = pq.read_table(filter_test_file)
        original_len = len(table)

        filtered = _apply_filter(table, None)

        assert len(filtered) == original_len

    def test_apply_filter_valid_filter(self, filter_test_file):
        """Test applying a valid filter."""
        table = pq.read_table(filter_test_file)
        filter_dict = {"column": "int_col", "op": "less", "value": 5}

        filtered = _apply_filter(table, filter_dict)

        assert len(filtered) == 4

    def test_apply_filter_invalid_returns_unfiltered(self, filter_test_file):
        """Test that invalid filter returns unfiltered table."""
        table = pq.read_table(filter_test_file)
        original_len = len(table)
        filter_dict = {"column": "nonexistent", "op": "equal", "value": 5}

        filtered = _apply_filter(table, filter_dict)

        assert len(filtered) == original_len

    def test_apply_filter_handles_exception(self, filter_test_file):
        """Test that filter errors are handled gracefully."""
        table = pq.read_table(filter_test_file)
        original_len = len(table)
        # This might cause an error during filtering
        filter_dict = {
            "column": "int_col",
            "op": "equal",
            "value": "invalid_type_comparison",
        }

        # Should return unfiltered table on error
        filtered = _apply_filter(table, filter_dict)

        # Should return original table (error handling)
        assert filtered is not None
        assert len(filtered) == original_len


class TestFilterIntegrationWithReadSlice:
    """Test filters integrated with read_slice."""

    @pytest.mark.asyncio
    async def test_read_slice_with_complex_and_filter(self, filter_test_file):
        """Test read_slice with complex AND filter."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        filter_json = json.dumps(
            {
                "and": [
                    {"column": "int_col", "op": "greater_equal", "value": 3},
                    {"column": "int_col", "op": "less_equal", "value": 7},
                ]
            }
        )

        result_str = await read_slice(filter_test_file, 0, 10, filter_json=filter_json)
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["slice_info"]["rows_after_filter"] == 5  # 3, 4, 5, 6, 7

    @pytest.mark.asyncio
    async def test_read_slice_with_not_filter(self, filter_test_file):
        """Test read_slice with NOT filter."""
        from parquet_mcp.capabilities.parquet_handler import read_slice

        filter_json = json.dumps(
            {"not": {"column": "bool_col", "op": "equal", "value": True}}
        )

        result_str = await read_slice(filter_test_file, 0, 10, filter_json=filter_json)
        result = json.loads(result_str)

        assert result["status"] == "success"
        assert result["slice_info"]["rows_after_filter"] == 5  # 5 False values
