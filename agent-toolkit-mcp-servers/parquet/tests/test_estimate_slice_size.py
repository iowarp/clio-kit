"""Tests for _estimate_slice_size function."""

import pytest
import pyarrow as pa
import pyarrow.parquet as pq
from parquet_mcp.capabilities.parquet_handler import _estimate_slice_size


@pytest.fixture
def size_test_file(tmp_path):
    """Create a test file for size estimation."""
    file_path = tmp_path / "size_test.parquet"

    # Create a table with multiple row groups
    table = pa.table(
        {
            "int_col": list(range(1000)),
            "float_col": [float(i) * 1.5 for i in range(1000)],
            "str_col": [f"string_{i:04d}" for i in range(1000)],
        }
    )

    # Write with specific row group size to ensure multiple row groups
    pq.write_table(table, file_path, row_group_size=100)
    return str(file_path)


class TestEstimateSliceSize:
    """Test _estimate_slice_size function."""

    def test_estimate_full_table(self, size_test_file):
        """Test estimating size of full table."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, 0, 1000)

        assert estimated_size > 0
        assert isinstance(estimated_size, int)

    def test_estimate_partial_slice(self, size_test_file):
        """Test estimating size of partial slice."""
        pq_file = pq.ParquetFile(size_test_file)

        full_size = _estimate_slice_size(pq_file, 0, 1000)
        half_size = _estimate_slice_size(pq_file, 0, 500)

        # Half the rows should be roughly half the size
        assert half_size < full_size
        assert half_size > 0

    def test_estimate_with_column_selection(self, size_test_file):
        """Test estimating size with column selection."""
        pq_file = pq.ParquetFile(size_test_file)

        all_columns_size = _estimate_slice_size(pq_file, 0, 1000)
        one_column_size = _estimate_slice_size(pq_file, 0, 1000, columns=["int_col"])

        # One column should be smaller than all columns
        assert one_column_size < all_columns_size
        assert one_column_size > 0

    def test_estimate_with_multiple_columns(self, size_test_file):
        """Test estimating size with multiple columns."""
        pq_file = pq.ParquetFile(size_test_file)

        two_columns_size = _estimate_slice_size(
            pq_file, 0, 1000, columns=["int_col", "float_col"]
        )
        one_column_size = _estimate_slice_size(pq_file, 0, 1000, columns=["int_col"])

        # Two columns should be larger than one column
        assert two_columns_size > one_column_size

    def test_estimate_single_row(self, size_test_file):
        """Test estimating size of single row."""
        pq_file = pq.ParquetFile(size_test_file)

        single_row_size = _estimate_slice_size(pq_file, 0, 1)

        assert single_row_size > 0
        assert single_row_size < _estimate_slice_size(pq_file, 0, 10)

    def test_estimate_with_negative_start_row(self, size_test_file):
        """Test estimation with negative start_row returns 0."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, -1, 100)

        assert estimated_size == 0

    def test_estimate_with_end_exceeds_total(self, size_test_file):
        """Test estimation with end_row exceeding total returns 0."""
        pq_file = pq.ParquetFile(size_test_file)
        metadata = pq_file.metadata
        total_rows = metadata.num_rows

        estimated_size = _estimate_slice_size(pq_file, 0, total_rows + 100)

        assert estimated_size == 0

    def test_estimate_with_start_equal_end(self, size_test_file):
        """Test estimation with start_row == end_row returns 0."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, 100, 100)

        assert estimated_size == 0

    def test_estimate_with_start_greater_than_end(self, size_test_file):
        """Test estimation with start_row > end_row returns 0."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, 200, 100)

        assert estimated_size == 0

    def test_estimate_with_nonexistent_columns(self, size_test_file):
        """Test estimation with non-existent columns."""
        pq_file = pq.ParquetFile(size_test_file)

        # Non-existent columns should be filtered out, returning 0
        estimated_size = _estimate_slice_size(
            pq_file, 0, 100, columns=["nonexistent_col"]
        )

        assert estimated_size == 0

    def test_estimate_with_mixed_valid_invalid_columns(self, size_test_file):
        """Test estimation with mix of valid and invalid columns."""
        pq_file = pq.ParquetFile(size_test_file)

        # Should only count valid column
        estimated_size = _estimate_slice_size(
            pq_file, 0, 100, columns=["int_col", "nonexistent"]
        )

        assert estimated_size > 0

    def test_estimate_middle_range(self, size_test_file):
        """Test estimating size of middle range."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, 400, 600)

        assert estimated_size > 0

    def test_estimate_last_rows(self, size_test_file):
        """Test estimating size of last rows."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, 900, 1000)

        assert estimated_size > 0

    def test_estimate_spanning_row_groups(self, size_test_file):
        """Test estimation for slice spanning multiple row groups."""
        pq_file = pq.ParquetFile(size_test_file)
        metadata = pq_file.metadata

        # File has row groups of 100 rows each
        # This slice spans 3 row groups (100-400)
        estimated_size = _estimate_slice_size(pq_file, 150, 350)

        assert estimated_size > 0
        # Verify we have multiple row groups
        assert metadata.num_row_groups > 1

    def test_estimate_exact_row_group_boundary(self, size_test_file):
        """Test estimation at exact row group boundaries."""
        pq_file = pq.ParquetFile(size_test_file)

        # Row groups are 100 rows each
        estimated_size = _estimate_slice_size(pq_file, 100, 200)

        assert estimated_size > 0

    def test_estimate_partial_row_group(self, size_test_file):
        """Test estimation for partial row group."""
        pq_file = pq.ParquetFile(size_test_file)

        # Only part of first row group
        estimated_size = _estimate_slice_size(pq_file, 10, 50)

        assert estimated_size > 0

    def test_estimate_returns_int(self, size_test_file):
        """Test that estimation always returns integer."""
        pq_file = pq.ParquetFile(size_test_file)

        estimated_size = _estimate_slice_size(pq_file, 0, 100)

        assert isinstance(estimated_size, int)

    def test_estimate_consistent_for_same_slice(self, size_test_file):
        """Test that estimation is consistent for same slice."""
        pq_file = pq.ParquetFile(size_test_file)

        size1 = _estimate_slice_size(pq_file, 100, 200)
        size2 = _estimate_slice_size(pq_file, 100, 200)

        assert size1 == size2


class TestEstimateSliceSizeWithSmallFile:
    """Test estimation with small file (single row group)."""

    @pytest.fixture
    def small_file(self, tmp_path):
        """Create a small test file with single row group."""
        file_path = tmp_path / "small_test.parquet"

        table = pa.table(
            {
                "a": [1, 2, 3, 4, 5],
                "b": ["x", "y", "z", "w", "v"],
            }
        )

        pq.write_table(table, file_path)
        return str(file_path)

    def test_estimate_small_file_full(self, small_file):
        """Test estimating size of full small file."""
        pq_file = pq.ParquetFile(small_file)

        estimated_size = _estimate_slice_size(pq_file, 0, 5)

        assert estimated_size > 0

    def test_estimate_small_file_partial(self, small_file):
        """Test estimating size of partial small file."""
        pq_file = pq.ParquetFile(small_file)

        full_size = _estimate_slice_size(pq_file, 0, 5)
        partial_size = _estimate_slice_size(pq_file, 0, 3)

        assert partial_size < full_size
        assert partial_size > 0


class TestEstimateSliceSizeEdgeCases:
    """Test edge cases for size estimation."""

    @pytest.fixture
    def edge_case_file(self, tmp_path):
        """Create a file for edge case testing."""
        file_path = tmp_path / "edge_case.parquet"

        # Single column, single value
        table = pa.table({"col": [42]})

        pq.write_table(table, file_path)
        return str(file_path)

    def test_estimate_single_value_file(self, edge_case_file):
        """Test estimation on file with single value."""
        pq_file = pq.ParquetFile(edge_case_file)

        estimated_size = _estimate_slice_size(pq_file, 0, 1)

        assert estimated_size > 0

    def test_estimate_empty_slice_on_valid_file(self, edge_case_file):
        """Test estimation of empty slice on valid file."""
        pq_file = pq.ParquetFile(edge_case_file)

        estimated_size = _estimate_slice_size(pq_file, 0, 0)

        assert estimated_size == 0
