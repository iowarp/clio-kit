"""Pytest configuration and shared fixtures."""

import pytest
import os


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "requires_train_meta: mark test as requiring train_meta.parquet dataset",
    )
    config.addinivalue_line(
        "markers",
        "requires_batch_large_strings: mark test as requiring batch_large_strings.parquet dataset",
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests that require missing datasets."""
    datasets_dir = os.path.join(os.path.dirname(__file__), "..", "datasets")

    # Check which datasets exist
    train_meta_exists = os.path.exists(os.path.join(datasets_dir, "train_meta.parquet"))
    batch_large_strings_exists = os.path.exists(
        os.path.join(datasets_dir, "batch_large_strings.parquet")
    )

    skip_train_meta = pytest.mark.skip(
        reason="train_meta.parquet dataset not available"
    )
    skip_batch_large_strings = pytest.mark.skip(
        reason="batch_large_strings.parquet dataset not available"
    )

    for item in items:
        # Check if test references train_meta.parquet in its code
        if hasattr(item, "function"):
            import inspect

            source = inspect.getsource(item.function)
            if "train_meta.parquet" in source and not train_meta_exists:
                item.add_marker(skip_train_meta)
            if (
                "batch_large_strings.parquet" in source
                and not batch_large_strings_exists
            ):
                item.add_marker(skip_batch_large_strings)


@pytest.fixture(scope="session")
def test_parquet_file():
    """Return path to a test Parquet file."""
    # Parquet files are now stored in datasets/ folder
    batch_file = os.path.join(
        os.path.dirname(__file__), "..", "datasets", "batch_1.parquet"
    )

    if not os.path.exists(batch_file):
        pytest.skip(f"batch_1.parquet not found at {batch_file}")

    return batch_file


@pytest.fixture
def nonexistent_file():
    """Return path to a file that doesn't exist."""
    return "/nonexistent/path/to/file.parquet"
