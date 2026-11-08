"""Tests for handling malformed, corrupted, and invalid files."""

import json
import tempfile
import os
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.fixture
def non_parquet_text_file():
    """Create a temporary text file that is not Parquet."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
        f.write("This is just plain text, not a Parquet file\n")
        f.write("Line 2\nLine 3\n")
        return f.name


@pytest.fixture
def non_parquet_json_file():
    """Create a temporary JSON file that is not Parquet."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump({"key": "value", "number": 42}, f)
        return f.name


@pytest.fixture
def corrupted_parquet_file():
    """Create a temporary corrupted Parquet file (partial/truncated)."""
    with tempfile.NamedTemporaryFile(mode="wb", suffix=".parquet", delete=False) as f:
        # Write some random bytes that start like Parquet but are truncated
        f.write(b"PAR1")  # Parquet magic number
        f.write(b"garbage data that is not valid parquet")
        return f.name


@pytest.fixture
def empty_file():
    """Create an empty file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".parquet", delete=False) as f:
        return f.name  # Leave empty


@pytest.mark.asyncio
async def test_summarize_text_file(non_parquet_text_file):
    """Test summarize with a text file."""
    result = await summarize(non_parquet_text_file)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"
    assert "message" in data

    # Cleanup
    os.unlink(non_parquet_text_file)


@pytest.mark.asyncio
async def test_summarize_json_file(non_parquet_json_file):
    """Test summarize with a JSON file."""
    result = await summarize(non_parquet_json_file)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"
    assert "message" in data

    # Cleanup
    os.unlink(non_parquet_json_file)


@pytest.mark.asyncio
async def test_summarize_corrupted_file(corrupted_parquet_file):
    """Test summarize with a corrupted Parquet file."""
    result = await summarize(corrupted_parquet_file)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"
    assert "message" in data

    # Cleanup
    os.unlink(corrupted_parquet_file)


@pytest.mark.asyncio
async def test_summarize_empty_file(empty_file):
    """Test summarize with an empty file."""
    result = await summarize(empty_file)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"
    assert "message" in data

    # Cleanup
    os.unlink(empty_file)


@pytest.mark.asyncio
async def test_read_slice_text_file(non_parquet_text_file):
    """Test read_slice with a text file."""
    result = await read_slice(non_parquet_text_file, start_row=0, end_row=5)
    data = json.loads(result)

    # Should return error - text file is not a valid Parquet file
    assert data["status"] == "error"
    assert (
        "magic bytes" in data["message"].lower() or "parquet" in data["message"].lower()
    )

    # Cleanup
    os.unlink(non_parquet_text_file)


@pytest.mark.asyncio
async def test_read_slice_json_file(non_parquet_json_file):
    """Test read_slice with a JSON file."""
    result = await read_slice(non_parquet_json_file, start_row=0, end_row=5)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"

    # Cleanup
    os.unlink(non_parquet_json_file)


@pytest.mark.asyncio
async def test_read_slice_corrupted_file(corrupted_parquet_file):
    """Test read_slice with a corrupted Parquet file."""
    result = await read_slice(corrupted_parquet_file, start_row=0, end_row=5)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"

    # Cleanup
    os.unlink(corrupted_parquet_file)


@pytest.mark.asyncio
async def test_column_preview_text_file(non_parquet_text_file):
    """Test get_column_preview with a text file."""
    result = await get_column_preview(non_parquet_text_file, "some_column")
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"

    # Cleanup
    os.unlink(non_parquet_text_file)


@pytest.mark.asyncio
async def test_summarize_directory_path(test_parquet_file):
    """Test summarize with a directory path instead of file."""
    # Use parent directory of test file
    dir_path = os.path.dirname(test_parquet_file)

    result = await summarize(dir_path)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_read_slice_directory_path(test_parquet_file):
    """Test read_slice with a directory path instead of file."""
    dir_path = os.path.dirname(test_parquet_file)

    result = await read_slice(dir_path, start_row=0, end_row=5)
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_column_preview_directory_path(test_parquet_file):
    """Test get_column_preview with a directory path instead of file."""
    dir_path = os.path.dirname(test_parquet_file)

    result = await get_column_preview(dir_path, "some_column")
    data = json.loads(result)

    # Should return error
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_summarize_symlink_to_valid_file(test_parquet_file):
    """Test summarize with a symlink to a valid file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        symlink_path = os.path.join(tmpdir, "symlink.parquet")

        # Create symlink
        try:
            os.symlink(test_parquet_file, symlink_path)

            result = await summarize(symlink_path)
            data = json.loads(result)

            # Should succeed (following symlink)
            if os.path.exists(symlink_path):
                assert data["status"] == "success"
        except OSError:
            # Symlinks might not be supported on all systems
            pytest.skip("Symlinks not supported on this system")


@pytest.mark.asyncio
async def test_summarize_broken_symlink(test_parquet_file):
    """Test summarize with a broken symlink."""
    with tempfile.TemporaryDirectory() as tmpdir:
        symlink_path = os.path.join(tmpdir, "broken_symlink.parquet")

        try:
            # Create symlink to non-existent file
            os.symlink("/nonexistent/file.parquet", symlink_path)

            result = await summarize(symlink_path)
            data = json.loads(result)

            # Should error
            assert data["status"] == "error"
        except OSError:
            # Symlinks might not be supported
            pytest.skip("Symlinks not supported on this system")
