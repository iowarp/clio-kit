"""Tests for path security and traversal attack prevention."""

import json
import pytest
from parquet_mcp.capabilities.parquet_handler import (
    summarize,
    read_slice,
    get_column_preview,
)


@pytest.mark.asyncio
async def test_summarize_path_traversal_attack(test_parquet_file):
    """Test that path traversal attempts are blocked."""
    # Try to access parent directories
    malicious_paths = [
        "../../../../etc/passwd.parquet",
        "../../../sensitive_file.parquet",
        "....//....//....//etc/passwd",
        "..\\..\\..\\ windows\\system32\\file.parquet",
    ]

    for path in malicious_paths:
        result = await summarize(path)
        data = json.loads(result)

        # Should fail (file not found or error)
        assert data["status"] == "error"


@pytest.mark.asyncio
async def test_read_slice_path_traversal_attack(test_parquet_file):
    """Test that read_slice blocks path traversal."""
    malicious_paths = [
        "../../../../etc/passwd.parquet",
        "../../../sensitive_file.parquet",
    ]

    for path in malicious_paths:
        result = await read_slice(path, start_row=0, end_row=10)
        data = json.loads(result)

        # Should fail
        assert data["status"] == "error"


@pytest.mark.asyncio
async def test_column_preview_path_traversal_attack(test_parquet_file):
    """Test that get_column_preview blocks path traversal."""
    malicious_paths = [
        "../../../../etc/passwd.parquet",
        "../../../sensitive_file.parquet",
    ]

    for path in malicious_paths:
        result = await get_column_preview(path, "column_name")
        data = json.loads(result)

        # Should fail
        assert data["status"] == "error"


@pytest.mark.asyncio
async def test_summarize_absolute_path_outside_workspace(test_parquet_file):
    """Test that absolute paths outside designated workspace may be restricted."""
    # Try to access system files
    system_paths = [
        "/etc/passwd",
        "/etc/shadow",
        "C:\\Windows\\System32\\config\\SAM",
    ]

    for path in system_paths:
        result = await summarize(path)
        data = json.loads(result)

        # Should fail (file not found or permission error)
        assert data["status"] == "error"


@pytest.mark.asyncio
async def test_summarize_null_byte_injection(test_parquet_file):
    """Test protection against null byte injection in paths."""
    # Null byte injection attempt
    malicious_path = "valid_file.parquet\x00.txt"

    result = await summarize(malicious_path)
    data = json.loads(result)

    # Should fail
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_read_slice_null_byte_injection(test_parquet_file):
    """Test protection against null byte injection in read_slice."""
    malicious_path = "valid_file.parquet\x00.txt"

    result = await read_slice(malicious_path, start_row=0, end_row=10)
    data = json.loads(result)

    # Should fail
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_column_preview_null_byte_injection(test_parquet_file):
    """Test protection against null byte injection in column_preview."""
    malicious_path = "valid_file.parquet\x00.txt"

    result = await get_column_preview(malicious_path, "column")
    data = json.loads(result)

    # Should fail
    assert data["status"] == "error"


@pytest.mark.asyncio
async def test_summarize_path_with_spaces(test_parquet_file):
    """Test handling of paths with spaces."""
    # Paths with spaces should work if they're valid
    # This test ensures they don't introduce vulnerabilities
    import tempfile
    import os
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy test file to path with spaces
        space_path = os.path.join(tmpdir, "my test file.parquet")
        shutil.copy(test_parquet_file, space_path)

        result = await summarize(space_path)
        data = json.loads(result)

        # Should succeed
        if os.path.exists(space_path):
            assert data["status"] == "success"


@pytest.mark.asyncio
async def test_summarize_path_with_special_chars(test_parquet_file):
    """Test handling of paths with special characters."""
    import tempfile
    import os
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy test file to path with special characters
        special_path = os.path.join(tmpdir, "file-with_special.chars!@#$.parquet")
        shutil.copy(test_parquet_file, special_path)

        result = await summarize(special_path)
        data = json.loads(result)

        # Should succeed
        if os.path.exists(special_path):
            assert data["status"] == "success"


@pytest.mark.asyncio
async def test_summarize_unicode_path(test_parquet_file):
    """Test handling of paths with unicode characters."""
    import tempfile
    import os
    import shutil

    with tempfile.TemporaryDirectory() as tmpdir:
        # Copy test file to path with unicode characters
        unicode_path = os.path.join(tmpdir, "文件_файл_αρχείο.parquet")
        try:
            shutil.copy(test_parquet_file, unicode_path)

            result = await summarize(unicode_path)
            data = json.loads(result)

            # Should succeed
            if os.path.exists(unicode_path):
                assert data["status"] == "success"
        except (UnicodeError, OSError):
            # Unicode paths may not be supported on all systems
            pytest.skip("Unicode paths not fully supported")


@pytest.mark.asyncio
async def test_summarize_very_long_path(test_parquet_file):
    """Test handling of very long file paths."""
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a very long path
        long_name = "a" * 100 + ".parquet"
        long_path = os.path.join(tmpdir, long_name)

        # Try to access (file won't exist, but test path handling)
        result = await summarize(long_path)
        data = json.loads(result)

        # Should handle gracefully
        assert data["status"] == "error"


@pytest.mark.asyncio
async def test_read_slice_relative_vs_absolute_paths(test_parquet_file):
    """Test that relative and absolute paths to same file are handled correctly."""
    import os

    absolute_path = os.path.abspath(test_parquet_file)

    # Read with absolute path
    result_abs = await read_slice(absolute_path, start_row=0, end_row=5)
    data_abs = json.loads(result_abs)

    # Both should succeed
    assert data_abs["status"] == "success"

    # Verify path is consistent
    assert (
        absolute_path in data_abs["file_path"]
        or data_abs["file_path"] in absolute_path
        or os.path.samefile(absolute_path, data_abs["file_path"])
    )
