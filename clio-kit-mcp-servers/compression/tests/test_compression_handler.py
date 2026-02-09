import pytest
import os
import tempfile
from capabilities.compression_base import compress_file


@pytest.fixture
def sample_file():
    # create a temporary file with some content
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content\n" * 100)
    yield f.name
    os.unlink(f.name)


# test successful compression of a file
@pytest.mark.asyncio
async def test_compress_success(sample_file):
    result = await compress_file(sample_file)
    assert isinstance(result, dict)
    assert not result["isError"]
    assert result["_meta"]["tool"] == "compress_file"
    assert os.path.exists(result["_meta"]["compressed_file"])
    os.unlink(result["_meta"]["compressed_file"])


# test compression of non-existent file
@pytest.mark.asyncio
async def test_compress_nonexistent_file():
    with pytest.raises(Exception) as exc_info:
        await compress_file("nonexistent_file.txt")
    assert "File not found" in str(exc_info.value)


# test compression of empty file
@pytest.mark.asyncio
async def test_compress_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("")
    try:
        result = await compress_file(f.name)
        assert isinstance(result, dict)
        assert not result["isError"]
        assert result["_meta"]["tool"] == "compress_file"
        # Empty file should still compress successfully
        assert os.path.exists(result["_meta"]["compressed_file"])
        os.unlink(result["_meta"]["compressed_file"])
    finally:
        os.unlink(f.name)


# test compression with permission error
@pytest.mark.asyncio
async def test_compress_permission_error():
    """Test that PermissionError is properly caught and raised"""
    import unittest.mock as mock

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                with pytest.raises(Exception) as exc_info:
                    await compress_file("/some/file.txt")
                assert "Permission denied" in str(exc_info.value)


# test compression with generic exception
@pytest.mark.asyncio
async def test_compress_generic_exception():
    """Test that generic exceptions are properly caught and raised"""
    import unittest.mock as mock

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", side_effect=RuntimeError("Disk error")):
            with pytest.raises(Exception) as exc_info:
                await compress_file("/some/file.txt")
            assert "Compression failed" in str(exc_info.value) or "Disk error" in str(
                exc_info.value
            )
