import pytest
import os
import tempfile
from compression_mcp.capabilities.compression_base import compress_file, decompress_file


@pytest.fixture
def sample_file():
    # create a temporary file with some content
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content\n" * 100)
    yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)


# test successful compression of a file
@pytest.mark.asyncio
async def test_compress_success(sample_file):
    result = await compress_file(sample_file)
    assert isinstance(result, dict)
    assert result["original_file"] == sample_file
    assert result["original_size"] > 0
    assert result["compressed_size"] > 0
    assert result["compression_ratio"] >= 0
    assert os.path.exists(result["compressed_file"])
    os.unlink(result["compressed_file"])


# test compression of non-existent file
@pytest.mark.asyncio
async def test_compress_nonexistent_file():
    with pytest.raises(FileNotFoundError, match="File not found"):
        await compress_file("nonexistent_file.txt")


# test compression of empty file
@pytest.mark.asyncio
async def test_compress_empty_file():
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("")
    try:
        result = await compress_file(f.name)
        assert isinstance(result, dict)
        assert result["original_size"] == 0
        assert result["compression_ratio"] == 0.0
        assert os.path.exists(result["compressed_file"])
        os.unlink(result["compressed_file"])
    finally:
        os.unlink(f.name)


# test compression with permission error
@pytest.mark.asyncio
async def test_compress_permission_error():
    """Test that PermissionError is properly raised."""
    import unittest.mock as mock

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", return_value=100):
            with mock.patch(
                "builtins.open", side_effect=PermissionError("Permission denied")
            ):
                with pytest.raises(PermissionError, match="Permission denied"):
                    await compress_file("/some/file.txt")


# test compression with generic exception
@pytest.mark.asyncio
async def test_compress_generic_exception():
    """Test that generic exceptions are properly raised."""
    import unittest.mock as mock

    with mock.patch("os.path.exists", return_value=True):
        with mock.patch("os.path.getsize", side_effect=RuntimeError("Disk error")):
            with pytest.raises(RuntimeError, match="Disk error"):
                await compress_file("/some/file.txt")


# --- DECOMPRESSION TESTS ---


@pytest.mark.asyncio
async def test_decompress_success(sample_file):
    """Test successful compress then decompress round-trip."""
    result = await compress_file(sample_file)
    gz_path = result["compressed_file"]
    os.unlink(sample_file)  # an existing output is never overwritten

    decomp_result = await decompress_file(gz_path)
    assert isinstance(decomp_result, dict)
    assert decomp_result["compressed_file"] == gz_path
    assert decomp_result["decompressed_size"] > 0
    assert os.path.exists(decomp_result["decompressed_file"])
    os.unlink(decomp_result["decompressed_file"])
    os.unlink(gz_path)


@pytest.mark.asyncio
async def test_decompress_nonexistent_file():
    """Test decompression of non-existent file."""
    with pytest.raises(FileNotFoundError, match="File not found"):
        await decompress_file("nonexistent_file.gz")


@pytest.mark.asyncio
async def test_decompress_non_gz_file(sample_file):
    """Test decompression of a file without .gz extension."""
    with pytest.raises(ValueError, match="Not a gzip file"):
        await decompress_file(sample_file)


# --- OUTPUT SAFETY (never overwrite, no partial output) ---


@pytest.mark.asyncio
async def test_existing_outputs_are_not_overwritten(tmp_path):
    source = tmp_path / "c.txt"
    source.write_bytes(b"payload\n" * 50)
    await compress_file(str(source))
    compressed = (tmp_path / "c.txt.gz").read_bytes()

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        await compress_file(str(source))
    assert (tmp_path / "c.txt.gz").read_bytes() == compressed

    source.write_bytes(b"KEEPME")
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        await decompress_file(str(tmp_path / "c.txt.gz"))
    assert source.read_bytes() == b"KEEPME"

    source.unlink()
    await decompress_file(str(tmp_path / "c.txt.gz"))
    assert source.read_bytes() == b"payload\n" * 50


@pytest.mark.asyncio
async def test_failed_decompress_leaves_no_partial_output(tmp_path):
    bad = tmp_path / "notgz.gz"
    bad.write_bytes(b"this is not gzip")

    with pytest.raises(OSError, match="Not a gzipped file"):
        await decompress_file(str(bad))
    assert not (tmp_path / "notgz").exists()
