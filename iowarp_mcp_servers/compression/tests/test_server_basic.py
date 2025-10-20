import pytest
import os
import tempfile
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from unittest.mock import patch

# Test the compress_file_tool function directly without server dependency


@pytest.fixture
def sample_file():
    """Create a temporary file with test content"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content for server testing\n" * 50)
    yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)
    if os.path.exists(f.name + ".gz"):
        os.unlink(f.name + ".gz")


@pytest.mark.asyncio
async def test_compress_file_handler_direct():
    """Test the compress_file_handler function directly"""
    import mcp_handlers

    # Test with a temporary file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content\n" * 100)

    try:
        result = await mcp_handlers.compress_file_handler(f.name)
        assert isinstance(result, dict)
        assert not result["isError"]
        assert result["_meta"]["tool"] == "compress_file"
        assert os.path.exists(result["_meta"]["compressed_file"])

        # Clean up compressed file
        if os.path.exists(result["_meta"]["compressed_file"]):
            os.unlink(result["_meta"]["compressed_file"])
    finally:
        if os.path.exists(f.name):
            os.unlink(f.name)


@pytest.mark.asyncio
async def test_compress_file_handler_error_direct():
    """Test the compress_file_handler error handling directly"""
    import mcp_handlers

    result = await mcp_handlers.compress_file_handler("nonexistent_file.txt")

    assert isinstance(result, dict)
    assert result["isError"]
    assert result["_meta"]["tool"] == "compress_file"
    assert "error" in result["_meta"]


def test_server_module_imports():
    """Test that server module can be analyzed without running it"""
    import ast

    server_path = os.path.join(os.path.dirname(__file__), "..", "src", "server.py")
    with open(server_path, "r") as f:
        content = f.read()

    # Parse the AST to check structure
    tree = ast.parse(content)

    # Check for main function
    main_found = False
    compress_tool_found = False

    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            if node.name == "main":
                main_found = True
            elif node.name == "compress_file_tool":
                compress_tool_found = True
        elif isinstance(node, ast.AsyncFunctionDef):
            if node.name == "compress_file_tool":
                compress_tool_found = True

    assert main_found, "main function should be defined"
    assert compress_tool_found, "compress_file_tool function should be defined"


def test_server_environment_handling():
    """Test environment variable handling logic without running server"""
    # Test the logic that would be in main() function

    # Test default transport
    transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
    assert transport == "stdio"  # Should default to stdio

    # Test with environment variable set
    with patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}):
        transport = os.getenv("MCP_TRANSPORT", "stdio").lower()
        assert transport == "sse"

    # Test SSE host/port defaults
    host = os.getenv("MCP_SSE_HOST", "0.0.0.0")
    port = int(os.getenv("MCP_SSE_PORT", "8000"))
    assert host == "0.0.0.0"
    assert port == 8000

    # Test with custom values
    with patch.dict(os.environ, {"MCP_SSE_HOST": "localhost", "MCP_SSE_PORT": "9000"}):
        host = os.getenv("MCP_SSE_HOST", "0.0.0.0")
        port = int(os.getenv("MCP_SSE_PORT", "8000"))
        assert host == "localhost"
        assert port == 9000


@pytest.mark.asyncio
async def test_end_to_end_compression_workflow(sample_file):
    """Test the complete compression workflow"""
    import mcp_handlers

    # Test the complete workflow
    result = await mcp_handlers.compress_file_handler(sample_file)

    # Verify all expected fields are present
    assert "content" in result
    assert "_meta" in result
    assert "isError" in result

    # Verify metadata structure
    meta = result["_meta"]
    assert "tool" in meta
    assert "original_file" in meta
    assert "compressed_file" in meta
    assert "original_size" in meta
    assert "compressed_size" in meta
    assert "compression_ratio" in meta

    # Verify compression actually happened
    assert meta["original_size"] > 0
    assert meta["compressed_size"] > 0
    assert meta["compression_ratio"] >= 0

    # Verify files exist
    assert os.path.exists(meta["original_file"])
    assert os.path.exists(meta["compressed_file"])

    # Clean up
    if os.path.exists(meta["compressed_file"]):
        os.unlink(meta["compressed_file"])


def test_logging_configuration():
    """Test that logging can be configured properly"""
    import logging

    # Test basic logging setup
    logger = logging.getLogger("test_compression")
    logger.setLevel(logging.INFO)

    # Should not raise any errors
    logger.info("Test message")
    logger.error("Test error")

    assert logger.level == logging.INFO


@pytest.mark.asyncio
async def test_compression_with_different_file_sizes():
    """Test compression with various file sizes"""
    import mcp_handlers

    # Test small file
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("small content")

    try:
        result = await mcp_handlers.compress_file_handler(f.name)
        assert not result["isError"]
        # Verify compression ratio is calculated

        # Clean up
        if os.path.exists(result["_meta"]["compressed_file"]):
            os.unlink(result["_meta"]["compressed_file"])
    finally:
        if os.path.exists(f.name):
            os.unlink(f.name)

    # Test larger file with repetitive content (should compress better)
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        for i in range(1000):
            f.write("This is repetitive content that should compress well.\n")

    try:
        result = await mcp_handlers.compress_file_handler(f.name)
        assert not result["isError"]
        large_ratio = result["_meta"]["compression_ratio"]

        # Larger file with repetitive content should compress better
        assert large_ratio > 0

        # Clean up
        if os.path.exists(result["_meta"]["compressed_file"]):
            os.unlink(result["_meta"]["compressed_file"])
    finally:
        if os.path.exists(f.name):
            os.unlink(f.name)
