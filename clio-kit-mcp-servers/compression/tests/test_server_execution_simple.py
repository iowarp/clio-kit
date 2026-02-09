import pytest
import os
import tempfile
import sys
from unittest.mock import patch, MagicMock

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))


@pytest.fixture
def sample_file():
    """Create a temporary file with test content"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as f:
        f.write("test content for server execution testing\n" * 50)
    yield f.name
    if os.path.exists(f.name):
        os.unlink(f.name)
    if os.path.exists(f.name + ".gz"):
        os.unlink(f.name + ".gz")


def test_compress_file_tool_definition_in_source():
    """Test that compress_file_tool is properly defined in source code"""
    import ast

    # Read server.py source directly to test the function definition
    server_path = os.path.join(os.path.dirname(__file__), "..", "src", "server.py")
    with open(server_path, "r") as f:
        source = f.read()

    # Parse AST to verify function exists and is async
    tree = ast.parse(source)

    compress_function_found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "compress_file_tool":
            compress_function_found = True
            # Verify it has the expected parameters
            assert len(node.args.args) == 1  # file_path parameter
            assert node.args.args[0].arg == "file_path"
            break

    assert compress_function_found, (
        "compress_file_tool async function not found in source"
    )


def test_main_function_execution_stdio():
    """Test main function execution with stdio transport"""
    with patch.dict("sys.modules", {"fastmcp": MagicMock(), "dotenv": MagicMock()}):
        with patch("logging.basicConfig"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                # Mock the mcp object that would be created
                mock_mcp_instance = MagicMock()

                with patch("server.mcp", mock_mcp_instance):
                    with patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}):
                        with patch("builtins.print") as mock_print:
                            with patch("server.logger") as mock_server_logger:
                                import server  # noqa: F401

                                # Execute main - this will test the actual function execution
                                try:
                                    server.main()
                                except SystemExit:
                                    pass  # Expected behavior

                                # Verify the key logging calls were made
                                expected_calls = [
                                    "Starting Compression MCP Server",
                                    "Starting stdio transport",
                                ]
                                for call_msg in expected_calls:
                                    mock_server_logger.info.assert_any_call(call_msg)

                            # Verify print was called for stderr output
                            mock_print.assert_called()

                            # Verify mcp.run was called
                            mock_mcp_instance.run.assert_called_with(transport="stdio")


def test_main_function_execution_sse():
    """Test main function execution with SSE transport"""
    with patch.dict("sys.modules", {"fastmcp": MagicMock(), "dotenv": MagicMock()}):
        with patch("logging.basicConfig"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                mock_mcp_instance = MagicMock()

                with patch("server.mcp", mock_mcp_instance):
                    with patch.dict(
                        os.environ,
                        {
                            "MCP_TRANSPORT": "sse",
                            "MCP_SSE_HOST": "localhost",
                            "MCP_SSE_PORT": "9000",
                        },
                    ):
                        with patch("builtins.print") as mock_print:
                            with patch("server.logger") as mock_server_logger:
                                import server  # noqa: F401

                                try:
                                    server.main()
                                except SystemExit:
                                    pass

                                # Verify logging calls
                                mock_server_logger.info.assert_any_call(
                                    "Starting Compression MCP Server"
                                )
                                mock_server_logger.info.assert_any_call(
                                    "Starting SSE transport on localhost:9000"
                                )

                            # Verify print was called
                            mock_print.assert_called()

                            # Verify mcp.run was called with SSE parameters
                            mock_mcp_instance.run.assert_called_with(
                                transport="sse", host="localhost", port=9000
                            )


def test_main_function_exception_handling():
    """Test main function exception handling path"""
    with patch.dict("sys.modules", {"fastmcp": MagicMock(), "dotenv": MagicMock()}):
        with patch("logging.basicConfig"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                # Mock mcp.run to raise an exception
                mock_mcp_instance = MagicMock()
                mock_mcp_instance.run.side_effect = Exception("Test server error")

                with patch("server.mcp", mock_mcp_instance):
                    with patch("builtins.print") as mock_print:
                        with patch("sys.exit") as mock_exit:
                            with patch("server.logger") as mock_server_logger:
                                import server  # noqa: F401

                                # Execute main - should catch exception
                                server.main()

                                # Verify error logging
                                mock_server_logger.error.assert_called_with(
                                    "Server error: Test server error"
                                )

                            # Verify error output to stderr
                            mock_print.assert_called()

                            # Verify sys.exit(1) was called
                            mock_exit.assert_called_with(1)


def test_module_imports_and_setup():
    """Test module-level imports and setup code execution"""
    with patch("logging.basicConfig") as mock_basic_config:
        with patch("logging.getLogger") as mock_get_logger:
            mock_logger = MagicMock()
            mock_get_logger.return_value = mock_logger

            with patch.dict(
                "sys.modules", {"fastmcp": MagicMock(), "dotenv": MagicMock()}
            ):
                # Force reimport to trigger module-level code
                if "server" in sys.modules:
                    del sys.modules["server"]

                # Import should trigger all module-level setup
                import server  # noqa: F401

                # Verify logging was configured
                mock_basic_config.assert_called_once_with(
                    level=20,  # logging.INFO
                    format="%(asctime)s - %(levelname)s - %(message)s",
                )

                # Verify logger was obtained
                mock_get_logger.assert_called()


def test_environment_variable_handling():
    """Test various environment variable combinations"""
    with patch.dict("sys.modules", {"fastmcp": MagicMock(), "dotenv": MagicMock()}):
        with patch("logging.basicConfig"):
            with patch("logging.getLogger") as mock_get_logger:
                mock_logger = MagicMock()
                mock_get_logger.return_value = mock_logger

                mock_mcp_instance = MagicMock()

                with patch("server.mcp", mock_mcp_instance):
                    # Test default values (no env vars set)
                    with patch.dict(os.environ, {}, clear=True):
                        with patch("builtins.print"):
                            import server  # noqa: F401

                            try:
                                server.main()
                            except SystemExit:
                                pass

                            # Should default to stdio
                            mock_mcp_instance.run.assert_called_with(transport="stdio")


def test_fastmcp_initialization():
    """Test FastMCP server initialization"""
    mock_fastmcp_class = MagicMock()
    mock_fastmcp_instance = MagicMock()
    mock_fastmcp_class.return_value = mock_fastmcp_instance

    with patch.dict(
        "sys.modules",
        {"fastmcp": MagicMock(FastMCP=mock_fastmcp_class), "dotenv": MagicMock()},
    ):
        with patch("logging.basicConfig"):
            with patch("logging.getLogger"):
                # Clear module to force fresh import
                if "server" in sys.modules:
                    del sys.modules["server"]

                # Import should create FastMCP instance
                import server  # noqa: F401

                # Verify FastMCP was instantiated with correct name
                mock_fastmcp_class.assert_called_with("CompressionMCP")


def test_dotenv_loading():
    """Test that dotenv.load_dotenv is called during module import"""
    mock_dotenv_module = MagicMock()

    with patch.dict(
        "sys.modules", {"fastmcp": MagicMock(), "dotenv": mock_dotenv_module}
    ):
        with patch("logging.basicConfig"):
            with patch("logging.getLogger"):
                # Clear module to force fresh import
                if "server" in sys.modules:
                    del sys.modules["server"]

                # Import should call load_dotenv
                import server  # noqa: F401

                # Verify load_dotenv was called
                mock_dotenv_module.load_dotenv.assert_called_once()


def test_tool_decorator_presence():
    """Test that the compress_file_tool function has the expected decorator pattern"""
    import ast

    # Read the server.py source directly
    server_path = os.path.join(os.path.dirname(__file__), "..", "src", "server.py")
    with open(server_path, "r") as f:
        source = f.read()

    # Parse the AST to find the decorator
    tree = ast.parse(source)

    # Find the compress_file_tool function and check for decorator
    for node in ast.walk(tree):
        if isinstance(node, ast.AsyncFunctionDef) and node.name == "compress_file_tool":
            # Should have at least one decorator
            assert len(node.decorator_list) > 0

            # The decorator should be a call to mcp.tool
            decorator = node.decorator_list[0]
            assert isinstance(decorator, ast.Call)

            # Verify this covers the @mcp.tool line
            break
    else:
        assert False, "compress_file_tool function not found"
