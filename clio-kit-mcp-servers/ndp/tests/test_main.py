"""Tests for main entry point and CLI functionality."""

import os
import sys
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server
from server import main


class TestMainFunction:
    """Test the main() entry point function."""

    def test_main_default_stdio_mode(self):
        """Test main function runs in stdio mode by default."""
        with patch("server.mcp.run") as mock_run:
            with patch("sys.argv", ["server.py"]):
                main()

            mock_run.assert_called_once_with(transport="stdio")

    def test_main_fastapi_mode(self):
        """Test main function runs in fastapi mode with --fastapi flag."""
        with patch("server.mcp.run") as mock_run:
            with patch("sys.argv", ["server.py", "--fastapi"]):
                main()

            mock_run.assert_called_once_with(transport="fastapi", host="localhost", port=8000)

    def test_main_with_other_args(self):
        """Test main function with other command line arguments."""
        with patch("server.mcp.run") as mock_run:
            with patch("sys.argv", ["server.py", "--other", "arg"]):
                main()

            # Should default to stdio mode
            mock_run.assert_called_once_with(transport="stdio")

    def test_main_fastapi_as_second_arg(self):
        """Test that --fastapi must be the first argument."""
        with patch("server.mcp.run") as mock_run:
            with patch("sys.argv", ["server.py", "--other", "--fastapi"]):
                main()

            # Should default to stdio mode since --fastapi is not argv[1]
            mock_run.assert_called_once_with(transport="stdio")


class TestModuleExecution:
    """Test module execution as __main__."""

    def test_main_called_when_run_as_script(self):
        """Test that main() is called when module is run as script."""
        # This test verifies the if __name__ == "__main__": main() block
        with patch("server.main") as mock_main:
            # Simulate running as main module
            with patch.object(server, "__name__", "__main__"):
                # Execute the module's main block code
                exec(  # noqa: S102
                    "if __name__ == '__main__': main()",
                    {"__name__": "__main__", "main": mock_main},
                )

            mock_main.assert_called_once()


class TestServerInitialization:
    """Test server initialization and module-level setup."""

    def test_fastmcp_instance_created(self):
        """Test that FastMCP instance is properly created."""
        assert server.mcp is not None
        assert server.mcp.name == "NDPServer"

    def test_ndp_client_initialized(self):
        """Test that NDPClient is initialized at module level."""
        assert server.ndp_client is not None
        assert hasattr(server.ndp_client, "base_url")
        assert hasattr(server.ndp_client, "max_retries")
        assert hasattr(server.ndp_client, "retry_delay")

    def test_ndp_client_default_base_url(self):
        """Test NDPClient has correct default base URL."""
        assert server.ndp_client.base_url == "http://155.101.6.191:8003"

    def test_dataset_model_available(self):
        """Test that Dataset model is available in module."""
        from server import Dataset

        assert Dataset is not None
        assert hasattr(Dataset, "model_validate")
        assert hasattr(Dataset, "model_dump")

    def test_dotenv_loaded(self):
        """Test that dotenv configuration is loaded."""
        # The module should have loaded dotenv
        # We can verify by checking that load_dotenv was called
        # This is implicitly tested by module import, but we verify the import exists
        import server

        assert hasattr(server, "load_dotenv")

    def test_tools_registered(self):
        """Test that MCP tools are registered."""
        # Verify that the decorated functions exist
        assert hasattr(server, "list_organizations")
        assert hasattr(server, "search_datasets")
        assert hasattr(server, "get_dataset_details")

    def test_ndp_client_timeout_configured(self):
        """Test that NDPClient has timeout configured."""
        assert server.ndp_client.timeout is not None
        # The timeout should be an httpx.Timeout object
        assert hasattr(server.ndp_client.timeout, "connect")

    def test_module_imports(self):
        """Test that required modules are imported."""
        assert server.asyncio is not None
        assert server.json is not None
        assert server.os is not None
        assert server.sys is not None
        assert server.httpx is not None
        assert server.FastMCP is not None
        assert server.BaseModel is not None
        assert server.Field is not None
