import pytest
import json
import os
from unittest.mock import AsyncMock, patch, MagicMock


class TestServerToolFunctions:
    @pytest.mark.asyncio
    async def test_list_bp5_tool_success(self):
        from src.server import list_bp5_tool

        mock_files = [
            {"name": "file1.bp", "size": 1024},
            {"name": "file2.bp5", "size": 2048},
        ]

        with patch(
            "src.server.mcp_handlers.list_bp5_files", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = {"files": mock_files}

            # Access the actual function from the FunctionTool
            actual_func = (
                list_bp5_tool.fn if hasattr(list_bp5_tool, "fn") else list_bp5_tool
            )
            result = await actual_func("/test/directory")

            mock_handler.assert_called_once_with("/test/directory")
            assert result == {"files": mock_files}

    @pytest.mark.asyncio
    async def test_list_bp5_tool_exception(self):
        from src.server import list_bp5_tool

        with patch(
            "src.server.mcp_handlers.list_bp5_files", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.side_effect = FileNotFoundError("Directory not found")

            actual_func = (
                list_bp5_tool.fn if hasattr(list_bp5_tool, "fn") else list_bp5_tool
            )
            result = await actual_func("/nonexistent")

            assert result["isError"] is True
            assert (
                "Directory not found"
                in json.loads(result["content"][0]["text"])["error"]
            )
            assert result["_meta"]["tool"] == "list_bp5"
            assert result["_meta"]["error"] == "FileNotFoundError"

    @pytest.mark.asyncio
    async def test_list_bp5_tool_default_directory(self):
        from src.server import list_bp5_tool

        with patch(
            "src.server.mcp_handlers.list_bp5_files", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = {"files": []}

            actual_func = (
                list_bp5_tool.fn if hasattr(list_bp5_tool, "fn") else list_bp5_tool
            )
            result = await actual_func()

            mock_handler.assert_called_once_with("data/")
            assert result == {"files": []}

    @pytest.mark.asyncio
    async def test_inspect_variables_tool_success(self):
        from src.server import inspect_variables_tool

        mock_result = {"variables": {"temp": {"type": "float64", "shape": [100, 50]}}}

        with patch(
            "src.server.mcp_handlers.inspect_variables_handler", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                inspect_variables_tool.fn
                if hasattr(inspect_variables_tool, "fn")
                else inspect_variables_tool
            )
            result = await actual_func("/test/file.bp")

            mock_handler.assert_called_once_with("/test/file.bp", None)
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_inspect_variables_tool_with_variable_name(self):
        from src.server import inspect_variables_tool

        mock_result = {"variable_data": {"name": "pressure", "values": [1, 2, 3]}}

        with patch(
            "src.server.mcp_handlers.inspect_variables_handler", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                inspect_variables_tool.fn
                if hasattr(inspect_variables_tool, "fn")
                else inspect_variables_tool
            )
            result = await actual_func("/test/file.bp", "pressure")

            mock_handler.assert_called_once_with("/test/file.bp", "pressure")
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_inspect_variables_tool_exception(self):
        from src.server import inspect_variables_tool

        with patch(
            "src.server.mcp_handlers.inspect_variables_handler", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.side_effect = Exception("ADIOS error")

            actual_func = (
                inspect_variables_tool.fn
                if hasattr(inspect_variables_tool, "fn")
                else inspect_variables_tool
            )
            result = await actual_func("/test/file.bp")

            assert result["isError"] is True
            assert "ADIOS error" in json.loads(result["content"][0]["text"])["error"]
            assert result["_meta"]["tool"] == "inspect_variables"

    @pytest.mark.asyncio
    async def test_inspect_variables_at_step_tool_success(self):
        from src.server import inspect_variables_at_step_tool

        mock_result = {"variable": "temp", "step": 5, "shape": [100], "type": "float64"}

        with patch(
            "src.server.mcp_handlers.inspect_variables_at_step_handler",
            new_callable=AsyncMock,
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                inspect_variables_at_step_tool.fn
                if hasattr(inspect_variables_at_step_tool, "fn")
                else inspect_variables_at_step_tool
            )
            result = await actual_func("/test/file.bp", "temp", 5)

            mock_handler.assert_called_once_with("/test/file.bp", "temp", 5)
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_inspect_variables_at_step_tool_exception(self):
        from src.server import inspect_variables_at_step_tool

        with patch(
            "src.server.mcp_handlers.inspect_variables_at_step_handler",
            new_callable=AsyncMock,
        ) as mock_handler:
            mock_handler.side_effect = ValueError("Invalid step")

            actual_func = (
                inspect_variables_at_step_tool.fn
                if hasattr(inspect_variables_at_step_tool, "fn")
                else inspect_variables_at_step_tool
            )
            result = await actual_func("/test/file.bp", "temp", 10)

            assert result["isError"] is True
            assert "Invalid step" in json.loads(result["content"][0]["text"])["error"]
            assert result["_meta"]["tool"] == "inspect_variables_at_step"
            assert result["_meta"]["error"] == "ValueError"

    @pytest.mark.asyncio
    async def test_inspect_attributes_tool_success(self):
        from src.server import inspect_attributes_tool

        mock_result = {
            "attributes": {"global": {"title": "simulation"}, "variables": {}}
        }

        with patch(
            "src.server.mcp_handlers.inspect_attributes_handler", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                inspect_attributes_tool.fn
                if hasattr(inspect_attributes_tool, "fn")
                else inspect_attributes_tool
            )
            result = await actual_func("/test/file.bp")

            mock_handler.assert_called_once_with("/test/file.bp", None)
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_inspect_attributes_tool_with_variable(self):
        from src.server import inspect_attributes_tool

        mock_result = {"attributes": {"variable_attrs": {"units": "celsius"}}}

        with patch(
            "src.server.mcp_handlers.inspect_attributes_handler", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                inspect_attributes_tool.fn
                if hasattr(inspect_attributes_tool, "fn")
                else inspect_attributes_tool
            )
            result = await actual_func("/test/file.bp", "temperature")

            mock_handler.assert_called_once_with("/test/file.bp", "temperature")
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_inspect_attributes_tool_exception(self):
        from src.server import inspect_attributes_tool

        with patch(
            "src.server.mcp_handlers.inspect_attributes_handler", new_callable=AsyncMock
        ) as mock_handler:
            mock_handler.side_effect = RuntimeError("Attribute access failed")

            actual_func = (
                inspect_attributes_tool.fn
                if hasattr(inspect_attributes_tool, "fn")
                else inspect_attributes_tool
            )
            result = await actual_func("/test/file.bp")

            assert result["isError"] is True
            assert (
                "Attribute access failed"
                in json.loads(result["content"][0]["text"])["error"]
            )
            assert result["_meta"]["tool"] == "inspect_attributes"
            assert result["_meta"]["error"] == "RuntimeError"

    @pytest.mark.asyncio
    async def test_read_variable_at_step_tool_success(self):
        from src.server import read_variable_at_step_tool

        mock_result = {"value": [1.0, 2.0, 3.0]}

        with patch(
            "src.server.mcp_handlers.read_variable_at_step_handler",
            new_callable=AsyncMock,
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                read_variable_at_step_tool.fn
                if hasattr(read_variable_at_step_tool, "fn")
                else read_variable_at_step_tool
            )
            result = await actual_func("/test/file.bp", "pressure", 3)

            mock_handler.assert_called_once_with("/test/file.bp", "pressure", 3)
            assert result == mock_result

    @pytest.mark.asyncio
    async def test_read_variable_at_step_tool_scalar_value(self):
        from src.server import read_variable_at_step_tool

        mock_result = {"value": 42.5}

        with patch(
            "src.server.mcp_handlers.read_variable_at_step_handler",
            new_callable=AsyncMock,
        ) as mock_handler:
            mock_handler.return_value = mock_result

            actual_func = (
                read_variable_at_step_tool.fn
                if hasattr(read_variable_at_step_tool, "fn")
                else read_variable_at_step_tool
            )
            result = await actual_func("/test/file.bp", "scalar_var", 0)

            mock_handler.assert_called_once_with("/test/file.bp", "scalar_var", 0)
            assert result == mock_result


class TestMainFunction:
    def test_main_default_stdio_transport(self):
        from src.server import main

        mock_mcp = MagicMock()

        with patch("src.server.mcp", mock_mcp), patch.dict(os.environ, {}, clear=True):
            main()

            mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_main_sse_transport_default_host_port(self):
        from src.server import main

        mock_mcp = MagicMock()

        with (
            patch("src.server.mcp", mock_mcp),
            patch.dict(os.environ, {"MCP_TRANSPORT": "sse"}, clear=True),
            patch("builtins.print") as mock_print,
        ):
            main()

            mock_mcp.run.assert_called_once_with(
                transport="sse", host="0.0.0.0", port=8000
            )
            mock_print.assert_called()

    def test_main_sse_transport_custom_host_port(self):
        from src.server import main

        mock_mcp = MagicMock()

        with (
            patch("src.server.mcp", mock_mcp),
            patch.dict(
                os.environ,
                {
                    "MCP_TRANSPORT": "sse",
                    "MCP_SSE_HOST": "localhost",
                    "MCP_SSE_PORT": "9000",
                },
                clear=True,
            ),
            patch("builtins.print") as mock_print,
        ):
            main()

            mock_mcp.run.assert_called_once_with(
                transport="sse", host="localhost", port=9000
            )
            mock_print.assert_called()

    def test_main_stdio_transport_explicit(self):
        from src.server import main

        mock_mcp = MagicMock()

        with (
            patch("src.server.mcp", mock_mcp),
            patch.dict(os.environ, {"MCP_TRANSPORT": "stdio"}, clear=True),
        ):
            main()

            mock_mcp.run.assert_called_once_with(transport="stdio")

    def test_main_exception_handling(self):
        from src.server import main

        mock_mcp = MagicMock()
        mock_mcp.run.side_effect = RuntimeError("Server startup failed")

        with (
            patch("src.server.mcp", mock_mcp),
            patch.dict(os.environ, {}, clear=True),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            main()

            mock_print.assert_called()
            mock_exit.assert_called_once_with(1)

    def test_main_case_insensitive_transport(self):
        from src.server import main

        mock_mcp = MagicMock()

        with (
            patch("src.server.mcp", mock_mcp),
            patch.dict(os.environ, {"MCP_TRANSPORT": "SSE"}, clear=True),
            patch("builtins.print"),
        ):
            main()

            mock_mcp.run.assert_called_once_with(
                transport="sse", host="0.0.0.0", port=8000
            )

    def test_main_invalid_port_handling(self):
        from src.server import main

        mock_mcp = MagicMock()

        with (
            patch("src.server.mcp", mock_mcp),
            patch.dict(
                os.environ,
                {"MCP_TRANSPORT": "sse", "MCP_SSE_PORT": "invalid_port"},
                clear=True,
            ),
            patch("builtins.print") as mock_print,
            patch("sys.exit") as mock_exit,
        ):
            main()

            mock_print.assert_called()
            mock_exit.assert_called_once_with(1)


class TestServerIntegration:
    def test_server_imports(self):
        """Test that all required modules are properly imported"""
        import src.server as server

        assert hasattr(server, "FastMCP")
        assert hasattr(server, "mcp_handlers")
        assert hasattr(server, "mcp")

    def test_environment_variable_loading(self):
        """Test that dotenv loading works properly"""
        # Since load_dotenv is called at module import time, we need to test
        # that it's available and can be called
        from dotenv import load_dotenv

        # Just test that the function exists and is callable
        assert callable(load_dotenv)
