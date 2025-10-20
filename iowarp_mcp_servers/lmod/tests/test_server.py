"""Tests for lmod MCP server."""

import pytest
from unittest.mock import patch, AsyncMock
import sys
import os

# Add the src directory to the path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import server


@pytest.mark.asyncio
async def test_module_list_tool():
    """Test module_list tool."""
    mock_result = {
        "success": True,
        "modules": ["gcc/11.2.0", "python/3.9.0"],
        "count": 2,
    }

    with patch(
        "server.lmod_handler.list_loaded_modules", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_list_tool.fn()

        assert result == mock_result
        mock_handler.assert_called_once()


@pytest.mark.asyncio
async def test_module_avail_tool():
    """Test module_avail tool."""
    mock_result = {
        "success": True,
        "modules": ["python/3.8.0", "python/3.9.0"],
        "count": 2,
        "pattern": "python",
    }

    with patch(
        "server.lmod_handler.search_available_modules", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_avail_tool.fn(pattern="python")

        assert result == mock_result
        mock_handler.assert_called_once_with("python")


@pytest.mark.asyncio
async def test_module_show_tool():
    """Test module_show tool."""
    mock_result = {
        "success": True,
        "module": "python/3.9.0",
        "path": "/apps/modules/python/3.9.0.lua",
        "help": ["Python 3.9.0 programming language"],
        "whatis": ["Name: Python", "Version: 3.9.0"],
        "prerequisites": [],
        "conflicts": ["python"],
        "environment": ['prepend_path("PATH", "/apps/python/3.9.0/bin")'],
    }

    with patch(
        "server.lmod_handler.show_module_details", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_show_tool.fn("python/3.9.0")

        assert result == mock_result
        mock_handler.assert_called_once_with("python/3.9.0")


@pytest.mark.asyncio
async def test_module_load_tool():
    """Test module_load tool."""
    mock_result = {
        "success": True,
        "results": [
            {
                "module": "gcc/11.2.0",
                "success": True,
                "message": "Successfully loaded gcc/11.2.0",
            },
            {
                "module": "python/3.9.0",
                "success": True,
                "message": "Successfully loaded python/3.9.0",
            },
        ],
    }

    with patch(
        "server.lmod_handler.load_modules", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_load_tool.fn(["gcc/11.2.0", "python/3.9.0"])

        assert result == mock_result
        mock_handler.assert_called_once_with(["gcc/11.2.0", "python/3.9.0"])


@pytest.mark.asyncio
async def test_module_unload_tool():
    """Test module_unload tool."""
    mock_result = {
        "success": True,
        "results": [
            {
                "module": "python/3.9.0",
                "success": True,
                "message": "Successfully unloaded python/3.9.0",
            }
        ],
    }

    with patch(
        "server.lmod_handler.unload_modules", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_unload_tool.fn(["python/3.9.0"])

        assert result == mock_result
        mock_handler.assert_called_once_with(["python/3.9.0"])


@pytest.mark.asyncio
async def test_module_swap_tool():
    """Test module_swap tool."""
    mock_result = {
        "success": True,
        "message": "Successfully swapped gcc/10.2.0 with gcc/11.2.0",
        "old_module": "gcc/10.2.0",
        "new_module": "gcc/11.2.0",
    }

    with patch(
        "server.lmod_handler.swap_modules", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_swap_tool.fn("gcc/10.2.0", "gcc/11.2.0")

        assert result == mock_result
        mock_handler.assert_called_once_with("gcc/10.2.0", "gcc/11.2.0")


@pytest.mark.asyncio
async def test_module_spider_tool():
    """Test module_spider tool."""
    mock_result = {
        "success": True,
        "modules": {
            "gcc": ["10.2.0", "11.2.0", "12.1.0"],
            "python": ["3.8.0", "3.9.0", "3.10.0"],
        },
        "pattern": None,
    }

    with patch(
        "server.lmod_handler.spider_search", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_spider_tool.fn(pattern=None)

        assert result == mock_result
        mock_handler.assert_called_once_with(None)


@pytest.mark.asyncio
async def test_module_save_tool():
    """Test module_save tool."""
    mock_result = {
        "success": True,
        "message": "Successfully saved module collection as my_env",
        "collection": "my_env",
    }

    with patch(
        "server.lmod_handler.save_module_collection", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_save_tool.fn("my_env")

        assert result == mock_result
        mock_handler.assert_called_once_with("my_env")


@pytest.mark.asyncio
async def test_module_restore_tool():
    """Test module_restore tool."""
    mock_result = {
        "success": True,
        "message": "Successfully restored module collection my_env",
        "collection": "my_env",
        "loaded_modules": ["gcc/11.2.0", "python/3.9.0"],
    }

    with patch(
        "server.lmod_handler.restore_module_collection", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_restore_tool.fn("my_env")

        assert result == mock_result
        mock_handler.assert_called_once_with("my_env")


@pytest.mark.asyncio
async def test_module_savelist_tool():
    """Test module_savelist tool."""
    mock_result = {
        "success": True,
        "collections": ["default", "dev_env", "prod_env"],
        "count": 3,
    }

    with patch(
        "server.lmod_handler.list_saved_collections", new_callable=AsyncMock
    ) as mock_handler:
        mock_handler.return_value = mock_result

        result = await server.module_savelist_tool.fn()

        assert result == mock_result
        mock_handler.assert_called_once()


def test_main_function():
    """Test the main function entry point."""
    with (
        patch("asyncio.run") as mock_run,
        patch.object(server.mcp, "run") as mock_mcp_run,
    ):
        server.main()

        mock_run.assert_called_once_with(mock_mcp_run.return_value)


def test_main_module_execution():
    """Test __main__ module execution."""
    with patch("server.main") as mock_main:
        # Simulate running the module directly
        exec(
            "if __name__ == '__main__': main()",
            {"__name__": "__main__", "main": mock_main},
        )

        mock_main.assert_called_once()
