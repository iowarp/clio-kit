import pytest
from unittest.mock import patch
from fastapi import HTTPException
from src.mcp_handlers import (
    handle_mcp_request,
    list_resources,
    list_tools,
    get_resource,
    call_tool,
    resources,
    tools,
)


class TestMCPHandlers:
    @pytest.mark.asyncio
    async def test_handle_mcp_request_invalid_jsonrpc(self):
        # Test invalid JSON-RPC request
        data = {"method": "test"}
        with pytest.raises(HTTPException) as exc_info:
            await handle_mcp_request(data)
        assert exc_info.value.status_code == 400
        assert "Invalid JSON-RPC request" in str(exc_info.value.detail)

    @pytest.mark.asyncio
    async def test_handle_mcp_request_missing_method(self):
        # Test missing method
        data = {"jsonrpc": "2.0"}
        with pytest.raises(HTTPException) as exc_info:
            await handle_mcp_request(data)
        assert exc_info.value.status_code == 400

    @pytest.mark.asyncio
    async def test_handle_mcp_request_list_resources(self):
        # Test list resources method
        data = {"jsonrpc": "2.0", "method": "mcp/listResources", "id": 1}
        result = await handle_mcp_request(data)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert "result" in result

    @pytest.mark.asyncio
    async def test_handle_mcp_request_list_tools(self):
        # Test list tools method
        data = {"jsonrpc": "2.0", "method": "mcp/listTools", "id": 2}
        result = await handle_mcp_request(data)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 2
        assert "result" in result

    @pytest.mark.asyncio
    async def test_handle_mcp_request_get_resource(self):
        # Test get resource method
        data = {
            "jsonrpc": "2.0",
            "method": "mcp/getResource",
            "params": {"id": "resource1"},
            "id": 3,
        }
        result = await handle_mcp_request(data)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 3

    @pytest.mark.asyncio
    async def test_handle_mcp_request_call_tool(self):
        # Test call tool method with mock
        with patch("src.mcp_handlers.read_column") as mock_read:
            mock_read.return_value = {"status": "success", "data": "test"}
            data = {
                "jsonrpc": "2.0",
                "method": "mcp/callTool",
                "params": {"tool": "parquet", "column": "temperature"},
                "id": 4,
            }
            result = await handle_mcp_request(data)
            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 4

    @pytest.mark.asyncio
    async def test_handle_mcp_request_unsupported_method(self):
        # Test unsupported method
        data = {"jsonrpc": "2.0", "method": "invalid/method", "id": 5}
        with pytest.raises(HTTPException) as exc_info:
            await handle_mcp_request(data)
        assert exc_info.value.status_code == 400
        assert "Method not supported" in str(exc_info.value.detail)

    def test_list_resources(self):
        # Test list resources function
        result = list_resources(1)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 1
        assert result["result"] == resources
        assert len(result["result"]) == 4

    def test_list_tools(self):
        # Test list tools function
        result = list_tools(2)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 2
        assert result["result"] == tools
        assert len(result["result"]) == 4

    def test_get_resource_valid_id(self):
        # Test get resource with valid ID
        result = get_resource({"id": "resource1"}, 3)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 3
        assert "result" in result
        assert result["result"]["id"] == "resource1"

    def test_get_resource_invalid_id(self):
        # Test get resource with invalid ID
        result = get_resource({"id": "invalid_id"}, 4)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 4
        assert "error" in result
        assert result["error"]["code"] == -32601

    def test_get_resource_missing_id(self):
        # Test get resource with missing ID
        result = get_resource({}, 5)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 5
        assert "error" in result
        assert result["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_call_tool_missing_tool(self):
        # Test call tool with missing tool parameter
        result = await call_tool({}, 6)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 6
        assert "error" in result
        assert result["error"]["code"] == -32602

    @pytest.mark.asyncio
    async def test_call_tool_parquet(self):
        # Test parquet tool
        with patch("src.mcp_handlers.read_column") as mock_read:
            mock_read.return_value = {"status": "success", "data": ["col1", "col2"]}
            result = await call_tool({"tool": "parquet", "column": "temperature"}, 7)
            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 7
            assert "result" in result
            mock_read.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_sort(self):
        # Test sort tool
        with patch("src.mcp_handlers.sort_log_by_timestamp") as mock_sort:
            mock_sort.return_value = {"status": "success", "sorted_entries": 100}
            result = await call_tool({"tool": "sort", "file": "test.log"}, 8)
            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 8
            assert "result" in result
            mock_sort.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_compress(self):
        # Test compress tool
        with patch("src.mcp_handlers.compress_file") as mock_compress:
            mock_compress.return_value = {
                "status": "success",
                "compressed_file": "test.gz",
            }
            result = await call_tool({"tool": "compress", "file": "test.txt"}, 9)
            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 9
            assert "result" in result
            mock_compress.assert_called_once()

    @pytest.mark.asyncio
    async def test_call_tool_pandas(self):
        # Test pandas tool
        with patch("src.mcp_handlers.analyze_csv") as mock_analyze:
            mock_analyze.return_value = {"status": "success", "data": []}
            result = await call_tool(
                {
                    "tool": "pandas",
                    "file": "test.csv",
                    "column": "score",
                    "threshold": 75,
                },
                10,
            )
            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 10
            assert "result" in result
            mock_analyze.assert_called_once_with("data/test.csv", "score", 75)

    @pytest.mark.asyncio
    async def test_call_tool_pandas_defaults(self):
        # Test pandas tool with default parameters
        with patch("src.mcp_handlers.analyze_csv") as mock_analyze:
            mock_analyze.return_value = {"status": "success", "data": []}
            result = await call_tool({"tool": "pandas"}, 11)
            assert result["jsonrpc"] == "2.0"
            assert result["id"] == 11
            mock_analyze.assert_called_once_with("data/data.csv", "marks", 50)

    @pytest.mark.asyncio
    async def test_call_tool_invalid_tool(self):
        # Test invalid tool
        result = await call_tool({"tool": "invalid_tool"}, 12)
        assert result["jsonrpc"] == "2.0"
        assert result["id"] == 12
        assert "error" in result
        assert result["error"]["code"] == -32601

    @pytest.mark.asyncio
    async def test_call_tool_parquet_default_file(self):
        # Test parquet tool with default file
        with patch("src.mcp_handlers.read_column") as mock_read:
            mock_read.return_value = {"status": "success"}
            await call_tool({"tool": "parquet", "column": "temperature"}, 13)
            mock_read.assert_called_once_with(
                "data/weather_data.parquet", "temperature"
            )

    @pytest.mark.asyncio
    async def test_call_tool_sort_default_file(self):
        # Test sort tool with default file
        with patch("src.mcp_handlers.sort_log_by_timestamp") as mock_sort:
            mock_sort.return_value = {"status": "success"}
            await call_tool({"tool": "sort"}, 14)
            mock_sort.assert_called_once_with("data/huge_log.txt")

    @pytest.mark.asyncio
    async def test_call_tool_compress_default_file(self):
        # Test compress tool with default file
        with patch("src.mcp_handlers.compress_file") as mock_compress:
            mock_compress.return_value = {"status": "success"}
            await call_tool({"tool": "compress"}, 15)
            mock_compress.assert_called_once_with("data/output.log")

    def test_resources_structure(self):
        # Test resources structure and content
        assert len(resources) == 4
        for resource in resources:
            assert "id" in resource
            assert "name" in resource
            assert "type" in resource
            assert "description" in resource
            assert "path" in resource
            assert "format" in resource

    def test_tools_structure(self):
        # Test tools structure and content
        assert len(tools) == 4
        for tool in tools:
            assert "id" in tool
            assert "name" in tool
            assert "description" in tool
            assert "usage" in tool

    @pytest.mark.asyncio
    async def test_handle_mcp_request_with_params(self):
        # Test handle_mcp_request with params
        data = {
            "jsonrpc": "2.0",
            "method": "mcp/getResource",
            "params": {"id": "resource2"},
            "id": 16,
        }
        result = await handle_mcp_request(data)
        assert result["result"]["id"] == "resource2"

    @pytest.mark.asyncio
    async def test_handle_mcp_request_no_params(self):
        # Test handle_mcp_request without params
        data = {"jsonrpc": "2.0", "method": "mcp/listResources", "id": 17}
        result = await handle_mcp_request(data)
        assert len(result["result"]) == 4
