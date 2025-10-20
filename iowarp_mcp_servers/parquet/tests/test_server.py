import pytest
from fastapi.testclient import TestClient
from src.server import app, main
from unittest.mock import patch


class TestParquetServer:
    @pytest.fixture
    def client(self):
        return TestClient(app)

    def test_mcp_endpoint_valid_request(self, client):
        # Test valid MCP request
        test_data = {"jsonrpc": "2.0", "method": "mcp/listResources", "id": 1}

        response = client.post("/mcp", json=test_data)
        assert response.status_code == 200
        result = response.json()
        assert "jsonrpc" in result
        assert result["jsonrpc"] == "2.0"
        assert "result" in result

    @patch("src.server.handle_mcp_request")
    def test_mcp_endpoint_handler_called(self, mock_handler, client):
        # Test that handle_mcp_request is called with correct data
        mock_handler.return_value = {"jsonrpc": "2.0", "result": "test", "id": 1}

        test_data = {"jsonrpc": "2.0", "method": "mcp/listResources", "id": 1}

        response = client.post("/mcp", json=test_data)
        assert response.status_code == 200
        mock_handler.assert_called_once_with(test_data)

    @patch("src.server.handle_mcp_request")
    def test_mcp_endpoint_async_response(self, mock_handler, client):
        # Test async response handling
        mock_handler.return_value = {"jsonrpc": "2.0", "result": "async_test", "id": 1}

        test_data = {
            "jsonrpc": "2.0",
            "method": "mcp/callTool",
            "params": {"tool": "parquet"},
            "id": 1,
        }

        response = client.post("/mcp", json=test_data)
        assert response.status_code == 200

    def test_mcp_endpoint_invalid_json(self, client):
        # Test invalid JSON handling - this tests FastAPI's built-in JSON parsing
        import pytest

        with pytest.raises(Exception):  # Could be JSONDecodeError or HTTP error
            client.post(
                "/mcp",
                content="invalid json",
                headers={"content-type": "application/json"},
            )

    def test_mcp_endpoint_empty_request(self, client):
        # Test empty request
        response = client.post("/mcp", json={})
        assert response.status_code == 400

    @patch("uvicorn.run")
    def test_main_function(self, mock_uvicorn):
        # Test main function calls uvicorn.run with correct parameters
        main()
        mock_uvicorn.assert_called_once_with(app, host="0.0.0.0", port=8000)

    def test_app_instance(self):
        # Test that app is properly initialized
        from src.server import app

        assert app is not None
        assert hasattr(app, "routes")

    def test_mcp_endpoint_multiple_methods(self, client):
        # Test different MCP methods
        methods = [
            "mcp/listResources",
            "mcp/listTools",
            "mcp/getResource",
            "mcp/callTool",
        ]

        for method in methods[:2]:  # Test first two that don't need params
            test_data = {"jsonrpc": "2.0", "method": method, "id": 1}
            response = client.post("/mcp", json=test_data)
            assert response.status_code == 200

    def test_mcp_endpoint_with_params(self, client):
        # Test MCP requests with parameters
        test_data = {
            "jsonrpc": "2.0",
            "method": "mcp/getResource",
            "params": {"id": "resource1"},
            "id": 1,
        }

        response = client.post("/mcp", json=test_data)
        assert response.status_code == 200
        result = response.json()
        assert "result" in result

    def test_mcp_endpoint_exception_handling(self, client):
        # Test exception handling in endpoint by using invalid method
        # This will trigger HTTPException in handle_mcp_request
        test_data = {"jsonrpc": "2.0", "method": "mcp/invalidMethod", "id": 1}

        response = client.post("/mcp", json=test_data)
        # Should return 400 for unsupported method
        assert response.status_code == 400
