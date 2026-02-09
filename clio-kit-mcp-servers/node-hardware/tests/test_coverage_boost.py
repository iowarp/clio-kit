"""
Comprehensive test suite to boost coverage from 81% to >90%
Targets uncovered lines in server.py, mcp_handlers.py, and output_formatter.py
"""

import os
import sys
import json
import pytest
from unittest.mock import patch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))


class TestServerToolsErrorPaths:
    """Test error paths in server.py tool functions"""

    def test_get_cpu_info_tool_error_path(self):
        """Test error handling in get_cpu_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.cpu_info_handler") as mock_handler:
            mock_handler.side_effect = Exception("CPU collection failed")

            # Get the actual function from the tool
            tool_func = (
                server.get_cpu_info_tool.fn
                if hasattr(server.get_cpu_info_tool, "fn")
                else server.get_cpu_info_tool
            )
            result = asyncio.run(tool_func())

            assert result is not None
            assert result["isError"] is True
            assert "CPU collection failed" in result["content"][0]["text"]
            assert result["_meta"]["tool"] == "get_cpu_info"

    def test_get_memory_info_tool_error_path(self):
        """Test error handling in get_memory_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.memory_info_handler") as mock_handler:
            mock_handler.side_effect = RuntimeError("Memory access denied")

            tool_func = (
                server.get_memory_info_tool.fn
                if hasattr(server.get_memory_info_tool, "fn")
                else server.get_memory_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Memory access denied" in result["content"][0]["text"]
            assert result["_meta"]["error"] == "MemoryCollectionError"

    def test_get_system_info_tool_error_path(self):
        """Test error handling in get_system_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.system_info_handler") as mock_handler:
            mock_handler.side_effect = PermissionError("Permission denied")

            tool_func = (
                server.get_system_info_tool.fn
                if hasattr(server.get_system_info_tool, "fn")
                else server.get_system_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Permission denied" in result["content"][0]["text"]

    def test_get_disk_info_tool_error_path(self):
        """Test error handling in get_disk_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.disk_info_handler") as mock_handler:
            mock_handler.side_effect = OSError("Disk not accessible")

            tool_func = (
                server.get_disk_info_tool.fn
                if hasattr(server.get_disk_info_tool, "fn")
                else server.get_disk_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Disk not accessible" in result["content"][0]["text"]

    def test_get_network_info_tool_error_path(self):
        """Test error handling in get_network_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.network_info_handler") as mock_handler:
            mock_handler.side_effect = ConnectionError("Network unavailable")

            tool_func = (
                server.get_network_info_tool.fn
                if hasattr(server.get_network_info_tool, "fn")
                else server.get_network_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Network unavailable" in result["content"][0]["text"]

    def test_get_gpu_info_tool_error_path(self):
        """Test error handling in get_gpu_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.gpu_info_handler") as mock_handler:
            mock_handler.side_effect = Exception("GPU not found")

            tool_func = (
                server.get_gpu_info_tool.fn
                if hasattr(server.get_gpu_info_tool, "fn")
                else server.get_gpu_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "GPU not found" in result["content"][0]["text"]

    def test_get_sensor_info_tool_error_path(self):
        """Test error handling in get_sensor_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.sensor_info_handler") as mock_handler:
            mock_handler.side_effect = Exception("Sensor read failed")

            tool_func = (
                server.get_sensor_info_tool.fn
                if hasattr(server.get_sensor_info_tool, "fn")
                else server.get_sensor_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Sensor read failed" in result["content"][0]["text"]

    def test_get_process_info_tool_error_path(self):
        """Test error handling in get_process_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.process_info_handler") as mock_handler:
            mock_handler.side_effect = Exception("Process enumeration failed")

            tool_func = (
                server.get_process_info_tool.fn
                if hasattr(server.get_process_info_tool, "fn")
                else server.get_process_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Process enumeration failed" in result["content"][0]["text"]

    def test_get_performance_info_tool_error_path(self):
        """Test error handling in get_performance_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.performance_monitor_handler") as mock_handler:
            mock_handler.side_effect = Exception("Performance monitoring failed")

            tool_func = (
                server.get_performance_info_tool.fn
                if hasattr(server.get_performance_info_tool, "fn")
                else server.get_performance_info_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "Performance monitoring failed" in result["content"][0]["text"]


class TestServerRemoteNodeInfo:
    """Test get_remote_node_info_tool comprehensive functionality"""

    def test_get_remote_node_info_tool_success(self):
        """Test successful remote node info collection"""
        import server
        import asyncio

        mock_result = {
            "content": [{"text": '{"success": true}'}],
            "_meta": {"tool": "get_remote_node_info", "success": True},
            "isError": False,
        }

        with patch("mcp_handlers.get_remote_node_info_handler") as mock_handler:
            mock_handler.return_value = mock_result

            tool_func = (
                server.get_remote_node_info_tool.fn
                if hasattr(server.get_remote_node_info_tool, "fn")
                else server.get_remote_node_info_tool
            )
            result = asyncio.run(
                tool_func(
                    hostname="test.example.com",
                    username="testuser",
                    port=22,
                    ssh_key="/path/to/key",
                    timeout=30,
                    components=["cpu", "memory"],
                    exclude_components=["processes"],
                    include_performance=True,
                    include_health=True,
                )
            )

            assert result is not None
            assert result["isError"] is False
            mock_handler.assert_called_once()

    def test_get_remote_node_info_tool_error_path(self):
        """Test error handling in get_remote_node_info_tool"""
        import server
        import asyncio

        with patch("mcp_handlers.get_remote_node_info_handler") as mock_handler:
            mock_handler.side_effect = Exception("SSH connection failed")

            tool_func = (
                server.get_remote_node_info_tool.fn
                if hasattr(server.get_remote_node_info_tool, "fn")
                else server.get_remote_node_info_tool
            )
            result = asyncio.run(
                tool_func(
                    hostname="test.example.com",
                    username="testuser",
                )
            )

            assert result["isError"] is True
            assert "SSH connection failed" in result["content"][0]["text"]
            assert "RemoteHardwareCollectionError" in result["content"][0]["text"]
            assert "troubleshooting" in result["content"][0]["text"]

    def test_get_remote_node_info_tool_with_defaults(self):
        """Test remote node info with default parameters"""
        import server
        import asyncio

        mock_result = {
            "content": [{"text": '{"success": true}'}],
            "_meta": {"tool": "get_remote_node_info"},
            "isError": False,
        }

        with patch("mcp_handlers.get_remote_node_info_handler") as mock_handler:
            mock_handler.return_value = mock_result

            tool_func = (
                server.get_remote_node_info_tool.fn
                if hasattr(server.get_remote_node_info_tool, "fn")
                else server.get_remote_node_info_tool
            )
            result = asyncio.run(tool_func(hostname="192.168.1.100"))

            assert result is not None
            mock_handler.assert_called_once_with(
                hostname="192.168.1.100",
                username=None,
                port=22,
                ssh_key=None,
                timeout=30,
                include_filters=None,
                exclude_filters=None,
            )


class TestServerHealthCheck:
    """Test health_check_tool comprehensive functionality"""

    def test_health_check_tool_success(self):
        """Test successful health check"""
        import server
        import asyncio

        tool_func = (
            server.health_check_tool.fn
            if hasattr(server.health_check_tool, "fn")
            else server.health_check_tool
        )
        result = asyncio.run(tool_func())

        assert result is not None
        assert result["isError"] is False
        assert result["_meta"]["tool"] == "health_check"
        assert result["_meta"]["status"] == "success"

        # Parse the content
        content_text = result["content"][0]["text"]
        health_data = json.loads(content_text)

        assert health_data["server_status"] == "healthy"
        assert "capabilities" in health_data
        assert health_data["capabilities"]["get_node_info"] == "available"
        assert health_data["capabilities"]["get_remote_node_info"] == "available"
        assert "system_compatibility" in health_data
        assert "performance_metrics" in health_data
        assert "health_indicators" in health_data

    def test_health_check_tool_error_path(self):
        """Test health check error handling"""
        import server
        import asyncio
        import json as json_module

        with patch.object(json_module, "dumps") as mock_dumps:
            mock_dumps.side_effect = Exception("JSON serialization failed")

            tool_func = (
                server.health_check_tool.fn
                if hasattr(server.health_check_tool, "fn")
                else server.health_check_tool
            )
            result = asyncio.run(tool_func())

            assert result["isError"] is True
            assert "JSON serialization failed" in result["content"][0]["text"]
            assert result["_meta"]["error"] == "HealthCheckError"


class TestServerMainFunction:
    """Test main function with different transports"""

    def test_main_with_stdio_transport(self):
        """Test main function with stdio transport"""
        import server

        with patch.dict("os.environ", {"MCP_TRANSPORT": "stdio"}):
            with patch.object(server.mcp, "run") as mock_run:
                server.main()
                mock_run.assert_called_once_with(transport="stdio")

    def test_main_with_sse_transport(self):
        """Test main function with SSE transport"""
        import server

        with patch.dict(
            "os.environ",
            {
                "MCP_TRANSPORT": "sse",
                "MCP_SSE_HOST": "127.0.0.1",
                "MCP_SSE_PORT": "9000",
            },
        ):
            with patch.object(server.mcp, "run") as mock_run:
                server.main()
                mock_run.assert_called_once_with(
                    transport="sse", host="127.0.0.1", port=9000
                )

    def test_main_with_sse_default_values(self):
        """Test main function with SSE transport and default values"""
        import server

        with patch.dict("os.environ", {"MCP_TRANSPORT": "sse"}, clear=False):
            # Remove SSE_HOST and SSE_PORT if they exist
            os.environ.pop("MCP_SSE_HOST", None)
            os.environ.pop("MCP_SSE_PORT", None)

            with patch.object(server.mcp, "run") as mock_run:
                server.main()
                mock_run.assert_called_once_with(
                    transport="sse", host="0.0.0.0", port=8000
                )


class TestMcpHandlersEdgeCases:
    """Test edge cases in mcp_handlers"""

    def test_cpu_info_handler_low_usage_insight(self):
        """Test CPU info handler with low usage"""
        import mcp_handlers

        mock_cpu_data = {
            "logical_cores": 8,
            "physical_cores": 4,
            "cpu_model": "Test CPU",
            "frequency": {"current": 2400.0},
            "cpu_usage": [10.0, 12.0, 8.0, 15.0],  # Low average usage
        }

        with patch("mcp_handlers.get_cpu_info") as mock_get:
            mock_get.return_value = mock_cpu_data

            result = mcp_handlers.cpu_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            # Check for low usage insight
            insights = response.get("💡 Insights", [])
            low_usage_found = any("efficiently" in str(insight) for insight in insights)
            assert low_usage_found

    def test_memory_info_handler_swap_usage_insight(self):
        """Test memory info handler with high swap usage"""
        import mcp_handlers

        mock_memory_data = {
            "total": 16000000000,
            "available": 8000000000,
            "used": 8000000000,
            "percent": 50.0,
            "swap_total": 8000000000,
            "swap_used": 5000000000,  # 62.5% swap usage
        }

        with patch("mcp_handlers.get_memory_info") as mock_get:
            mock_get.return_value = mock_memory_data

            result = mcp_handlers.memory_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            # Check for high swap usage insight
            insights = response.get("💡 Insights", [])
            swap_insight_found = any(
                "swap" in str(insight).lower() for insight in insights
            )
            assert swap_insight_found

    def test_disk_info_handler_low_usage_insight(self):
        """Test disk info handler with low disk usage"""
        import mcp_handlers

        mock_disk_data = {
            "partitions": [
                {
                    "mountpoint": "/",
                    "usage": {"percent": 15.0},  # Low usage
                }
            ],
            "disk_io": {},
        }

        with patch("mcp_handlers.get_disk_info") as mock_get:
            mock_get.return_value = mock_disk_data

            result = mcp_handlers.disk_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            good_space_found = any(
                "Good disk space" in str(insight) for insight in insights
            )
            assert good_space_found

    def test_system_info_handler_high_uptime_insight(self):
        """Test system info handler with high uptime"""
        import mcp_handlers

        mock_system_data = {
            "hostname": "test-server",
            "os_info": {"system": "Linux"},
            "uptime": {"days": 35},  # Over 30 days
            "total_users": 3,
        }

        with patch("mcp_handlers.get_system_info") as mock_get:
            mock_get.return_value = mock_system_data

            result = mcp_handlers.system_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            uptime_insight = any("30 days" in str(insight) for insight in insights)
            assert uptime_insight

    def test_system_info_handler_moderate_uptime_insight(self):
        """Test system info handler with moderate uptime"""
        import mcp_handlers

        mock_system_data = {
            "hostname": "test-server",
            "os_info": {"system": "Linux"},
            "uptime": {"days": 15},  # 7-30 days
            "total_users": 0,
        }

        with patch("mcp_handlers.get_system_info") as mock_get:
            mock_get.return_value = mock_system_data

            result = mcp_handlers.system_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            stability_insight = any(
                "stability" in str(insight).lower() for insight in insights
            )
            assert stability_insight

    def test_process_info_handler_high_cpu_processes(self):
        """Test process info handler with high CPU processes"""
        import mcp_handlers

        mock_process_data = {
            "processes": [
                {"name": "proc1", "status": "running", "cpu_percent": 15.0},
                {"name": "proc2", "status": "running", "cpu_percent": 25.0},
                {"name": "proc3", "status": "sleeping", "cpu_percent": 5.0},
            ]
        }

        with patch("mcp_handlers.get_process_info") as mock_get:
            mock_get.return_value = mock_process_data

            result = mcp_handlers.process_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            high_cpu_insight = any(
                "high CPU usage" in str(insight) for insight in insights
            )
            assert high_cpu_insight

    def test_hardware_summary_handler_all_components(self):
        """Test hardware summary handler with all components"""
        import mcp_handlers

        mock_summary_data = {
            "hostname": "test-host",
            "cpu_info": {"model": "Test CPU"},
            "memory_info": {"total": 16000000000},
            "disk_info": {"partitions": []},
            "network_info": {"interfaces": []},
        }

        with patch("mcp_handlers.get_hardware_summary") as mock_get:
            mock_get.return_value = mock_summary_data

            result = mcp_handlers.hardware_summary_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            # Should have insights for all 4 components
            assert len(insights) >= 4

    def test_performance_monitor_handler_all_high(self):
        """Test performance monitor handler with all high usage"""
        import mcp_handlers

        mock_perf_data = {
            "cpu_usage": 85.0,  # High
            "memory_usage": 90.0,  # High
            "disk_usage": 95.0,  # High
        }

        with patch("mcp_handlers.monitor_performance") as mock_monitor:
            mock_monitor.return_value = mock_perf_data

            result = mcp_handlers.performance_monitor_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            # Should have 3 high usage insights
            assert len(insights) >= 3

    def test_gpu_info_handler_no_gpus(self):
        """Test GPU info handler with no GPUs"""
        import mcp_handlers

        mock_gpu_data = {"gpus": [], "nvidia_available": False}

        with patch("mcp_handlers.get_gpu_info") as mock_get:
            mock_get.return_value = mock_gpu_data

            result = mcp_handlers.gpu_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            no_gpu_insight = any(
                "No GPUs detected" in str(insight) for insight in insights
            )
            assert no_gpu_insight

    def test_sensor_info_handler_no_sensors(self):
        """Test sensor info handler with no sensors"""
        import mcp_handlers

        mock_sensor_data = {"sensors": []}

        with patch("mcp_handlers.get_sensor_info") as mock_get:
            mock_get.return_value = mock_sensor_data

            result = mcp_handlers.sensor_info_handler()

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            no_sensor_insight = any(
                "No sensors detected" in str(insight) for insight in insights
            )
            assert no_sensor_insight


class TestGetNodeInfoHandlerValidation:
    """Test get_node_info_handler validation and edge cases"""

    def test_get_node_info_handler_invalid_include_filters_type(self):
        """Test validation of include_filters parameter type"""
        import mcp_handlers

        result = mcp_handlers.get_node_info_handler(
            include_filters="cpu,memory",  # Should be list, not string
        )

        assert result is not None
        content_text = result["content"][0]["text"]
        response = json.loads(content_text)

        assert response["❌ Status"] == "Error"
        assert "InvalidParameterType" in response["🚨 Error Type"]
        assert "must be a list" in response["📝 Error Message"]

    def test_get_node_info_handler_invalid_exclude_filters_type(self):
        """Test validation of exclude_filters parameter type"""
        import mcp_handlers

        result = mcp_handlers.get_node_info_handler(
            exclude_filters="processes",  # Should be list, not string
        )

        assert result is not None
        content_text = result["content"][0]["text"]
        response = json.loads(content_text)

        assert response["❌ Status"] == "Error"
        assert "InvalidParameterType" in response["🚨 Error Type"]

    def test_get_node_info_handler_with_error_result(self):
        """Test handling of error from get_node_info"""
        import mcp_handlers

        with patch("mcp_handlers.get_node_info") as mock_get:
            mock_get.return_value = {
                "error": "Hardware access failed",
                "error_type": "HardwareError",
            }

            result = mcp_handlers.get_node_info_handler(
                include_filters=["cpu"], max_response_size=10000
            )

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            assert response["❌ Status"] == "Error"
            assert "Hardware access failed" in response["📝 Error Message"]

    def test_get_node_info_handler_with_filters_applied(self):
        """Test get_node_info_handler with filters applied"""
        import mcp_handlers

        mock_result = {
            "_metadata": {
                "hostname": "test-host",
                "components_requested": ["cpu", "memory"],
                "components_collected": ["cpu", "memory"],
                "collection_method": "local",
                "errors": [],
                "response_size_controlled": False,
            },
            "cpu_info": {"model": "Test CPU"},
            "memory_info": {"total": 16000000000},
        }

        with patch("mcp_handlers.get_node_info") as mock_get:
            mock_get.return_value = mock_result

            result = mcp_handlers.get_node_info_handler(
                include_filters=["cpu", "memory"], exclude_filters=["processes"]
            )

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            include_found = any(
                "include filters" in str(insight).lower() for insight in insights
            )
            exclude_found = any(
                "exclude filters" in str(insight).lower() for insight in insights
            )
            assert include_found
            assert exclude_found

    def test_get_node_info_handler_with_size_control(self):
        """Test get_node_info_handler with response size control"""
        import mcp_handlers

        mock_result = {
            "_metadata": {
                "hostname": "test-host",
                "components_requested": ["cpu", "memory", "disk"],
                "components_collected": ["cpu", "memory"],  # One missing
                "collection_method": "local",
                "errors": [],
                "response_size_controlled": True,
            },
            "cpu_info": {"model": "Test CPU"},
            "memory_info": {"total": 16000000000},
        }

        with patch("mcp_handlers.get_node_info") as mock_get:
            mock_get.return_value = mock_result

            result = mcp_handlers.get_node_info_handler(max_response_size=5000)

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            size_control_found = any(
                "size was controlled" in str(insight).lower() for insight in insights
            )
            assert size_control_found


class TestGetRemoteNodeInfoHandler:
    """Test get_remote_node_info_handler functionality"""

    def test_get_remote_node_info_handler_success(self):
        """Test successful remote node info collection"""
        import mcp_handlers

        mock_result = {
            "_metadata": {
                "hostname": "remote-host",
                "ssh_hostname": "remote-host",
                "ssh_username": "testuser",
                "collection_method": "ssh",
                "ssh_timeout": 30,
                "ssh_key_used": True,
            },
            "cpu_info": {"model": "Remote CPU"},
        }

        with patch("mcp_handlers.get_remote_node_info") as mock_remote:
            mock_remote.return_value = mock_result

            result = mcp_handlers.get_remote_node_info_handler(
                hostname="remote-host",
                username="testuser",
                port=22,
                ssh_key="/path/to/key",
                timeout=30,
                include_filters=["cpu"],
                exclude_filters=["processes"],
            )

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            assert response["✅ Status"] == "Success"
            insights = response.get("💡 Insights", [])
            ssh_insight = any("SSH" in str(insight) for insight in insights)
            assert ssh_insight

    def test_get_remote_node_info_handler_with_error(self):
        """Test remote node info handler with error result"""
        import mcp_handlers

        mock_result = {
            "error": "Connection refused",
            "error_type": "ConnectionError",
        }

        with patch("mcp_handlers.get_remote_node_info") as mock_remote:
            mock_remote.return_value = mock_result

            result = mcp_handlers.get_remote_node_info_handler(
                hostname="unreachable-host"
            )

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            assert response["❌ Status"] == "Error"
            assert "Connection refused" in response["📝 Error Message"]

    def test_get_remote_node_info_handler_exception(self):
        """Test remote node info handler with exception"""
        import mcp_handlers

        with patch("mcp_handlers.get_remote_node_info") as mock_remote:
            mock_remote.side_effect = Exception("Network timeout")

            result = mcp_handlers.get_remote_node_info_handler(hostname="test-host")

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            assert response["❌ Status"] == "Error"
            assert "Network timeout" in response["📝 Error Message"]

    def test_get_remote_node_info_handler_password_auth(self):
        """Test remote node info handler with password authentication"""
        import mcp_handlers

        mock_result = {
            "_metadata": {
                "hostname": "remote-host",
                "ssh_hostname": "remote-host",
                "ssh_username": "testuser",
                "collection_method": "ssh",
                "ssh_timeout": 30,
                "ssh_key_used": False,  # Password auth
            },
            "system_info": {"os": "Linux"},
        }

        with patch("mcp_handlers.get_remote_node_info") as mock_remote:
            mock_remote.return_value = mock_result

            result = mcp_handlers.get_remote_node_info_handler(
                hostname="remote-host", username="testuser"
            )

            assert result is not None
            content_text = result["content"][0]["text"]
            response = json.loads(content_text)

            insights = response.get("💡 Insights", [])
            password_auth = any(
                "Password authentication" in str(insight) for insight in insights
            )
            assert password_auth


class TestOutputFormatterFilteredResponse:
    """Test create_filtered_response in output_formatter"""

    def test_create_filtered_response_with_filters(self):
        """Test create_filtered_response with filter information"""
        from utils.output_formatter import NodeHardwareFormatter

        data = {"items": [1, 2, 3]}
        filters = {"type": "hardware", "status": "active"}

        result = NodeHardwareFormatter.create_filtered_response(
            operation="test_operation",
            data=data,
            filters=filters,
            total_items=100,
            filtered_items=3,
        )

        assert result is not None
        assert "🔍 Applied Filters" in result
        assert "📊 Filter Results" in result
        assert result["📊 Filter Results"]["🔢 Total Items"] == 100
        assert result["📊 Filter Results"]["✅ Filtered Items"] == 3
        assert result["📊 Filter Results"]["📉 Filtered Out"] == 97
        assert "3.0%" in result["📊 Filter Results"]["📊 Filter Ratio"]

    def test_create_filtered_response_without_filters(self):
        """Test create_filtered_response without filter information"""
        from utils.output_formatter import NodeHardwareFormatter

        data = {"items": [1, 2, 3]}

        result = NodeHardwareFormatter.create_filtered_response(
            operation="test_operation", data=data
        )

        assert result is not None
        assert "🔍 Applied Filters" not in result
        assert "📊 Filter Results" not in result

    def test_create_filtered_response_zero_total_items(self):
        """Test create_filtered_response with zero total items"""
        from utils.output_formatter import NodeHardwareFormatter

        data = {"items": []}

        result = NodeHardwareFormatter.create_filtered_response(
            operation="test_operation", data=data, total_items=0, filtered_items=0
        )

        assert result is not None
        assert "📊 Filter Results" in result
        assert result["📊 Filter Results"]["📊 Filter Ratio"] == "0%"


class TestOutputFormatterInsightFormatting:
    """Test insight formatting variations in output_formatter"""

    def test_format_insights_error_keyword(self):
        """Test insight formatting with error keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["An error occurred during processing"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("🚨")

    def test_format_insights_fail_keyword(self):
        """Test insight formatting with fail keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["Operation failed to complete"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("🚨")

    def test_format_insights_warning_keyword(self):
        """Test insight formatting with warning keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["Warning: High resource usage"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("⚠️")

    def test_format_insights_high_keyword(self):
        """Test insight formatting with high keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["High CPU utilization detected"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("⚠️")

    def test_format_insights_good_keyword(self):
        """Test insight formatting with good keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["Good system performance"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("✅")

    def test_format_insights_success_keyword(self):
        """Test insight formatting with success keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["Successfully completed operation"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("✅")

    def test_format_insights_recommend_keyword(self):
        """Test insight formatting with recommend keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["Recommend upgrading memory"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("💡")

    def test_format_insights_suggest_keyword(self):
        """Test insight formatting with suggest keyword"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["Suggest increasing disk space"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("💡")

    def test_format_insights_default(self):
        """Test insight formatting with default emoji"""
        from utils.output_formatter import NodeHardwareFormatter

        insights = ["System is operating normally"]
        formatted = NodeHardwareFormatter._format_insights(insights)

        assert len(formatted) == 1
        assert formatted[0].startswith("ℹ️")


class TestOutputFormatterSummaryFormatting:
    """Test summary formatting edge cases"""

    def test_format_summary_with_all_keys(self):
        """Test summary formatting with various key types"""
        from utils.output_formatter import NodeHardwareFormatter

        summary = {
            "count": 10,
            "total_items": 100,
            "response_time": 1.5,
            "memory_size": 8000000000,
            "errors": 0,  # Changed from error_count to errors to get 🚨 emoji
            "success_rate": 100,
            "hostname": "test-host",
            "nodes_active": 5,
            "other_metric": 42,
        }

        formatted = NodeHardwareFormatter._format_summary(summary)

        # Check that emojis are present in formatted keys
        formatted_keys = list(formatted.keys())
        assert any("📊" in key for key in formatted_keys)  # count
        assert any("📈" in key for key in formatted_keys)  # total
        assert any("⏱️" in key for key in formatted_keys)  # time
        assert any("💾" in key for key in formatted_keys)  # size/memory
        assert any("🚨" in key for key in formatted_keys)  # error
        assert any("✅" in key for key in formatted_keys)  # success
        assert any("🌐" in key for key in formatted_keys)  # host
        assert any("🖥️" in key for key in formatted_keys)  # nodes


class TestOutputFormatterMetadataFormatting:
    """Test metadata formatting edge cases"""

    def test_format_metadata_with_all_keys(self):
        """Test metadata formatting with various key types"""
        from utils.output_formatter import NodeHardwareFormatter

        metadata = {
            "hostname": "test-host",
            "username": "testuser",
            "method": "ssh",
            "protocol": "SSHv2",
            "port": 22,
            "timeout": 30,
            "version": "1.0.0",
            "other_info": "value",
        }

        formatted = NodeHardwareFormatter._format_metadata(metadata)

        assert len(formatted) == len(metadata)
        assert "🌐" in list(formatted.keys())[0]  # hostname
        assert "👤" in list(formatted.keys())[1]  # user
        assert "🔧" in list(formatted.keys())[2]  # method
        assert "🔌" in list(formatted.keys())[3]  # protocol
        assert "🚪" in list(formatted.keys())[4]  # port
        assert "⏳" in list(formatted.keys())[5]  # timeout
        assert "📋" in list(formatted.keys())[6]  # version


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
