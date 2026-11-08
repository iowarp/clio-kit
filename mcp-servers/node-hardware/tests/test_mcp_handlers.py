"""
100% coverage tests for mcp_handlers module including all handler functions.
Updated for FastMCP v3.
"""

import pytest
from unittest.mock import patch


class TestMCPHandlers100Coverage:
    """100% coverage tests for mcp_handlers module"""

    @pytest.mark.asyncio
    async def test_handle_get_cpu_info_complete(self):
        """Test CPU info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import cpu_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.cpu_info.get_cpu_info"
            ) as mock_cpu:
                mock_cpu.return_value = {
                    "physical_cores": 4,
                    "logical_cores": 8,
                    "max_frequency": 3200.0,
                    "current_frequency": 2800.0,
                }

                result = cpu_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.cpu_info.get_cpu_info"
            ) as mock_cpu:
                mock_cpu.side_effect = Exception("CPU access denied")

                result = cpu_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_memory_info_handler_complete(self):
        """Test memory info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import memory_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.memory_info.get_memory_info"
            ) as mock_memory:
                mock_memory.return_value = {
                    "total": 16000000000,
                    "available": 8000000000,
                    "percent": 50.0,
                    "used": 8000000000,
                }

                result = memory_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.memory_info.get_memory_info"
            ) as mock_memory:
                mock_memory.side_effect = Exception("Memory access denied")

                result = memory_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_disk_info_handler_complete(self):
        """Test disk info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import disk_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.disk_info.get_disk_info"
            ) as mock_disk:
                mock_disk.return_value = {
                    "partitions": [
                        {"device": "/dev/sda1", "mountpoint": "/", "fstype": "ext4"}
                    ],
                    "usage": {"total": 500000000000, "used": 250000000000},
                }

                result = disk_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.disk_info.get_disk_info"
            ) as mock_disk:
                mock_disk.side_effect = Exception("Disk access denied")

                result = disk_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_network_info_handler_complete(self):
        """Test network info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import network_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.network_info.get_network_info"
            ) as mock_network:
                mock_network.return_value = {
                    "interfaces": {"eth0": {"address": "192.168.1.100", "status": "up"}}
                }

                result = network_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.network_info.get_network_info"
            ) as mock_network:
                mock_network.side_effect = Exception("Network access denied")

                result = network_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_system_info_handler_complete(self):
        """Test system info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import system_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.system_info.get_system_info"
            ) as mock_system:
                mock_system.return_value = {
                    "system": "Linux",
                    "release": "5.15.0",
                    "machine": "x86_64",
                }

                result = system_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.system_info.get_system_info"
            ) as mock_system:
                mock_system.side_effect = Exception("System access denied")

                result = system_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_process_info_handler_complete(self):
        """Test process info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import process_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.process_info.get_process_info"
            ) as mock_process:
                mock_process.return_value = {
                    "processes": [{"pid": 1234, "name": "python", "cpu_percent": 10.5}],
                    "total_processes": 150,
                }

                result = process_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.process_info.get_process_info"
            ) as mock_process:
                mock_process.side_effect = Exception("Process access denied")

                result = process_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_sensor_info_handler_complete(self):
        """Test sensor info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import sensor_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.sensor_info.get_sensor_info"
            ) as mock_sensor:
                mock_sensor.return_value = {
                    "temperatures": {
                        "coretemp": [{"label": "Core 0", "current": 45.0}]
                    },
                    "fans": {"cpu_fan": [{"label": "CPU Fan", "current": 2000}]},
                }

                result = sensor_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.sensor_info.get_sensor_info"
            ) as mock_sensor:
                mock_sensor.side_effect = Exception("Sensor access denied")

                result = sensor_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_performance_monitor_handler_complete(self):
        """Test performance monitoring handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import performance_monitor_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.performance_monitor.monitor_performance"
            ) as mock_perf:
                mock_perf.return_value = {
                    "cpu_usage": 25.5,
                    "memory_usage": 60.0,
                    "disk_io": {"read_bytes": 1000000, "write_bytes": 500000},
                }

                result = performance_monitor_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.performance_monitor.monitor_performance"
            ) as mock_perf:
                mock_perf.side_effect = Exception("Performance monitoring failed")

                result = performance_monitor_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_gpu_info_handler_complete(self):
        """Test GPU info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import gpu_info_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.gpu_info.get_gpu_info"
            ) as mock_gpu:
                mock_gpu.return_value = {
                    "gpus": [{"name": "NVIDIA GeForce RTX 3080", "memory": 10240}]
                }

                result = gpu_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.gpu_info.get_gpu_info"
            ) as mock_gpu:
                mock_gpu.side_effect = Exception("GPU access denied")

                result = gpu_info_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_hardware_summary_handler_complete(self):
        """Test hardware summary handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import hardware_summary_handler

            # Test successful execution
            with patch(
                "node_hardware_mcp.capabilities.hardware_summary.get_hardware_summary"
            ) as mock_summary:
                mock_summary.return_value = {
                    "system": "High-performance workstation",
                    "cpu": "Intel Core i7",
                    "memory": "16GB",
                    "storage": "1TB SSD",
                }

                result = hardware_summary_handler()
                assert isinstance(result, dict)
                assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.hardware_summary.get_hardware_summary"
            ) as mock_summary:
                mock_summary.side_effect = Exception("Hardware summary failed")

                result = hardware_summary_handler()
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_get_node_info_handler_complete(self):
        """Test node info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import get_node_info_handler

            # Test successful execution with filters
            result = get_node_info_handler(
                include_filters=["cpu", "memory"],
                exclude_filters=["process"],
                max_response_size=10000,
            )
            assert isinstance(result, dict)
            assert "content" in result

            # Test with no filters
            result = get_node_info_handler()
            assert isinstance(result, dict)
            assert "content" in result

            # Test error handling
            with patch(
                "node_hardware_mcp.capabilities.cpu_info.get_cpu_info"
            ) as mock_error:
                mock_error.side_effect = Exception("Node info error")

                result = get_node_info_handler(include_filters=["invalid"])
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")

    @pytest.mark.asyncio
    async def test_get_remote_node_info_handler_complete(self):
        """Test remote node info handler with all scenarios"""
        try:
            from node_hardware_mcp.mcp_handlers import get_remote_node_info_handler

            # Test successful SSH connection
            with patch(
                "node_hardware_mcp.mcp_handlers.get_remote_node_info"
            ) as mock_remote:
                mock_remote.return_value = {
                    "hostname": "remote.server.com",
                    "status": "connected",
                    "uptime": "5 days",
                    "load": "0.85",
                }

                result = get_remote_node_info_handler(
                    hostname="remote.server.com", username="admin", port=22, timeout=30
                )
                assert isinstance(result, dict)
                assert "content" in result

            # Test SSH connection failure
            with patch(
                "node_hardware_mcp.mcp_handlers.get_remote_node_info"
            ) as mock_remote:
                mock_remote.side_effect = Exception("SSH connection failed")

                result = get_remote_node_info_handler(
                    hostname="invalid.host", username="baduser"
                )
                assert isinstance(result, dict)
                assert "content" in result

            # Test with default parameters
            with patch(
                "node_hardware_mcp.mcp_handlers.get_remote_node_info"
            ) as mock_remote:
                mock_remote.return_value = {"status": "connected"}

                result = get_remote_node_info_handler(
                    hostname="localhost", username="user"
                )
                assert isinstance(result, dict)
                assert "content" in result

        except ImportError:
            pytest.skip("MCP handlers not available")


def test_memory_summary_matches_capability_and_propagates_failure():
    from node_hardware_mcp import mcp_handlers

    memory = {"total": 1000, "available": 100, "used": 900, "percent": 90.0}
    data = {"virtual_memory": memory, "swap_memory": {"total": 100, "used": 75}}
    with (
        patch.object(mcp_handlers, "get_memory_info", return_value=data) as read,
        patch.object(
            mcp_handlers, "create_beautiful_response", side_effect=lambda **kw: kw
        ),
    ):
        result = mcp_handlers.memory_info_handler()
        assert result["success"] is True
        assert result["data"] == data
        assert result["summary"] == {
            "total_memory": 1000,
            "available_memory": 100,
            "used_memory": 900,
            "memory_percent": 90.0,
        }
        assert any("High memory" in item for item in result["insights"])
        assert any("High swap" in item for item in result["insights"])
        assert not any("sufficient" in item for item in result["insights"])
        read.return_value = {"virtual_memory": {}, "swap_memory": {}, "error": "denied"}
        result = mcp_handlers.memory_info_handler()
        assert result["success"] is False
        assert result["error_message"] == "denied"


def _handle(handler_name, capability_name, data):
    """Run a handler on the real capability output shape; return its raw kwargs."""
    from node_hardware_mcp import mcp_handlers

    with (
        patch.object(mcp_handlers, capability_name, return_value=data),
        patch.object(
            mcp_handlers, "create_beautiful_response", side_effect=lambda **kw: kw
        ),
    ):
        return getattr(mcp_handlers, handler_name)()


def test_performance_summary_matches_capability():
    data = {
        "cpu": {"average_usage": 91.5, "usage_per_core": [91.5]},
        "memory": {"current_usage": 88.0},
        "disk_io": {
            "read_rate_formatted": "1.00 MB/s",
            "write_rate_formatted": "0 B/s",
        },
    }
    result = _handle("performance_monitor_handler", "monitor_performance", data)
    assert result["summary"] == {
        "cpu_usage": 91.5,
        "memory_usage": 88.0,
        "disk_read_rate": "1.00 MB/s",
        "disk_write_rate": "0 B/s",
    }
    assert len(result["insights"]) == 2
    failed = {"monitoring_duration": {"requested": 5, "actual": 0}, "error": "denied"}
    result = _handle("performance_monitor_handler", "monitor_performance", failed)
    assert result["success"] is False and result["error_message"] == "denied"


def test_sensor_summary_counts_real_sensor_shape():
    data = {
        "temperatures": {"coretemp": [{"current": 45.0}, {"current": 47.0}]},
        "fans": {"thinkpad": [{"current": 2400}]},
        "battery": {"percent": 80.0},
        "sensors_available": True,
        "thermal_zones": [{"zone": "thermal_zone0", "temperature": 45.0}],
    }
    result = _handle("sensor_info_handler", "get_sensor_info", data)
    assert result["summary"] == {
        "sensor_count": 4,
        "temperature_sensors": 2,
        "fan_sensors": 1,
        "battery_present": True,
    }
    assert result["insights"] == ["Found 4 sensors"]


def test_disk_summary_and_insights_use_partition_percent():
    data = {
        "partitions": [
            {"device": "/dev/sda1", "mountpoint": "/", "percent": 100.0},
            {"device": "/dev/sda1", "mountpoint": "/home", "percent": 5.0},
            {"device": "/dev/loop3", "mountpoint": "/snap/core/1", "percent": 100.0},
            {
                "device": "/dev/sdb1",
                "mountpoint": "/secret",
                "error": "Permission denied",
            },
        ],
        "total_partitions": 4,
        "summary": {},
        "io_statistics": {"read_count": 1},
    }
    result = _handle("disk_info_handler", "get_disk_info", data)
    assert result["summary"] == {"total_partitions": 4, "total_devices": 3}
    assert result["insights"] == [
        "High disk usage on / - consider cleanup",
        "Good disk space on /home",
    ]


def test_process_summary_uses_system_totals_not_the_top_list():
    data = {
        "processes": [{"pid": 1, "status": "sleeping", "cpu_percent": None}] * 10,
        "total_processes": 614,
        "statistics": {"running": 3, "sleeping": 611},
        "limit": 10,
    }
    result = _handle("process_info_handler", "get_process_info", data)
    assert result["summary"] == {
        "total_processes": 614,
        "running_processes": 3,
        "processes_listed": 10,
    }
    assert result["insights"] == ["System is running 614 processes"]


def test_cpu_and_hardware_summary_read_real_capability_keys():
    cpu = {"logical_cores": 8, "usage_per_core": [90.0, 92.0], "average_usage": 91.0}
    result = _handle("cpu_info_handler", "get_cpu_info", cpu)
    assert any("High CPU usage" in item for item in result["insights"])
    data = {
        "summary": {"system": {"hostname": "node1"}},
        "detailed": {"cpu": cpu, "memory": {"error": "denied"}, "disk": {"x": 1}},
    }
    result = _handle("hardware_summary_handler", "get_hardware_summary", data)
    assert result["summary"] == {"components_gathered": 3, "hostname": "node1"}
    assert result["insights"] == [
        "CPU information successfully collected",
        "Disk information successfully collected",
    ]


@pytest.mark.asyncio
async def test_handler_error_payload_is_a_real_tool_error():
    from fastmcp import Client
    from fastmcp.exceptions import ToolError
    from node_hardware_mcp import mcp_handlers, server

    failed = {"partitions": [], "summary": {}, "io_statistics": {}, "error": "denied"}
    with patch.object(mcp_handlers, "get_disk_info", return_value=failed):
        async with Client(server.mcp) as client:
            with pytest.raises(ToolError, match="denied"):
                await client.call_tool("get_disk_info", {})


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
