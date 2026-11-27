"""Native MCP tools expose plain data, not a nested display envelope."""

import json
import pytest
from fastmcp import Client
from node_hardware_mcp.server import mcp


@pytest.mark.asyncio
async def test_disk_result_is_structured_and_preserves_source_keys(monkeypatch):
    from node_hardware_mcp import mcp_handlers

    data = {"partitions": [], "summary": {"total_size": 100, "total_free": 40}}
    monkeypatch.setattr(mcp_handlers, "get_disk_info", lambda: data)
    async with Client(mcp) as client:
        result = await client.call_tool("get_disk_info", {})
    record = result.structured_content
    assert record["success"] is True
    assert record["data"] == data
    assert record["summary"]["total_partitions"] == 0
    assert "content" not in record
    assert json.loads(result.content[0].text) == record
