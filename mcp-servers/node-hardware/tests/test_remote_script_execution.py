"""Execute the generated SSH payload through a real shell."""

import json
import subprocess
from node_hardware_mcp.capabilities.remote_node_info import _create_remote_info_script


def test_remote_script_quotes_python_and_decodes_nullable_filters():
    command = _create_remote_info_script(["cpu", "memory", "quoted'component"], None)
    result = subprocess.run(
        command, shell=True, capture_output=True, text=True, check=True, timeout=20
    )
    data = json.loads(result.stdout)
    assert data["cpu"]["logical_cores"] > 0
    assert "system" not in data
    assert "error" not in data


def test_remote_failure_is_an_mcp_error(monkeypatch):
    import asyncio
    from fastmcp import Client
    from node_hardware_mcp import server

    monkeypatch.setattr(
        server.mcp_handlers,
        "get_remote_node_info_handler",
        lambda **kwargs: {
            "isError": True,
            "content": [{"text": "SSH connection refused"}],
        },
    )

    async def exchange():
        async with Client(server.mcp) as client:
            result = await client.call_tool(
                "get_remote_node_info", {"hostname": "localhost"}, raise_on_error=False
            )
        assert result.is_error
        assert "SSH connection refused" in result.content[0].text

    asyncio.run(exchange())
