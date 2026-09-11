"""Language-independent MCP discovery using a real stdio client session."""

from __future__ import annotations

import asyncio
import os
from typing import Any


async def inspect_stdio(
    command: str, args: list[str], *, timeout: float = 180
) -> dict[str, Any]:
    try:
        from mcp import Client, StdioServerParameters
    except ImportError as exc:
        raise RuntimeError(
            "Install clio-kit[verification] to inspect real MCP sessions"
        ) from exc

    async def inspect() -> dict[str, Any]:
        async with Client(
            StdioServerParameters(command=command, args=args, env=dict(os.environ))
        ) as client:
            capabilities = client.server_capabilities
            result: dict[str, Any] = {
                "server": client.server_info.model_dump(by_alias=True)
                if client.server_info is not None
                else {},
                "protocol_version": client.protocol_version,
                "tools": [],
            }
            for key, method, enabled in (
                ("tools", client.list_tools, capabilities.tools),
                ("resources", client.list_resources, capabilities.resources),
                ("prompts", client.list_prompts, capabilities.prompts),
            ):
                if enabled is None:
                    continue
                cursor = None
                seen: set[str] = set()
                result[key] = []
                while True:
                    page = await method(cursor=cursor)
                    dumped = page.model_dump(mode="json", by_alias=True)
                    result[key].extend(dumped[key])
                    cursor = page.next_cursor
                    if not cursor:
                        break
                    if cursor in seen:
                        raise ValueError("MCP returned a repeated pagination cursor")
                    seen.add(cursor)
            return result

    return await asyncio.wait_for(inspect(), timeout=timeout)
