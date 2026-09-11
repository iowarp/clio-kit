"""Language-independent MCP discovery using a real stdio client session."""

from __future__ import annotations

import asyncio
import os
from typing import Any


async def inspect_stdio(
    command: str, args: list[str], *, timeout: float = 180
) -> dict[str, Any]:
    try:
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        from mcp.types import PaginatedRequestParams
    except ImportError as exc:
        raise RuntimeError(
            "Install clio-kit[verification] to inspect real MCP sessions"
        ) from exc

    async def inspect() -> dict[str, Any]:
        async with stdio_client(
            StdioServerParameters(command=command, args=args, env=dict(os.environ))
        ) as (read, write):
            async with ClientSession(read, write) as session:
                initialized = await session.initialize()
                result: dict[str, Any] = {
                    "server": initialized.serverInfo.model_dump(by_alias=True),
                    "tools": [],
                }
                for key, method, enabled in (
                    ("tools", session.list_tools, initialized.capabilities.tools),
                    (
                        "resources",
                        session.list_resources,
                        initialized.capabilities.resources,
                    ),
                    ("prompts", session.list_prompts, initialized.capabilities.prompts),
                ):
                    if not enabled:
                        continue
                    cursor = None
                    seen: set[str] = set()
                    result[key] = []
                    while True:
                        page = await method(
                            params=PaginatedRequestParams(cursor=cursor)
                            if cursor
                            else None
                        )
                        dumped = page.model_dump(mode="json", by_alias=True)
                        result[key].extend(dumped[key])
                        cursor = dumped.get("nextCursor")
                        if not cursor:
                            break
                        if cursor in seen:
                            raise ValueError(
                                "MCP returned a repeated pagination cursor"
                            )
                        seen.add(cursor)
                return result

    return await asyncio.wait_for(inspect(), timeout=timeout)
