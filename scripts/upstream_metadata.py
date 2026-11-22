"""What a wrapped upstream server advertises, read from the running server over stdio.

``extract_mcp_metadata.py`` reads a server module's ``mcp`` object. A server that runs
a third-party MCP server as published ([upstream] in mcp-server-versions.toml) never
imports that package, so it has no ``mcp``. It exposes ``command(argv)`` and
``metadata_env()`` instead, and the extractor builds an :class:`UpstreamServer` from
them: it starts the upstream command, asks it for its tools, resources, resource
templates and prompts with plain JSON-RPC, and offers the answers through the same
small interface a FastMCP instance does. The metadata is therefore exactly what a
client of the server would be told. One copy serves every wrapper, so no wrapper
carries more than its own launcher.
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import subprocess
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

#: The protocol version and client identity sent with ``initialize``.
INITIALIZE = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "clio-kit-metadata", "version": "1"},
}

#: The list requests, keyed by the capability that must be advertised for each.
LISTS = {
    "tools": ("tools", "tools/list", "tools"),
    "resources": ("resources", "resources/list", "resources"),
    "resource_templates": (
        "resources",
        "resources/templates/list",
        "resourceTemplates",
    ),
    "prompts": ("prompts", "prompts/list", "prompts"),
}


#: A server's ``initialize`` result and its lists, keyed as in :data:`LISTS`.
Answer = tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]


@dataclass(frozen=True)
class Annotations:
    """A tool's annotations as sent, offering pydantic's ``model_dump`` spelling."""

    values: Mapping[str, Any]

    def model_dump(self, *, by_alias: bool = True, **_: Any) -> dict[str, Any]:
        """Return the annotations exactly as the server sent them (already aliased)."""
        return dict(self.values)


@dataclass(frozen=True)
class Tool:
    """One advertised tool."""

    name: str
    description: str
    annotations: Annotations | None
    tags: frozenset[str] = field(default_factory=frozenset)


@dataclass(frozen=True)
class Resource:
    """One advertised resource."""

    uri: str
    name: str
    description: str


@dataclass(frozen=True)
class ResourceTemplate:
    """One advertised resource template."""

    uri_template: str
    name: str
    description: str


@dataclass(frozen=True)
class Prompt:
    """One advertised prompt."""

    name: str
    description: str


def _query(argv: Sequence[str], env: Mapping[str, str], timeout: float) -> Answer:
    """Start ``argv`` as an MCP stdio server; return its ``initialize`` result and lists."""
    child = subprocess.Popen(
        list(argv),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, **env},
        text=True,
        encoding="utf-8",
    )
    assert child.stdin is not None and child.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    stdout = child.stdout

    def read() -> None:
        for line in stdout:
            lines.put(line)

    threading.Thread(target=read, daemon=True).start()
    next_id = 0

    def request(method: str, params: Mapping[str, Any] | None = None) -> dict[str, Any]:
        nonlocal next_id
        next_id += 1
        message: dict[str, Any] = {"jsonrpc": "2.0", "id": next_id, "method": method}
        if params is not None:
            message["params"] = dict(params)
        assert child.stdin is not None
        child.stdin.write(json.dumps(message) + "\n")
        child.stdin.flush()
        while True:
            reply = json.loads(lines.get(timeout=timeout))
            if reply.get("id") != next_id:
                continue
            if "error" in reply:
                raise RuntimeError(f"{method} failed: {reply['error']}")
            result: dict[str, Any] = reply["result"]
            return result

    try:
        info = request("initialize", INITIALIZE)
        child.stdin.write(
            json.dumps({"jsonrpc": "2.0", "method": "notifications/initialized"}) + "\n"
        )
        child.stdin.flush()
        capabilities = info.get("capabilities") or {}
        lists: dict[str, list[dict[str, Any]]] = {}
        for key, (capability, method, items) in LISTS.items():
            lists[key] = []
            if capability not in capabilities:
                continue
            cursor: str | None = None
            while True:
                page = request(method, {"cursor": cursor} if cursor else None)
                lists[key].extend(page.get(items) or [])
                cursor = page.get("nextCursor")
                if not cursor:
                    break
        return info, lists
    finally:
        child.kill()
        child.wait(timeout=30)


def _annotations(item: Mapping[str, Any]) -> Annotations | None:
    """A listed tool's annotations, if it sent any."""
    values = item.get("annotations")
    return Annotations(values) if values else None


class UpstreamServer:
    """The upstream server's advertised metadata, with FastMCP's listing interface."""

    def __init__(
        self,
        argv: Sequence[str],
        env: Mapping[str, str] | None = None,
        timeout: float = 120.0,
    ) -> None:
        """Describe the server started by ``argv`` with ``env`` added to the environment."""
        self._argv = list(argv)
        self._env = dict(env or {})
        self._timeout = timeout
        self._answer: Answer | None = None

    def _ask(self) -> Answer:
        if self._answer is None:
            self._answer = _query(self._argv, self._env, self._timeout)
        return self._answer

    async def _lists(self) -> dict[str, list[dict[str, Any]]]:
        return (await asyncio.to_thread(self._ask))[1]

    @property
    def name(self) -> str:
        """The name the upstream server reports."""
        return str(self._ask()[0].get("serverInfo", {}).get("name", ""))

    @property
    def instructions(self) -> str:
        """The instructions the upstream server sends with ``initialize``."""
        return str(self._ask()[0].get("instructions") or "")

    async def list_tools(self) -> list[Tool]:
        """The advertised tools."""
        return [
            Tool(
                name=item["name"],
                description=item.get("description") or "",
                annotations=_annotations(item),
                tags=frozenset(item.get("_meta", {}).get("tags") or ()),
            )
            for item in (await self._lists())["tools"]
        ]

    async def list_resources(self) -> list[Resource]:
        """The advertised resources."""
        return [
            Resource(item["uri"], item.get("name", ""), item.get("description") or "")
            for item in (await self._lists())["resources"]
        ]

    async def list_resource_templates(self) -> list[ResourceTemplate]:
        """The advertised resource templates."""
        return [
            ResourceTemplate(
                item["uriTemplate"], item.get("name", ""), item.get("description") or ""
            )
            for item in (await self._lists())["resource_templates"]
        ]

    async def list_prompts(self) -> list[Prompt]:
        """The advertised prompts."""
        return [
            Prompt(item["name"], item.get("description") or "")
            for item in (await self._lists())["prompts"]
        ]
