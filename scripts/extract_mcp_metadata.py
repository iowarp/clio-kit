#!/usr/bin/env python3
"""Extract MCP server metadata via FastMCP 3.0 async API.

Run from within a server directory:
    cd mcp-servers/compression && uv run python ../../scripts/extract_mcp_metadata.py

Outputs JSON to stdout with tools, resources, prompts, annotations, and tags.

A server listed in mcp-server-versions.toml [upstream] runs a third-party server
as published and has no ``mcp`` object to import, so it is started from this
environment and asked over the MCP stdio protocol instead, with the table's
``probe-environment``.
"""

import asyncio
import importlib
import json
import logging
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import IO, Any, Callable, cast

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]


def find_server_module() -> str:
    """Determine the server module name from pyproject.toml entry point."""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        print("ERROR: No pyproject.toml found", file=sys.stderr)
        sys.exit(1)

    with open(pyproject_path, "rb") as f:
        data = tomllib.load(f)

    scripts = data.get("project", {}).get("scripts", {})
    if not scripts:
        print("ERROR: No [project.scripts] found", file=sys.stderr)
        sys.exit(1)

    entry_point = next(iter(scripts.values()))
    return entry_point.split(":")[0]


REPO_ROOT = Path(__file__).resolve().parents[1]
SERVER_VERSIONS = REPO_ROOT / "mcp-server-versions.toml"
PROTOCOL_VERSION = "2025-06-18"
PROBE_TIMEOUT_SECONDS = 180.0


def upstream_entry(server_name: str) -> dict[str, Any] | None:
    """This server's [upstream] row, when it runs a third-party server as published."""
    with open(SERVER_VERSIONS, "rb") as handle:
        upstream = tomllib.load(handle).get("upstream", {})
    row = upstream.get(server_name)
    return row if isinstance(row, dict) else None


class StdioSession:
    """A minimal JSON-RPC client for one MCP server started over stdio."""

    def __init__(self, argv: list[str], env: dict[str, str]) -> None:
        self.child = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            env=env,
            text=True,
            encoding="utf-8",
        )
        self.lines: queue.Queue[str | None] = queue.Queue()
        self.next_id = 0
        threading.Thread(
            target=self._read, args=(cast(IO[str], self.child.stdout),), daemon=True
        ).start()

    def _read(self, stream: IO[str]) -> None:
        for line in stream:
            self.lines.put(line)
        self.lines.put(None)

    def send(self, message: dict[str, Any]) -> None:
        stdin = cast(IO[str], self.child.stdin)
        stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        stdin.flush()

    def request(self, method: str, params: dict[str, Any] | None = None) -> Any:
        self.next_id += 1
        request_id = self.next_id
        self.send({"id": request_id, "method": method, "params": params or {}})
        while True:
            line = self.lines.get(timeout=PROBE_TIMEOUT_SECONDS)
            if line is None:
                raise RuntimeError(
                    f"server exited (code {self.child.poll()}) before answering {method}"
                )
            message = json.loads(line)
            if message.get("id") != request_id:
                continue
            if "error" in message:
                raise RuntimeError(f"{method} failed: {message['error']}")
            return message["result"]

    def list_all(self, method: str, key: str) -> list[dict[str, Any]]:
        items: list[dict[str, Any]] = []
        cursor: str | None = None
        seen: set[str] = set()
        while True:
            page = self.request(method, {"cursor": cursor} if cursor else None)
            items.extend(page.get(key, []))
            cursor = page.get("nextCursor")
            if not cursor:
                return items
            if cursor in seen:
                raise RuntimeError(f"{method} returned a repeated pagination cursor")
            seen.add(cursor)

    def close(self) -> None:
        self.child.kill()
        self.child.wait(timeout=30)


def _tags(item: dict[str, Any]) -> list[str]:
    meta = item.get("_meta") or {}
    fastmcp = meta.get("fastmcp") or meta.get("_fastmcp") or {}
    return sorted(fastmcp.get("tags") or [])


def extract_over_stdio(module_path: str, env: dict[str, str]) -> dict[str, Any]:
    """Start the server's entry point and read its metadata over the protocol."""
    session = StdioSession(
        [sys.executable, "-c", f"from {module_path} import main; main()"],
        {**os.environ, **env},
    )
    try:
        initialized = session.request(
            "initialize",
            {
                "protocolVersion": PROTOCOL_VERSION,
                "capabilities": {},
                "clientInfo": {"name": "clio-kit-metadata", "version": "1"},
            },
        )
        session.send({"method": "notifications/initialized"})
        capabilities = initialized.get("capabilities", {})
        tools = (
            session.list_all("tools/list", "tools") if "tools" in capabilities else []
        )
        resources: list[dict[str, Any]] = []
        templates: list[dict[str, Any]] = []
        if "resources" in capabilities:
            resources = session.list_all("resources/list", "resources")
            templates = session.list_all(
                "resources/templates/list", "resourceTemplates"
            )
        prompts = (
            session.list_all("prompts/list", "prompts")
            if "prompts" in capabilities
            else []
        )
    finally:
        session.close()
    return {
        "name": initialized.get("serverInfo", {}).get("name", ""),
        "instructions": initialized.get("instructions") or "",
        "tools": [
            {
                "name": t["name"],
                "description": t.get("description") or "",
                "annotations": t.get("annotations") or {},
                "tags": _tags(t),
            }
            for t in tools
        ],
        "resources": [
            {
                "uri": r["uri"],
                "name": r["name"],
                "description": r.get("description") or "",
            }
            for r in resources
        ],
        "resource_templates": [
            {
                "uri_template": t["uriTemplate"],
                "name": t["name"],
                "description": t.get("description") or "",
            }
            for t in templates
        ],
        "prompts": [
            {"name": p["name"], "description": p.get("description") or ""}
            for p in prompts
        ],
    }


async def extract(module_path: str) -> dict[str, Any]:
    """Import the server and extract all metadata."""
    module = importlib.import_module(module_path)
    metadata_profile = getattr(module, "MCP_METADATA_PROFILE", None)
    if metadata_profile is not None:
        profile_applier = getattr(module, "apply_tool_profile", None)
        if not isinstance(metadata_profile, str) or not callable(profile_applier):
            raise RuntimeError("invalid MCP_METADATA_PROFILE contract")
        cast(Callable[[str], None], profile_applier)(metadata_profile)
    mcp = getattr(module, "mcp")

    tools = await mcp.list_tools()
    resources = await mcp.list_resources()
    templates = await mcp.list_resource_templates()
    prompts = await mcp.list_prompts()

    return {
        "name": mcp.name,
        "instructions": mcp.instructions or "",
        "tools": [
            {
                "name": t.name,
                "description": t.description or "",
                "annotations": t.annotations.model_dump(by_alias=True)
                if t.annotations
                else {},
                "tags": sorted(t.tags) if hasattr(t, "tags") and t.tags else [],
            }
            for t in tools
        ],
        "resources": [
            {
                "uri": str(r.uri),
                "name": r.name,
                "description": r.description or "",
            }
            for r in resources
        ],
        "resource_templates": [
            {
                "uri_template": str(t.uri_template),
                "name": t.name,
                "description": t.description or "",
            }
            for t in templates
        ],
        "prompts": [
            {
                "name": p.name,
                "description": p.description or "",
            }
            for p in prompts
        ],
    }


def main() -> None:
    # Suppress all logging and stray prints during import
    logging.disable(logging.CRITICAL)
    # Redirect stdout during import, then restore for JSON output
    old_stdout = sys.stdout
    sys.stdout = open(os.devnull, "w")
    try:
        module_path = find_server_module()
        upstream = upstream_entry(Path.cwd().name)
        if upstream is None:
            result = asyncio.run(extract(module_path))
        else:
            probe_env = upstream.get("probe-environment", {})
            result = extract_over_stdio(
                module_path, {str(k): str(v) for k, v in probe_env.items()}
            )
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout
        logging.disable(logging.NOTSET)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
