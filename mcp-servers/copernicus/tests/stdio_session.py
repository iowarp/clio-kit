"""A minimal MCP client over stdio: start the server, send requests, read the replies.

Written against JSON-RPC directly so the test needs nothing the upstream server does not
already bring, and so it sees exactly the bytes a client would.
"""

from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from collections.abc import Mapping, Sequence
from typing import Any

INITIALIZE = {
    "protocolVersion": "2025-06-18",
    "capabilities": {},
    "clientInfo": {"name": "clio-kit-test", "version": "1"},
}


def session(
    module: str,
    requests: Sequence[Mapping[str, Any]],
    *,
    env: Mapping[str, str] | None = None,
    timeout: float = 120.0,
) -> dict[int, dict[str, Any]]:
    """Run ``module``'s ``main`` as a server and return the replies by id (0: initialize)."""
    child = subprocess.Popen(
        [sys.executable, "-c", f"from {module} import main; main()"],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
        env={**os.environ, **(env or {})},
        text=True,
    )
    assert child.stdin is not None and child.stdout is not None
    lines: queue.Queue[str] = queue.Queue()
    reader = threading.Thread(
        target=lambda: [lines.put(line) for line in child.stdout], daemon=True
    )
    reader.start()

    def send(message: Mapping[str, Any]) -> None:
        assert child.stdin is not None
        child.stdin.write(json.dumps({"jsonrpc": "2.0", **message}) + "\n")
        child.stdin.flush()

    replies: dict[int, dict[str, Any]] = {}
    try:
        send({"id": 0, "method": "initialize", "params": INITIALIZE})
        send({"method": "notifications/initialized"})
        for index, request in enumerate(requests, start=1):
            send({"id": index, **request})
        while len(replies) < len(requests) + 1:
            message = json.loads(lines.get(timeout=timeout))
            if isinstance(message.get("id"), int):
                replies[message["id"]] = message
    finally:
        child.kill()
        child.wait(timeout=30)
    return replies
