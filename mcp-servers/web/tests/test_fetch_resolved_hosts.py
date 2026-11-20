"""fetch refuses non-public addresses in every spelling, at the connection.

These tests use a real loopback listener and no HTTP mock, so the request goes
through the public-only network backend.
"""

from __future__ import annotations

import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
from fastmcp import Client

from web_mcp import server
from web_mcp.server import REASON_BLOCKED_HOST, mcp


@pytest.fixture
def listener():
    hits: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            hits.append(self.path)
            body = b"internal-only-body"
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            return None

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd.server_address[1], hits
    httpd.shutdown()


@pytest.mark.asyncio
@pytest.mark.parametrize("host", ["127.1", "2130706433", "0x7f000001", "localhost."])
async def test_other_spellings_of_loopback_are_refused(listener, host, monkeypatch) -> None:
    port, hits = listener
    monkeypatch.setattr(server.settings, "allow_private_hosts", False)
    async with Client(mcp) as client:
        with pytest.raises(Exception) as excinfo:
            await client.call_tool("fetch", {"target": f"http://{host}:{port}/secret"})
    assert REASON_BLOCKED_HOST in str(excinfo.value)
    assert "internal-only-body" not in str(excinfo.value)
    assert hits == []


@pytest.mark.asyncio
async def test_the_opt_in_reaches_a_private_host(listener, monkeypatch) -> None:
    port, hits = listener
    monkeypatch.setattr(server.settings, "allow_private_hosts", True)
    async with Client(mcp) as client:
        result = await client.call_tool("fetch", {"target": f"http://127.1:{port}/ok"})
    assert hits == ["/ok"]
    assert "internal-only-body" in str(result.content)
