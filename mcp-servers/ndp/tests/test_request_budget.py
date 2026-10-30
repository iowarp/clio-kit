"""NDP requests fail fast and typed inside one bounded budget (issue #351).

Before: a 30 s timeout per attempt and three attempts on every exception, so a
hung NDP endpoint burned 93 s or more invisibly and the caller's own 180 s
ceiling fired first, with no typed error at all.
"""

import asyncio
import logging
import socket
import time
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastmcp.exceptions import ToolError

from ndp_mcp import server
from ndp_mcp.http_budget import NDPRequestError, _request_budget_s
from ndp_mcp.server import NDPClient


@pytest.fixture
async def hanging_endpoint():
    """An endpoint that accepts connections and never answers."""
    held: list[asyncio.StreamWriter] = []

    async def accept(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        held.append(writer)
        await reader.read(65536)
        await asyncio.sleep(3600)

    srv = await asyncio.start_server(accept, "127.0.0.1", 0)
    port = srv.sockets[0].getsockname()[1]
    yield f"http://127.0.0.1:{port}"
    for writer in held:
        writer.close()
    srv.close()


def _closed_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


@pytest.mark.asyncio
async def test_hanging_remote_is_a_typed_timeout_within_the_budget(hanging_endpoint):
    client = NDPClient(hanging_endpoint)
    client.request_budget_s = 1.0

    start = time.monotonic()
    with pytest.raises(NDPRequestError) as exc_info:
        await client._make_request("GET", "/search")
    elapsed = time.monotonic() - start

    assert exc_info.value.kind == "remote_timeout"
    assert exc_info.value.attempts == 1
    assert elapsed < 2.5, elapsed


@pytest.mark.asyncio
async def test_hanging_remote_reaches_the_tool_as_a_typed_error(hanging_endpoint):
    client = NDPClient(hanging_endpoint)
    client.request_budget_s = 1.0
    with patch.object(server, "ndp_client", client):
        with pytest.raises(ToolError, match="NDP remote_timeout"):
            await server.list_organizations()


@pytest.mark.asyncio
async def test_unreachable_remote_retries_with_logging_then_fails_typed(caplog):
    client = NDPClient(f"http://127.0.0.1:{_closed_port()}")
    client.retry_delay = 0.01

    with caplog.at_level(logging.WARNING, logger="ndp_mcp.http_budget"):
        with pytest.raises(NDPRequestError) as exc_info:
            await client._make_request("GET", "/organization")

    assert exc_info.value.kind == "remote_unreachable"
    assert exc_info.value.attempts == client.max_retries
    retries = [r for r in caplog.records if "retrying" in r.getMessage()]
    assert len(retries) == client.max_retries - 1


@pytest.mark.asyncio
async def test_retries_stop_when_the_backoff_would_exceed_the_budget():
    client = NDPClient("http://test.example.com")
    client.request_budget_s = 1.0
    client.retry_delay = 5.0  # the first backoff alone outlasts the budget
    sleep = AsyncMock()
    with patch("httpx.AsyncClient") as mock_client, patch("asyncio.sleep", new=sleep):
        mock_client.return_value.__aenter__.return_value.get.side_effect = httpx.ConnectError(
            "refused"
        )
        with pytest.raises(NDPRequestError) as exc_info:
            await client._make_request("GET", "/test")

    assert exc_info.value.attempts == 1
    sleep.assert_not_called()


@pytest.mark.asyncio
async def test_server_errors_are_retried_then_typed():
    client = NDPClient("http://test.example.com")
    error = MagicMock(status_code=503, text="unavailable")
    with patch("httpx.AsyncClient") as mock_client, patch("asyncio.sleep", new=AsyncMock()):
        get = mock_client.return_value.__aenter__.return_value.get
        get.side_effect = httpx.HTTPStatusError("503", request=MagicMock(), response=error)
        with pytest.raises(NDPRequestError) as exc_info:
            await client._make_request("GET", "/test")

    assert exc_info.value.kind == "remote_error"
    assert "HTTP 503" in str(exc_info.value)
    assert get.call_count == client.max_retries


@pytest.mark.parametrize(
    ("value", "expected"), [(None, 30.0), ("12", 12.0), ("0.2", 1.0), ("soon", 30.0)]
)
def test_request_budget_env_override(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("NDP_REQUEST_BUDGET_S", raising=False)
    else:
        monkeypatch.setenv("NDP_REQUEST_BUDGET_S", value)
    assert _request_budget_s() == expected
