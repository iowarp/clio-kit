"""query_arcgis_features reaches only public hosts and follows no cross-host redirect."""

from __future__ import annotations

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from geo_mcp.implementation import arcgis
from geo_mcp.implementation.public_http import BlockedHostError, public_addresses

_FEATURES = {"features": [], "fields": []}


@pytest.fixture(autouse=True)
def artifacts_in_tmp(monkeypatch, tmp_path):
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(tmp_path))
    monkeypatch.delenv("GEO_ALLOW_PRIVATE_HOSTS", raising=False)


@pytest.fixture
def server():
    """A loopback ArcGIS stand-in that records each request path."""
    seen: list[str] = []
    redirect_to: list[str] = []

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            seen.append(self.path)
            if redirect_to:
                self.send_response(302)
                self.send_header("Location", redirect_to[0])
                self.end_headers()
                return
            body = json.dumps(_FEATURES).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args) -> None:
            return None

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield httpd.server_address[1], seen, redirect_to
    httpd.shutdown()


def query(url: str):
    return asyncio.run(
        arcgis.query_arcgis_features(feature_service_url=url, output_path="out.geojson")
    )


@pytest.mark.parametrize("host", ["127.0.0.1", "127.1", "2130706433", "0x7f000001", "localhost"])
def test_loopback_in_any_spelling_is_refused_before_a_request(server, host):
    port, seen, _ = server
    with pytest.raises(arcgis.ArcGISQueryError, match="GEO_ALLOW_PRIVATE_HOSTS"):
        query(f"http://{host}:{port}/arcgis/rest/services/S/FeatureServer/0/query")
    assert seen == []


def test_private_hosts_are_reachable_when_allowed(server, monkeypatch):
    port, seen, _ = server
    monkeypatch.setenv("GEO_ALLOW_PRIVATE_HOSTS", "true")
    query(f"http://127.0.0.1:{port}/S/FeatureServer/0/query")
    assert seen and seen[0].startswith("/S/FeatureServer/0/query?")


def test_a_redirect_to_another_host_is_not_followed(server, monkeypatch):
    port, seen, redirect_to = server
    monkeypatch.setenv("GEO_ALLOW_PRIVATE_HOSTS", "true")
    redirect_to.append(f"http://127.0.0.2:{port}/elsewhere/query")
    with pytest.raises(arcgis.ArcGISQueryError, match="another host"):
        query(f"http://127.0.0.1:{port}/S/FeatureServer/0/query")
    assert all(not path.startswith("/elsewhere") for path in seen)


@pytest.mark.parametrize("host", ["127.1", "localhost", "10.1.2.3", "169.254.169.254", "::1"])
def test_non_public_addresses_are_refused(host):
    with pytest.raises(BlockedHostError):
        asyncio.run(public_addresses(host, 80))


def test_the_tool_is_marked_as_reaching_the_network():
    from geo_mcp.server import mcp

    async def annotations():
        tool = await mcp.get_tool("query_arcgis_features")
        return tool.annotations

    found = asyncio.run(annotations())
    assert found.readOnlyHint is False
    assert found.openWorldHint is True
