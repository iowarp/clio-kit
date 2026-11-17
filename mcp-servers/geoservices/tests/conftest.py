"""Local fixtures: a threaded HTTP server serving canned documents, and an artifacts dir."""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlsplit

import pytest

Route = Callable[[dict[str, str]], tuple[int, str, bytes]]


class Site:
    """Routes by path; each route receives the query parameters and returns a response."""

    def __init__(self) -> None:
        self.routes: dict[str, Route] = {}
        self.requests: list[str] = []
        self.base = ""

    def json(self, path: str, body: Any) -> None:
        self.routes[path] = lambda q: (
            200,
            "application/json",
            json.dumps(body).encode(),
        )

    def bytes(
        self, path: str, body: bytes, kind: str = "application/octet-stream"
    ) -> None:
        self.routes[path] = lambda q: (200, kind, body)


@pytest.fixture
def site() -> Iterator[Site]:
    state = Site()

    class Handler(BaseHTTPRequestHandler):
        def _serve(self, head: bool = False) -> None:
            parts = urlsplit(self.path)
            state.requests.append(self.path)
            route = state.routes.get(parts.path)
            if route is None:
                self.send_response(404)
                self.end_headers()
                return
            query = {k: v[0] for k, v in parse_qs(parts.query).items()}
            status, kind, body = route(query)
            ranged = self.headers.get("Range")
            if ranged and status == 200:
                start, _, end = ranged.removeprefix("bytes=").partition("-")
                first = int(start)
                last = min(int(end) if end else len(body) - 1, len(body) - 1)
                chunk = body[first : last + 1]
                self.send_response(206)
                self.send_header("Content-Range", f"bytes {first}-{last}/{len(body)}")
                body = chunk
            else:
                self.send_response(status)
            self.send_header("Content-Type", kind)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Accept-Ranges", "bytes")
            self.end_headers()
            if not head:
                self.wfile.write(body)

        def do_GET(self) -> None:
            self._serve()

        def do_HEAD(self) -> None:
            self._serve(head=True)

        def log_message(self, *args: Any) -> None:
            return

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    state.base = f"http://127.0.0.1:{server.server_address[1]}"
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()


@pytest.fixture(autouse=True)
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "artifacts"
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(root))
    monkeypatch.delenv("HTTP_PROXY", raising=False)
    monkeypatch.delenv("HTTPS_PROXY", raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    monkeypatch.setenv("no_proxy", "127.0.0.1,localhost")
    return root
