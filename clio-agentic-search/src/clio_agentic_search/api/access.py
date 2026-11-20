"""Who may call the HTTP API.

The service usually runs on this computer's loopback interface, which every web
page the user opens can also send requests to. Three checks keep it to its
intended callers:

- **Origins.** No browser origin may read responses unless it is listed in
  ``CLIO_CORS_ORIGINS`` (comma-separated). Without CORS a web page can neither
  read a response nor send the JSON requests the API takes.
- **Host.** The ``Host`` header must name loopback (``localhost``,
  ``127.0.0.1``, ``::1``) or a name listed in ``CLIO_ALLOWED_HOSTS``, which stops
  a page that rebinds its own domain to ``127.0.0.1``.
- **Token.** When ``CLIO_API_TOKEN`` is set, every request except ``/health``
  must send ``Authorization: Bearer <token>``. ``clio serve`` requires it before
  listening on anything but loopback.
"""

from __future__ import annotations

import hmac
import os

from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Receive, Scope, Send

LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})


def _csv(name: str) -> list[str]:
    return [item.strip() for item in os.environ.get(name, "").split(",") if item.strip()]


def cors_origins() -> list[str]:
    """Browser origins allowed to read responses (none unless configured)."""
    return _csv("CLIO_CORS_ORIGINS")


def allowed_hosts() -> frozenset[str]:
    """Host names the API answers to."""
    return LOOPBACK_HOSTS | {host.lower() for host in _csv("CLIO_ALLOWED_HOSTS")}


def api_token() -> str | None:
    """The bearer token every request must carry, if one is configured."""
    return os.environ.get("CLIO_API_TOKEN", "").strip() or None


def is_loopback_host(host: str) -> bool:
    """Whether a bind or ``Host`` name is this computer's loopback interface."""
    host = host.strip().strip("[]").lower()
    return host in LOOPBACK_HOSTS or host.startswith("127.")


def _host_name(header: str) -> str:
    header = header.strip().lower()
    if header.startswith("["):
        return header[1 : header.find("]")] if "]" in header else header
    return header.rsplit(":", 1)[0] if header.count(":") == 1 else header


class AccessMiddleware:
    """Refuse requests to an unexpected ``Host`` or without the configured token."""

    def __init__(self, app: ASGIApp) -> None:
        self._app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return
        headers = {k.decode("latin-1"): v.decode("latin-1") for k, v in scope["headers"]}
        host = _host_name(headers.get("host", ""))
        if not (is_loopback_host(host) or host in allowed_hosts()):
            response = JSONResponse(
                {"error": f"host {host!r} is not allowed; add it to CLIO_ALLOWED_HOSTS"},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        token = api_token()
        if token is not None and scope.get("path") != "/health":
            supplied = headers.get("authorization", "")
            expected = f"Bearer {token}"
            if not hmac.compare_digest(supplied.encode(), expected.encode()):
                response = JSONResponse(
                    {"error": "a valid bearer token is required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer"},
                )
                await response(scope, receive, send)
                return
        await self._app(scope, receive, send)
