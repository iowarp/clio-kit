"""The front's HTTP client for the node half's loopback port."""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any

from .errors import ParaViewError


class LinkError(Exception):
    """The node half could not be reached over the link.

    ``busy`` is true when the node half took the request but did not answer in time:
    it is alive and holding its loop (a long software render does), not gone.
    """

    def __init__(self, message: str, busy: bool = False) -> None:
        super().__init__(message)
        self.busy = busy


def _timed_out(exc: BaseException) -> bool:
    reason = exc.reason if isinstance(exc, urllib.error.URLError) else exc
    return isinstance(reason, TimeoutError)


class NodeClient:
    """Sends scene commands to the node half with its token."""

    def __init__(self, port: int, token: str) -> None:
        self.base = f"http://127.0.0.1:{port}"
        self._token = token

    def _request(
        self, method: str, path: str, body: Any = None, timeout: float = 30.0
    ) -> Any:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(
            self.base + path,
            data=data,
            method=method,
            headers={
                "Authorization": f"Bearer {self._token}",
                "Content-Type": "application/json",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return json.loads(response.read())
        except urllib.error.HTTPError as exc:
            try:
                payload = json.loads(exc.read())
            except ValueError:
                payload = {}
            error = payload.get("error") or {}
            if exc.code == 401:
                raise LinkError("the node half refused the token") from exc
            raise ParaViewError(
                str(error.get("code") or f"http_{exc.code}"),
                str(error.get("message") or f"The node half answered {exc.code}."),
            ) from exc
        except (urllib.error.URLError, OSError, ValueError) as exc:
            raise LinkError(str(exc), busy=_timed_out(exc)) from exc

    def health(self) -> dict[str, Any]:
        return dict(self._request("GET", "/clio/health", timeout=10.0))

    def ping(self) -> None:
        self._request("POST", "/clio/ping", {}, timeout=10.0)

    def command(
        self, name: str, args: dict[str, Any], timeout: float = 900.0, by: str = "agent"
    ) -> Any:
        payload = self._request(
            "POST",
            "/clio/command",
            {"command": name, "args": args, "by": by},
            timeout=timeout,
        )
        return payload["result"]
