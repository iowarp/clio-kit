"""The front's HTTP client for the node half's loopback port."""

from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .errors import ParaViewError
from .transport import Transport

# The link is loopback only: a client's HTTP_PROXY must not carry it, since an
# egress proxy cannot reach this machine's 127.0.0.1 and answers 502.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
#: Seconds the node half holds each ask about a running call before it says how
#: the call goes; a cancel is seen within this.
JOB_WAIT = 5.0
#: Seconds past ``JOB_WAIT`` before an ask with no answer means the node half holds
#: its loop. This is a check for a stall, not a limit on the call.
ANSWER_GRACE = 30.0


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
            with _OPENER.open(request, timeout=timeout) as response:
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

    def cancel(self, job: str) -> None:
        """Ask the node half to stop a call; a node that cannot be reached has
        stopped it already."""
        try:
            self._request("POST", "/clio/cancel", {"job": job}, timeout=10.0)
        except (LinkError, ParaViewError):
            pass

    def command(
        self,
        name: str,
        args: dict[str, Any],
        by: str = "agent",
        stop: threading.Event | None = None,
        progress: Callable[[dict[str, Any]], None] | None = None,
        wait: float = JOB_WAIT,
    ) -> Any:
        """Run one command and return its result, however long it takes.

        The call runs on the node half under its own id; this asks how it goes every
        ``wait`` seconds and tells ``progress``. There is no limit on its length: it
        ends with its result, when ``stop`` is set (the call is cancelled there too),
        or when the node half is gone (refused or reset). A node half that took the
        request but holds its loop, as a long render does, is busy, not gone: that is
        waited for, and said.
        """
        job = uuid.uuid4().hex
        body = {"command": name, "args": args, "by": by, "job": job, "wait": wait}
        heard = time.monotonic()
        while True:
            if stop is not None and stop.is_set():
                self.cancel(job)
                raise ParaViewError(
                    "cancelled", f"{name} was cancelled before it ended."
                )
            try:
                payload = self._request(
                    "POST", "/clio/command", body, timeout=wait + ANSWER_GRACE
                )
            except LinkError as exc:
                if not exc.busy:
                    raise
                if progress is not None:
                    progress(
                        {
                            "note": "ParaView has not answered for "
                            f"{time.monotonic() - heard:.0f} s: it is busy (a long "
                            "render holds it), not gone, so this waits"
                        }
                    )
                continue
            heard = time.monotonic()
            if isinstance(payload, dict) and payload.get("pending"):
                if progress is not None:
                    progress(dict(payload.get("progress") or {}))
                continue
            return payload["result"]


@dataclass
class Link:
    """A live attachment: the transport, the forwarded port and the client."""

    transport: Transport
    folder: str
    local_port: int
    client: NodeClient
    node: dict[str, Any]
    pvpython: str
    paraview: str
    built_from: str | None = None
    built_now: bool = False
    uploaded_now: bool = False
    timings: dict[str, float] = field(default_factory=dict)
    slot: str = "node"

    @property
    def token(self) -> str:
        return str(self.node["token"])
