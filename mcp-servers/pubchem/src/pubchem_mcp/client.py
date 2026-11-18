"""A small PubChem PUG REST client: rate-limited, retried, with typed errors.

PubChem asks for at most 5 requests per second (and 400 per minute); every request
goes through one limiter that keeps 0.21 s between request starts. A 503 (server busy),
504/500 timeout, or a network error is retried with exponential backoff; after the last
try the call fails with ``pubchem_unavailable``. A 404 is ``not_found``.
"""

from __future__ import annotations

import asyncio
import time
import urllib.parse
from typing import Any

import httpx

from .errors import PubChemError

PUG = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
AUTOCOMPLETE = "https://pubchem.ncbi.nlm.nih.gov/rest/autocomplete/compound"
MIN_INTERVAL_S = 0.21
RETRY_STATUS = {500, 502, 503, 504}
USER_AGENT = "pubchem-mcp/1.0.0 (clio-kit; https://github.com/iowarp/clio-kit)"


def quote(text: str) -> str:
    """Percent-encode a name for a URL path segment (``/`` included)."""
    return urllib.parse.quote(text, safe="")


class PubChemClient:
    """One shared HTTP client; ``transport`` lets tests serve recorded responses."""

    def __init__(
        self,
        transport: httpx.AsyncBaseTransport | None = None,
        retries: int = 3,
        backoff_s: float = 0.5,
        timeout_s: float = 30.0,
    ) -> None:
        self._transport = transport
        self.retries = retries
        self.backoff_s = backoff_s
        self.timeout_s = timeout_s
        self._lock = asyncio.Lock()
        self._last = 0.0
        self.requests = 0

    async def _wait_turn(self) -> None:
        async with self._lock:
            delay = self._last + MIN_INTERVAL_S - time.monotonic()
            if delay > 0:
                await asyncio.sleep(delay)
            self._last = time.monotonic()

    async def get(self, url: str, params: dict[str, Any] | None = None) -> httpx.Response:
        """GET with rate limit and retries; 404 raises ``not_found``."""
        last_error = ""
        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self.timeout_s,
            headers={"User-Agent": USER_AGENT},
            follow_redirects=True,
        ) as http:
            for attempt in range(self.retries + 1):
                if attempt:
                    await asyncio.sleep(self.backoff_s * 2 ** (attempt - 1))
                await self._wait_turn()
                self.requests += 1
                try:
                    response = await http.get(url, params=params)
                except httpx.HTTPError as exc:
                    last_error = f"{type(exc).__name__}: {exc}"
                    continue
                if response.status_code == 404:
                    raise PubChemError("not_found", _fault(response), url=str(response.url))
                if response.status_code == 400:
                    raise PubChemError(
                        "bad_request",
                        f"PubChem refused the request: {_fault(response)}",
                        url=str(response.url),
                    )
                if response.status_code in RETRY_STATUS:
                    last_error = f"HTTP {response.status_code}: {_fault(response)}"
                    continue
                if response.status_code >= 400:
                    raise PubChemError(
                        "pubchem_unavailable",
                        f"PubChem answered HTTP {response.status_code}: {_fault(response)}",
                        url=str(response.url),
                    )
                return response
        raise PubChemError(
            "pubchem_unavailable",
            f"PubChem did not answer after {self.retries + 1} tries ({last_error}). "
            "It may be busy or unreachable from here; try again in a minute.",
            url=url,
        )

    async def json(self, url: str, params: dict[str, Any] | None = None) -> Any:
        return (await self.get(url, params)).json()

    async def text(self, url: str, params: dict[str, Any] | None = None) -> str:
        return (await self.get(url, params)).text


def _fault(response: httpx.Response) -> str:
    """PubChem's own words for an error: its Fault message and details."""
    try:
        fault = response.json().get("Fault", {})
    except ValueError:
        return response.text.strip().replace("\n", " ")[:300] or response.reason_phrase
    parts = [fault.get("Message", "")] + list(fault.get("Details", []))
    return "; ".join(p for p in parts if p) or response.reason_phrase
