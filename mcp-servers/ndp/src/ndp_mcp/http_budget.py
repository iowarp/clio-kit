"""NDP API requests inside one bounded time budget, with typed failures (#351).

The client used to give every attempt its own 30 s timeout and retry three
times on any exception, so a hung NDP endpoint burned 93 s or more before an
untyped error surfaced, and a caller with its own ceiling (clio-agent's is
180 s) usually timed out first with nothing to go on.
"""

import asyncio
import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class NDPRequestError(Exception):
    """A failed NDP API request, typed so a caller can tell a hung remote from an
    unreachable one, and see how much of the request budget it spent.

    ``kind`` is ``remote_timeout`` (no response within the budget),
    ``remote_unreachable`` (connection failed), ``remote_error`` (HTTP 5xx),
    ``http_error`` (HTTP 4xx) or ``request_failed`` (anything else).
    """

    def __init__(
        self, kind: str, method: str, url: str, detail: str, attempts: int, elapsed_s: float
    ):
        self.kind = kind
        self.attempts = attempts
        self.elapsed_s = elapsed_s
        super().__init__(
            f"NDP {kind}: {method} {url}: {detail} (attempts={attempts}, elapsed={elapsed_s:.1f}s)"
        )


def _request_budget_s() -> float:
    """Total seconds one NDP request may take, retries included (NDP_REQUEST_BUDGET_S)."""
    try:
        return max(1.0, float(os.getenv("NDP_REQUEST_BUDGET_S", "30")))
    except ValueError:
        return 30.0


async def budgeted_request(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    json_data: dict[str, Any] | None,
    *,
    budget_s: float,
    max_retries: int,
    retry_delay: float,
) -> dict[str, Any]:
    """Make one HTTP request within ``budget_s``, retrying only fast failures.

    A timeout is not retried: it has already spent the budget, and a hung
    remote would only hang again. Connection failures, HTTP 5xx and other
    errors are retried with backoff while attempts and budget remain, and each
    retry is logged.

    Returns:
        The decoded JSON body.

    Raises:
        ValueError: for an HTTP method other than GET or POST.
        NDPRequestError: typed by kind, with the attempt count and elapsed time.
    """
    method = method.upper()
    if method not in ("GET", "POST"):
        raise ValueError(f"Unsupported HTTP method: {method}")
    start = time.monotonic()
    attempt = 0
    while True:
        attempt += 1
        remaining = budget_s - (time.monotonic() - start)
        try:
            response = await asyncio.wait_for(
                _send(method, url, params, json_data, remaining), remaining
            )
            response.raise_for_status()
            return response.json()  # type: ignore[no-any-return]
        except (asyncio.TimeoutError, TimeoutError, httpx.TimeoutException) as e:
            raise NDPRequestError(
                "remote_timeout",
                method,
                url,
                f"no response within the {budget_s:.0f}s request budget",
                attempt,
                time.monotonic() - start,
            ) from e
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            kind = "remote_error" if status >= 500 else "http_error"
            detail = f"HTTP {status}: {e.response.text[:500]}"
            if status < 500:
                raise NDPRequestError(
                    kind, method, url, detail, attempt, time.monotonic() - start
                ) from e
            cause: Exception = e
        except httpx.TransportError as e:
            kind, detail, cause = "remote_unreachable", str(e) or type(e).__name__, e
        except Exception as e:
            kind, detail, cause = "request_failed", f"Request failed: {e}", e
        delay = retry_delay * (2 ** (attempt - 1))
        elapsed = time.monotonic() - start
        if attempt >= max_retries or elapsed + delay >= budget_s:
            raise NDPRequestError(kind, method, url, detail, attempt, elapsed) from cause
        logger.warning(
            "NDP %s %s failed (%s: %s); attempt %d/%d, retrying in %.1fs",
            method,
            url,
            kind,
            detail,
            attempt,
            max_retries,
            delay,
        )
        await asyncio.sleep(delay)


async def _send(
    method: str,
    url: str,
    params: dict[str, Any] | None,
    json_data: dict[str, Any] | None,
    budget_s: float,
) -> httpx.Response:
    """One attempt, with httpx's own timeouts bounded by what remains."""
    timeout = httpx.Timeout(budget_s, connect=min(10.0, budget_s))
    async with httpx.AsyncClient(timeout=timeout) as client:
        if method == "GET":
            return await client.get(url, params=params)
        return await client.post(url, params=params, json=json_data)
