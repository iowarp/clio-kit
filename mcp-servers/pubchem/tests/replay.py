"""Recorded PubChem responses: a replaying httpx transport and the script that records them.

``fixtures/pubchem_responses.json`` maps ``GET <url>`` to the status, content type and
body PubChem returned. Re-record with ``uv run python -m tests.replay`` (needs network).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import httpx

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "pubchem_responses.json"


def key(request: httpx.Request) -> str:
    return f"{request.method} {request.url}"


class ReplayTransport(httpx.AsyncBaseTransport):
    """Serve recorded responses; an unrecorded request fails the test loudly."""

    def __init__(self, extra: dict[str, Any] | None = None) -> None:
        self.recorded: dict[str, Any] = json.loads(FIXTURES.read_text(encoding="utf-8"))
        self.recorded.update(extra or {})
        self.seen: list[str] = []

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        k = key(request)
        self.seen.append(k)
        if k not in self.recorded:
            raise AssertionError(f"no recorded response for {k}; re-record the fixtures")
        entry = self.recorded[k]
        if entry.get("raise"):
            raise httpx.ConnectError(entry["raise"], request=request)
        return httpx.Response(
            entry["status"],
            headers={"content-type": entry["content_type"]},
            content=entry["body"].encode("utf-8"),
            request=request,
        )


class RecordTransport(httpx.AsyncBaseTransport):
    """Pass requests to the network and keep each response."""

    def __init__(self, store: dict[str, Any]) -> None:
        self.inner = httpx.AsyncHTTPTransport()
        self.store = store

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self.inner.handle_async_request(request)
        body = (await response.aread()).decode("utf-8")
        self.store[key(request)] = {
            "status": response.status_code,
            "content_type": response.headers.get("content-type", ""),
            "body": body,
        }
        return httpx.Response(
            response.status_code,
            headers={"content-type": response.headers.get("content-type", "")},
            content=body.encode("utf-8"),
            request=request,
        )


# What the offline tests replay: (call, argument).
SCENARIOS = [
    ("resolve_name", "xylene"),
    ("resolve_name", "p-xylene"),
    ("resolve_name", "carvone"),
    ("resolve_name", "asprin"),
    ("get_compound", 2244),
    ("get_compound", 999999999),
    ("fetch_structure", 7809),
    ("fetch_structure", 23978),
]


async def record(out_dir: Path) -> None:
    from pubchem_mcp import compounds
    from pubchem_mcp.client import PubChemClient
    from pubchem_mcp.errors import PubChemError

    store: dict[str, Any] = {}
    client = PubChemClient(transport=RecordTransport(store))
    for call, arg in SCENARIOS:
        try:
            if call == "fetch_structure":
                await compounds.fetch_structure(client, arg, str(out_dir))
            else:
                await getattr(compounds, call)(client, arg)
        except PubChemError as exc:
            print(f"{call}({arg!r}) -> {exc.text()}")
    FIXTURES.parent.mkdir(parents=True, exist_ok=True)
    FIXTURES.write_text(json.dumps(store, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    print(f"recorded {len(store)} responses to {FIXTURES}")


if __name__ == "__main__":
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        asyncio.run(record(Path(tmp)))
