"""Rate limit, retries and typed errors of the PubChem client, with fake transports."""

from __future__ import annotations

import time

import httpx
import pytest

from pubchem_mcp import client as client_module
from pubchem_mcp import compounds
from pubchem_mcp.client import PubChemClient
from pubchem_mcp.errors import PubChemError

URL = "https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/cid/2244/property/Title/JSON"
OK = {"PropertyTable": {"Properties": [{"CID": 2244, "Title": "Aspirin"}]}}
BUSY = {"Fault": {"Code": "PUGREST.ServerBusy", "Message": "Too many requests or server too busy"}}


def _sequence(*responses: tuple[int, dict]) -> tuple[httpx.MockTransport, list[int]]:
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        status, body = responses[min(len(calls), len(responses) - 1)]
        calls.append(status)
        return httpx.Response(status, json=body)

    return httpx.MockTransport(handler), calls


@pytest.fixture(autouse=True)
def _fast(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "MIN_INTERVAL_S", 0.0)


async def test_busy_then_ok_is_retried() -> None:
    transport, calls = _sequence((503, BUSY), (503, BUSY), (200, OK))
    client = PubChemClient(transport=transport, backoff_s=0.0)
    assert (await client.json(URL)) == OK
    assert calls == [503, 503, 200]


async def test_always_busy_is_unavailable() -> None:
    transport, calls = _sequence((503, BUSY))
    client = PubChemClient(transport=transport, retries=2, backoff_s=0.0)
    with pytest.raises(PubChemError) as err:
        await client.json(URL)
    assert err.value.reason == "pubchem_unavailable"
    assert "server too busy" in str(err.value)
    assert len(calls) == 3


async def test_network_error_is_unavailable() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route to host", request=request)

    client = PubChemClient(transport=httpx.MockTransport(handler), retries=1, backoff_s=0.0)
    with pytest.raises(PubChemError) as err:
        await client.json(URL)
    assert err.value.reason == "pubchem_unavailable"
    assert err.value.text().endswith("[pubchem_unavailable]")


async def test_404_is_not_found_without_retry() -> None:
    transport, calls = _sequence((404, {"Fault": {"Message": "No CID found"}}))
    client = PubChemClient(transport=transport, backoff_s=0.0)
    with pytest.raises(PubChemError) as err:
        await client.json(URL)
    assert err.value.reason == "not_found" and calls == [404]


async def test_unknown_name_everywhere_is_not_found() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if "autocomplete" in request.url.path:
            return httpx.Response(200, json={"status": {"code": 0}, "total": 0})
        return httpx.Response(404, json={"Fault": {"Message": "No CID found"}})

    client = PubChemClient(transport=httpx.MockTransport(handler), backoff_s=0.0)
    with pytest.raises(PubChemError) as err:
        await compounds.resolve_name(client, "notachemicalxyz")
    assert err.value.reason == "not_found"
    assert "Check the spelling" in str(err.value)


async def test_several_exact_cids_are_ambiguous() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if "autocomplete" in path:
            return httpx.Response(200, json={"total": 0})
        if "/name/" in path:
            if request.url.params.get("name_type") == "word":
                return httpx.Response(404, json={"Fault": {"Message": "No CID found"}})
            return httpx.Response(200, json={"IdentifierList": {"CID": [11, 22]}})
        if "/synonyms/" in path:
            return httpx.Response(404, json={"Fault": {"Message": "No synonyms"}})
        rows = [
            {"CID": cid, "Title": f"T{cid}", "MolecularFormula": "CH4", "MolecularWeight": "16.04"}
            for cid in (11, 22)
        ]
        return httpx.Response(200, json={"PropertyTable": {"Properties": rows}})

    client = PubChemClient(transport=httpx.MockTransport(handler), backoff_s=0.0)
    data = await compounds.resolve_name(client, "gas")
    assert data["status"] == "ambiguous"
    assert [c["cid"] for c in data["candidates"]] == [11, 22]
    assert "maps the name 'gas' to 2 different compounds" in data["reason"]
    assert data["candidates"][0]["found_by"] == ["exact name"]


async def test_rate_limit_spaces_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(client_module, "MIN_INTERVAL_S", 0.21)
    transport, _ = _sequence((200, OK))
    client = PubChemClient(transport=transport)
    start = time.monotonic()
    for _ in range(6):
        await client.json(URL)
    assert time.monotonic() - start >= 5 * 0.21 - 0.02  # never more than 5 per second


def test_sdf_atom_count() -> None:
    sdf = "7809\n  -OEChem-\n\n 18 18  0     0  0  0  0  0  0999 V2000\n"
    assert compounds.sdf_atom_count(sdf) == 18
