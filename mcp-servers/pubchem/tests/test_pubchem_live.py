"""Against the live PubChem API; skipped when it cannot be reached."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from pubchem_mcp import compounds
from pubchem_mcp.client import PUG, PubChemClient

pytestmark = pytest.mark.integration


def _reachable() -> bool:
    try:
        return httpx.get(f"{PUG}/compound/cid/2244/property/Title/TXT", timeout=10).is_success
    except httpx.HTTPError:
        return False


skip = pytest.mark.skipif(not _reachable(), reason="PubChem is not reachable from here")


@skip
async def test_live_xylene_candidates() -> None:
    data = await compounds.resolve_name(PubChemClient(), "xylene")
    assert data["status"] == "ambiguous"
    assert {7237, 7929, 7809} <= {c["cid"] for c in data["candidates"]}


@skip
async def test_live_p_xylene_and_structure(tmp_path: Path) -> None:
    client = PubChemClient()
    data = await compounds.resolve_name(client, "p-xylene")
    assert data["status"] == "resolved" and data["compound"]["cid"] == 7809
    result = await compounds.fetch_structure(client, 7809, str(tmp_path))
    assert result["has_3d"] and result["structure"]["n_atoms"] == 18
