"""The three tools through an in-memory MCP client, on recorded PubChem responses."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from pubchem_mcp.server import mcp

from .replay import ReplayTransport


async def _call(name: str, args: dict[str, Any]) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, args)


async def test_tool_list() -> None:
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {"resolve_name", "get_compound", "fetch_structure"}
    assert tools["fetch_structure"].meta["ai.iowarp.clio/paths"] == {
        "output_dir": {"machine": "local", "access": "write"}
    }
    assert tools["resolve_name"].annotations.read_only_hint is True


async def test_xylene_is_ambiguous_with_three_isomers(replay: ReplayTransport) -> None:
    result = await _call("resolve_name", {"name": "xylene"})
    data = result.structured_content
    assert data["status"] == "ambiguous"
    cids = [c["cid"] for c in data["candidates"]]
    assert {7237, 7929, 7809} <= set(cids)  # o-, m-, p-xylene
    assert len(cids) == 8  # the default max_candidates
    by_cid = {c["cid"]: c for c in data["candidates"]}
    assert by_cid[7809]["isomeric_smiles"] == "CC1=CC=C(C=C1)C"
    assert by_cid[7809]["formula"] == "C8H10"
    assert by_cid[7809]["stereo"]["atom_centres"] == 0
    assert by_cid[7809]["found_by"]
    text = result.content[0].text
    assert "No compound in PubChem is named exactly 'xylene'" in text
    assert "Ask the person which compound they mean" in text
    assert "O-Xylene" in text and "M-Xylene" in text and "P-Xylene" in text


async def test_p_xylene_resolves(replay: ReplayTransport) -> None:
    result = await _call("resolve_name", {"name": "p-xylene"})
    data = result.structured_content
    assert data["status"] == "resolved"
    assert data["compound"]["cid"] == 7809
    assert data["compound"]["inchikey"] == "URLKBWYHVLBVBO-UHFFFAOYSA-N"
    assert data["compound"]["molecular_weight_g_mol"] == pytest.approx(106.16)
    assert "CID 7809" in result.content[0].text


async def test_racemic_name_reports_undefined_stereo(replay: ReplayTransport) -> None:
    result = await _call("resolve_name", {"name": "carvone"})
    data = result.structured_content
    assert data["status"] == "resolved"
    assert data["compound"]["stereo"]["undefined_atom_centres"] == 1
    assert "stereochemistry is not defined" in result.content[0].text


async def test_misspelling_gives_candidates(replay: ReplayTransport) -> None:
    data = (await _call("resolve_name", {"name": "asprin"})).structured_content
    assert data["status"] == "ambiguous"
    assert 2244 in [c["cid"] for c in data["candidates"]]


async def test_get_compound(replay: ReplayTransport) -> None:
    result = await _call("get_compound", {"cid": 2244})
    data = result.structured_content
    assert data["title"] == "Aspirin"
    assert data["formula"] == "C9H8O4"
    assert any(s.lower() == "acetylsalicylic acid" for s in data["synonyms"])
    assert len(data["synonyms"]) <= 20


async def test_unknown_cid_is_not_found(replay: ReplayTransport) -> None:
    with pytest.raises(ToolError, match=r"no compound with CID 999999999.*\[not_found\]$"):
        await _call("get_compound", {"cid": 999999999})


async def test_fetch_3d_structure(replay: ReplayTransport, artifacts: Path) -> None:
    result = await _call("fetch_structure", {"cid": 7809})
    data = result.structured_content
    s = data["structure"]
    path = Path(s["path"])
    assert path.is_relative_to(artifacts / "chemistry" / "pubchem" / "cid7809-p-xylene")
    assert s["format"] == "sdf" and s["dimension"] == "3d" and s["formula"] == "C8H10"
    assert s["n_atoms"] == 18  # 8 C + 10 H
    assert s["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert data["has_3d"] is True and data["fallback"] is None
    assert Path(data["results_file"]).is_file()
    assert "3D conformer" in result.content[0].text


async def test_fetch_without_3d_says_so(replay: ReplayTransport, tmp_path: Path) -> None:
    result = await _call("fetch_structure", {"cid": 23978, "output_dir": str(tmp_path)})
    data = result.structured_content
    assert data["has_3d"] is False and data["fallback"] == "2d_record"
    assert data["structure"]["dimension"] == "2d"
    assert Path(data["structure"]["path"]).is_relative_to(tmp_path / "chemistry" / "pubchem")
    text = result.content[0].text
    assert "no 3D conformer" in text and "ase.smiles_to_structure" in text


async def test_empty_name_is_bad_request(replay: ReplayTransport) -> None:
    with pytest.raises(ToolError, match=r"\[bad_request\]$"):
        await _call("resolve_name", {"name": "  "})
