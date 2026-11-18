"""The tools through an in-memory MCP client, with DECIMER replaced by a stub."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from decimer_mcp import recognize
from decimer_mcp.server import mcp

from .conftest import CAFFEINE


def _stub(smiles: str, downloaded: bool = False):
    def predict(path: str, hand_drawn: bool) -> tuple[str, list[tuple[str, float]], dict]:
        return (
            smiles,
            [("C", 0.99), ("N", 0.62), ("1", 0.97)],
            {
                "cold_start": downloaded,
                "load_s": 95.0 if downloaded else 0.0,
                "predict_s": 1.2,
                "downloaded": ["DECIMER"] if downloaded else [],
                "downloaded_mb": 299 if downloaded else 0,
            },
        )

    return predict


async def _call(name: str, args: dict[str, Any]) -> Any:
    async with Client(mcp) as client:
        return await client.call_tool(name, args)


async def test_tools_listed_with_paths_meta() -> None:
    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {"image_to_smiles", "crop_image"}
    paths = (tools["image_to_smiles"].meta or {})["ai.iowarp.clio/paths"]
    assert paths["image_path"] == {"machine": "local", "access": "read"}
    crop_paths = (tools["crop_image"].meta or {})["ai.iowarp.clio/paths"]
    assert crop_paths["output_dir"]["access"] == "write"
    assert tools["image_to_smiles"].annotations.read_only_hint is True


async def test_valid_reading(caffeine_png: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(recognize, "_predict", _stub(CAFFEINE, downloaded=True))
    result = await _call("image_to_smiles", {"image_path": str(caffeine_png)})
    data = result.structured_content
    assert data["rdkit"]["valid"] is True
    assert data["rdkit"]["formula"] == "C8H10N4O2"
    assert data["rdkit"]["canonical_smiles"] == "Cn1c(=O)c2c(ncn2C)n(C)c1=O"
    assert data["confidence"]["min"] == 0.62
    text = result.content[0].text
    assert "C8H10N4O2" in text and "194.19" in text
    assert "about 299 MB from Zenodo" in text
    assert "model's reading" in text and "Ketcher" in text


async def test_invalid_smiles_keeps_raw(
    caffeine_png: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(recognize, "_predict", _stub("C1CC(C"))
    result = await _call("image_to_smiles", {"image_path": str(caffeine_png)})
    data = result.structured_content
    assert data["smiles"] == "C1CC(C"
    assert data["rdkit"]["valid"] is False
    assert "RDKit cannot use it" in result.content[0].text


async def test_unreadable_image_is_tool_error(tmp_path: Path) -> None:
    bad = tmp_path / "x.png"
    bad.write_text("hello")
    with pytest.raises(ToolError, match=r"\[unreadable_image\]$"):
        await _call("image_to_smiles", {"image_path": str(bad)})


async def test_crop_tool(caffeine_png: Path, artifacts: Path) -> None:
    result = await _call(
        "crop_image", {"image_path": str(caffeine_png), "box": [0, 0, 0.5, 0.5], "unit": "fraction"}
    )
    assert Path(result.structured_content["image"]["path"]).is_file()
    assert "image_to_smiles" in result.content[0].text


def test_check_smiles_overlong() -> None:
    assert recognize.check_smiles("C" * 5000)["valid"] is False
    assert recognize.check_smiles("")["valid"] is False
