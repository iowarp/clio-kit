"""DECIMER itself reading RDKit-rendered drawings back.

Runs when the model weights are already cached, or when DECIMER_MCP_DOWNLOAD=1 allows
the first-use download (about 300 MB from Zenodo). A clean RDKit depiction is the easy
case; scanned figures from papers are harder.
"""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path

import pytest

from decimer_mcp import images, recognize

from .conftest import ASPIRIN, CAFFEINE, render

pytestmark = pytest.mark.integration


def _available() -> str | None:
    if importlib.util.find_spec("DECIMER") is None:
        return "DECIMER is not installed in this environment"
    if not recognize._model_present(False) and os.environ.get("DECIMER_MCP_DOWNLOAD") != "1":
        return (
            f"DECIMER weights are not cached in {recognize.cache_dir()}; "
            "set DECIMER_MCP_DOWNLOAD=1 to download them (about 300 MB)"
        )
    return None


@pytest.mark.skipif(_available() is not None, reason=str(_available()))
@pytest.mark.parametrize(("name", "smiles"), [("caffeine", CAFFEINE), ("aspirin", ASPIRIN)])
def test_reads_rendered_structure(tmp_path: Path, name: str, smiles: str) -> None:
    from rdkit import Chem

    image = images.check_image(str(render(smiles, tmp_path / f"{name}.png")))
    data = recognize.image_to_smiles(image, hand_drawn=False)
    assert data["rdkit"]["valid"], data
    assert data["rdkit"]["canonical_smiles"] == Chem.MolToSmiles(Chem.MolFromSmiles(smiles))
