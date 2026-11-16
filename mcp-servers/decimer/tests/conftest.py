"""Shared fixtures: structure drawings rendered with RDKit, so no image files are checked in."""

from __future__ import annotations

from pathlib import Path

import pytest

CAFFEINE = "CN1C=NC2=C1C(=O)N(C(=O)N2C)C"
ASPIRIN = "CC(=O)OC1=CC=CC=C1C(=O)O"


def render(smiles: str, path: Path, size: int = 400) -> Path:
    """Draw ``smiles`` as a PNG the way a paper's clean 2D depiction looks."""
    from rdkit import Chem
    from rdkit.Chem import Draw

    Draw.MolToFile(Chem.MolFromSmiles(smiles), str(path), size=(size, size))
    return path


@pytest.fixture
def caffeine_png(tmp_path: Path) -> Path:
    return render(CAFFEINE, tmp_path / "caffeine.png")


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "artifacts"
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(folder))
    return folder
