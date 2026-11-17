"""Synthetic layers and rasters with known answers."""

from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "artifacts"
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(root))
    return root
