from __future__ import annotations

import shutil
from pathlib import Path

import pytest

REAL = all(shutil.which(n) for n in ("avogadro2", "Xvfb", "x11vnc"))


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(tmp_path / "ws"))
    return tmp_path / "ws"


@pytest.fixture
def water(tmp_path: Path) -> Path:
    p = tmp_path / "water.xyz"
    p.write_text("3\nwater\nO 0 0 0.117\nH 0 0.757 -0.469\nH 0 -0.757 -0.469\n")
    return p
