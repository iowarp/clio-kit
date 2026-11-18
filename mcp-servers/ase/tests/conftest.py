"""Shared fixtures. Slow tests (building calculator environments, real xTB/MACE runs)
run only with ``--run-slow`` or ``ASE_MCP_RUN_SLOW=1``."""

from __future__ import annotations

import os
from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--run-slow",
        action="store_true",
        default=False,
        help="run slow tests (build xTB/MACE environments, real calculations)",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if config.getoption("--run-slow") or os.environ.get("ASE_MCP_RUN_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="slow: pass --run-slow (or ASE_MCP_RUN_SLOW=1)")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An output folder; CLIO_KIT_ARTIFACTS points at it so nothing lands in the cwd."""
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(tmp_path))
    return tmp_path


@pytest.fixture
def water(tmp_path: Path) -> Path:
    from ase.build import molecule
    from ase.io import write

    path = tmp_path / "water.xyz"
    write(str(path), molecule("H2O"))
    return path


@pytest.fixture
def ethane(tmp_path: Path) -> Path:
    from ase.build import molecule
    from ase.io import write

    path = tmp_path / "ethane.xyz"
    write(str(path), molecule("C2H6"))
    return path
