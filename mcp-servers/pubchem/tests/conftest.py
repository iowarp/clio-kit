"""Offline by default: the server's PubChem client replays recorded responses."""

from __future__ import annotations

from pathlib import Path

import pytest

from pubchem_mcp import client as client_module
from pubchem_mcp import server
from pubchem_mcp.client import PubChemClient

from .replay import ReplayTransport


@pytest.fixture
def replay(monkeypatch: pytest.MonkeyPatch) -> ReplayTransport:
    """Point the server at the recorded responses, with no rate-limit pause."""
    transport = ReplayTransport()
    monkeypatch.setattr(client_module, "MIN_INTERVAL_S", 0.0)
    monkeypatch.setattr(server, "CLIENT", PubChemClient(transport=transport, backoff_s=0.0))
    return transport


@pytest.fixture
def artifacts(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "artifacts"
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(folder))
    return folder
