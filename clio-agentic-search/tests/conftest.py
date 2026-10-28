from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest

from clio_agentic_search.api.app import reset_app_state


@pytest.fixture(autouse=True)
def _isolated_app_state(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Keep default namespace scans and writable storage inside this test's directory."""
    local = tmp_path / "default-local"
    objects = tmp_path / "default-objects"
    local.mkdir()
    objects.mkdir()
    monkeypatch.setenv("CLIO_LOCAL_ROOT", str(local))
    monkeypatch.setenv("CLIO_OBJECT_STORE_ROOT", str(objects))
    monkeypatch.setenv("CLIO_STORAGE_PATH", str(tmp_path / "search.duckdb"))
    reset_app_state()
    yield
    reset_app_state()
