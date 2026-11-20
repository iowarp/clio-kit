from __future__ import annotations

from collections.abc import Iterator

import pytest

from clio_agentic_search.api.app import reset_app_state


@pytest.fixture(autouse=True)
def _test_client_host(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test clients send Host: testserver or test; test_api_access.py covers the check."""
    monkeypatch.setenv("CLIO_ALLOWED_HOSTS", "testserver,test")
    monkeypatch.delenv("CLIO_API_TOKEN", raising=False)


@pytest.fixture(autouse=True)
def _isolated_app_state() -> Iterator[None]:
    reset_app_state()
    yield
    reset_app_state()
