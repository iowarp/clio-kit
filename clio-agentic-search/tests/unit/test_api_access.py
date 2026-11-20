"""Who may call the API: browser origins, Host names and the optional token."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from clio_agentic_search.api.app import app
from clio_agentic_search.cli.main import main


@pytest.fixture
def client() -> TestClient:
    return TestClient(app, base_url="http://127.0.0.1:8000")


def test_no_browser_origin_is_granted_by_default(client: TestClient) -> None:
    preflight = client.options(
        "/query",
        headers={
            "Origin": "http://evil.example",
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert "access-control-allow-origin" not in preflight.headers
    response = client.get("/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in response.headers


@pytest.mark.parametrize("host", ["evil.example:8000", "rebound.example"])
def test_an_unexpected_host_is_refused(client: TestClient, host: str) -> None:
    response = client.get("/documents", headers={"Host": host})
    assert response.status_code == 403
    assert "CLIO_ALLOWED_HOSTS" in response.json()["error"]


@pytest.mark.parametrize("host", ["127.0.0.1:8000", "localhost:8000", "[::1]:8000"])
def test_loopback_hosts_are_answered(client: TestClient, host: str) -> None:
    assert client.get("/health", headers={"Host": host}).status_code == 200


def test_a_configured_token_is_required_except_for_health(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("CLIO_API_TOKEN", "s3cret")
    assert client.get("/health").status_code == 200
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"Authorization": "Bearer wrong"}).status_code == 401
    authorized = client.get("/documents", headers={"Authorization": "Bearer s3cret"})
    assert authorized.status_code != 401


def test_serve_refuses_a_network_bind_without_a_token(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.delenv("CLIO_API_TOKEN", raising=False)
    assert main(["serve", "--host", "0.0.0.0"]) == 2
    assert "CLIO_API_TOKEN" in capsys.readouterr().err
