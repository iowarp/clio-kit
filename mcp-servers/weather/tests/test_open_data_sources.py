"""An ECMWF run gone from ECMWF's server is read from ECMWF's cloud mirrors, and said."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import pytest

from weather_mcp.implementation import fetch
from weather_mcp.implementation.common import WeatherToolError


def _client(available: set[str]) -> type:
    class Client:
        def __init__(self, source: str, model: str, **_retries: Any) -> None:
            self.source = source

        def retrieve(self, target: str, **request: Any) -> Any:
            if self.source not in available:
                raise RuntimeError(f"404 Client Error: Not Found ({self.source})")
            return types.SimpleNamespace(datetime="2026-11-06T00:00:00")

    return Client


def _install(monkeypatch: pytest.MonkeyPatch, available: set[str]) -> None:
    module = types.ModuleType("ecmwf.opendata")
    module.Client = _client(available)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ecmwf.opendata", module)


def test_an_old_run_comes_from_the_first_mirror_that_has_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, {"aws"})
    _reply, used, url, failures = fetch._retrieve_open_data("ifs", {}, tmp_path / "x")
    assert used == "the AWS mirror" and "amazonaws" in url
    assert [f.split(":")[0] for f in failures] == [
        "ECMWF's server",
        "the Google Cloud mirror",
    ]


def test_the_latest_run_comes_from_ecmwf_without_a_note(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, {"ecmwf", "aws", "google"})
    _reply, used, _url, failures = fetch._retrieve_open_data("ifs", {}, tmp_path / "x")
    assert used == "ECMWF's server" and failures == []


def test_a_run_no_source_has_fails_with_every_reason(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _install(monkeypatch, set())
    with pytest.raises(WeatherToolError) as raised:
        fetch._retrieve_open_data("ifs", {}, tmp_path / "x")
    text = str(raised.value)
    assert (
        "ECMWF's server" in text
        and "AWS mirror" in text
        and "Google Cloud mirror" in text
    )
