"""Successful Node/Go builds must survive warm and concurrent starts."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
import time

import pytest

from clio_kit.runtime_build import cached_build


def test_concurrent_and_warm_launches_build_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "clio_kit.runtime_build.runtime_executable", lambda runtime: sys.executable
    )
    project = tmp_path / "source-hash"
    project.mkdir()
    calls = []

    def build() -> bool:
        calls.append(1)
        time.sleep(0.05)
        (project / "node_modules").mkdir()
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(cached_build, "node", project, build) for _ in range(2)]
        assert all(future.result() for future in futures)
    assert cached_build("node", project, build)
    assert len(calls) == 1


def test_failed_build_is_retried_without_a_completion_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "clio_kit.runtime_build.runtime_executable", lambda runtime: sys.executable
    )
    project = tmp_path / "source-hash"
    project.mkdir()
    assert not cached_build("go", project, lambda: False)
    assert not list(tmp_path.glob("*.built.json"))

    def build() -> bool:
        (project / "bin").mkdir()
        (project / "bin" / "server").write_text("built")
        return True

    assert cached_build("go", project, build)
    assert cached_build(
        "go", project, lambda: pytest.fail("successful build was repeated")
    )
