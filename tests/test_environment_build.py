"""Self-healing of half-built cached MCP environments."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import clio_kit
from clio_kit.environment_build import (
    create_environment_atomically,
    discard_partial_environment,
    environment_interpreter,
)
from clio_kit.environment_locks import EnvironmentBuildLock, _FileLock, locks_dir


def _poison(environment: Path) -> None:
    """Recreate what an interrupted first ``uv sync`` left on a Windows host."""
    (environment / "Scripts").mkdir(parents=True)
    (environment / ".gitignore").write_text("*\n", encoding="utf-8")
    (environment / "CACHEDIR.TAG").write_text("Signature\n", encoding="utf-8")


def _complete(environment: Path) -> None:
    interpreter = environment_interpreter(environment)
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"")
    (environment / "pyvenv.cfg").write_text("home = x\n", encoding="utf-8")


def _environment(cache_root: Path) -> Path:
    return cache_root / "mcp-environments" / f"web-{'a' * 24}"


def test_discard_removes_environment_without_interpreter(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    _poison(environment)
    events: list[dict[str, Any]] = []

    assert discard_partial_environment(
        tmp_path, environment, server="web", emit=lambda e: events.append(dict(e))
    )

    assert not environment.exists()
    assert not list(environment.parent.glob("*.tmp"))
    assert events[0]["event"] == "partial_environment_discarded"


def test_discard_keeps_complete_and_empty_environments(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    environment.mkdir(parents=True)
    assert not discard_partial_environment(
        tmp_path, environment, server="web", emit=lambda _e: None
    )
    _complete(environment)
    assert not discard_partial_environment(
        tmp_path, environment, server="web", emit=lambda _e: None
    )
    assert environment_interpreter(environment).is_file()


def test_discard_keeps_environment_another_launcher_holds(tmp_path: Path) -> None:
    environment = _environment(tmp_path)
    _poison(environment)
    lock_dir = locks_dir(tmp_path, environment.name)
    lock_dir.mkdir(parents=True)
    other = _FileLock(lock_dir / f"{os.getpid() + 1}.lock")
    other.acquire()
    try:
        assert not discard_partial_environment(
            tmp_path, environment, server="web", emit=lambda _e: None
        )
    finally:
        other.release()
    assert (environment / "CACHEDIR.TAG").exists()


def test_create_renames_complete_venv_into_place(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)
    environment.mkdir(parents=True)
    seen: list[Path] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        staging = Path(cmd[-1])
        seen.append(staging)
        _complete(staging)
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert create_environment_atomically(
        environment, uv="uv", project=tmp_path, child_environment={}
    )
    assert seen[0].parent == environment.parent and seen[0] != environment
    assert environment_interpreter(environment).is_file()
    assert not seen[0].exists()


def test_failed_create_leaves_no_partial_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    environment = _environment(tmp_path)

    def interrupted(
        cmd: list[str], **_kwargs: Any
    ) -> subprocess.CompletedProcess[bytes]:
        _poison(Path(cmd[-1]))
        return subprocess.CompletedProcess(cmd, 1, b"", b"interrupted")

    monkeypatch.setattr(subprocess, "run", interrupted)

    assert not create_environment_atomically(
        environment, uv="uv", project=tmp_path, child_environment={}
    )
    assert not environment.exists()
    assert not list(environment.parent.glob("*.tmp"))


def test_build_lock_is_exclusive_and_times_out(tmp_path: Path) -> None:
    with EnvironmentBuildLock(tmp_path, "web-x") as first:
        assert first
        with EnvironmentBuildLock(tmp_path, "web-x", timeout=0) as second:
            assert not second
    with EnvironmentBuildLock(tmp_path, "web-x", timeout=0) as again:
        assert again


def test_launch_heals_poisoned_environment_before_sync(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    server_path = tmp_path / "servers" / "web"
    (server_path / "src").mkdir(parents=True)
    (server_path / "pyproject.toml").write_text(
        "[project]\nname = 'web-mcp'\nversion = '1.0.0'\n", encoding="utf-8"
    )
    (server_path / "uv.lock").write_text("version = 1\n", encoding="utf-8")
    (server_path / "src" / "server.py").write_text("VALUE = 1\n", encoding="utf-8")
    cache_root = tmp_path / "cache"
    monkeypatch.setenv("CLIO_KIT_CACHE_DIR", str(cache_root))
    monkeypatch.setattr(clio_kit, "uv_command", lambda: "uv")
    identity = clio_kit.locked_server_project_identity(server_path)
    environment = clio_kit._locked_server_environment_path(
        server_path, project_sha256=identity["project_sha256"]
    )
    _poison(environment)
    commands: list[list[str]] = []

    def fake_run(cmd: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[bytes]:
        commands.append(cmd)
        if "venv" in cmd:
            _complete(Path(cmd[-1]))
        if "sync" in cmd:
            assert environment_interpreter(environment).is_file()
        return subprocess.CompletedProcess(cmd, 0, b"", b"")

    monkeypatch.setattr(subprocess, "run", fake_run)

    clio_kit._run_locked_local_server(server_path, "web-mcp", (), os.environ.copy())

    assert not (environment / "CACHEDIR.TAG").exists()
    assert [cmd[1] for cmd in commands[:2]] == ["venv", "sync"]
