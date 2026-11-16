"""Worker environments: shipped project files, lock hash, uv lookup, commands."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ase_mcp import envs
from ase_mcp.errors import AseError


def test_project_files_and_hash() -> None:
    files = envs.project_files()
    assert set(files) == {"pyproject.toml", "uv.lock"}
    assert b"[project.optional-dependencies]" in files["pyproject.toml"]
    assert len(envs.lock_hash()) == 12 and envs.lock_hash() == envs.lock_hash()


def test_extras_and_conflict_declared() -> None:
    import tomllib

    data = tomllib.loads(envs.project_files()["pyproject.toml"].decode())
    assert set(data["project"]["optional-dependencies"]) == {"xtb", "mace", "uma"}
    assert [{"extra": "mace"}, {"extra": "uma"}] in data["tool"]["uv"]["conflicts"]


def test_uv_missing(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("UV", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    with pytest.raises(AseError) as info:
        envs.uv_executable()
    assert info.value.reason == "uv_missing"


def test_status_and_commands(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(envs.ENV_DIR_VAR, str(tmp_path / "cache"))
    st = envs.status("xtb")
    assert st["prepared"] is False and st["download"]
    cmd, env = envs.worker_command(None, tmp_path / "input.json")
    assert cmd[0] == sys.executable and cmd[1:3] == ["-m", "ase_mcp.worker"]
    assert Path(env["PYTHONPATH"]) == envs.PACKAGE_DIR.parent
    cmd, env = envs.worker_command("mace", tmp_path / "input.json")
    assert "mace-" in cmd[0]
    code = Path(env["PYTHONPATH"])
    assert (code / "ase_mcp" / "worker.py").is_file()
    assert code.parent.parent == tmp_path / "cache"


def test_ready_marker(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv(envs.ENV_DIR_VAR, str(tmp_path))
    d = envs.env_dir("xtb")
    d.mkdir(parents=True)
    (d / "ready.json").write_text('{"build_seconds": 1.0}')
    assert envs.status("xtb")["prepared"] is True
    assert envs.ensure("xtb")["built_now"] is False
