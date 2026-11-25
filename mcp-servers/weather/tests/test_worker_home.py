"""The Earth2Studio worker gets a writable home (a sandboxed server's home is read-only)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest

from weather_mcp.implementation import workers


def test_the_worker_runs_with_a_home_of_its_own(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Q19 rerun: earth2studio failed writing '/root/.config/intake-esgf'."""
    monkeypatch.setenv("WEATHER_MCP_WORKER_ENVS", str(tmp_path / "envs"))
    monkeypatch.setenv("HOME", "/read-only-home")
    seen: list[dict[str, str]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(kwargs["env"])
        out = json.dumps({"status": "success"}) if "sync" not in args else ""
        return subprocess.CompletedProcess(args, 0, out, "")

    monkeypatch.setattr(workers.subprocess, "run", run)
    name = next(iter(workers.WORKERS))
    try:
        workers.run_worker(name, {}, timeout_s=10)
    except workers.WeatherToolError:
        pass  # the stand-in result need not satisfy the worker's reader
    env = seen[-1]
    home = Path(env["HOME"])
    assert home != Path("/read-only-home") and home.is_dir()
    assert env["XDG_CONFIG_HOME"].startswith(env["HOME"])
    assert home.parent == tmp_path / "envs"
