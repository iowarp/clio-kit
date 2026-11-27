"""Validate planner fixtures and cleanup; model scientific judgments need review."""

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "planner_eval", ROOT / "evals/claude_planner_eval.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_available_reader_reduces_requested_column_and_rejects_missing_values(tmp_path):
    project = tmp_path / "project"
    runner.prepare(project, "available-reader")
    data = project / "readings.csv"
    data.write_text("time_s,signal_mV\n1000,1\n2000,3\n3000,8\n")
    command = [sys.executable, str(project / "column_mean.py"), str(data), "signal_mV"]
    result = subprocess.run(command, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {
        "count": 3,
        "mean": 4.0,
        "coverage": "all rows",
    }
    data.write_text("time_s,signal_mV\n1000,1\n2000,nan\n")
    assert subprocess.run(command, capture_output=True).returncode != 0


def test_failed_live_run_removes_auth_link_and_installed_plugin(tmp_path, monkeypatch):
    auth_source = tmp_path / "original"
    auth_source.mkdir()
    original = auth_source / ".credentials.json"
    original.write_text("test credential; never sent to a model")
    monkeypatch.setenv("CLAUDE_CONFIG_DIR", str(auth_source))
    args = SimpleNamespace(output=tmp_path / "output", timeout=1, model="test")
    args.output.mkdir()
    commands = []

    def fake_run(argv, **kwargs):
        if argv[0] != "claude":
            raise AssertionError("Unexpected executable")
        commands.append(argv)
        profile = Path(kwargs["env"]["CLAUDE_CONFIG_DIR"])
        if "--agent" in argv:
            assert (profile / ".credentials.json").is_symlink()
            assert argv[argv.index("--effort") + 1] == "medium"
            raise subprocess.TimeoutExpired(argv, 1, output=b"partial trace")
        output = (
            json.dumps(
                [
                    {
                        "id": "clio-agents@clio-kit",
                        "installPath": str(ROOT / "plugins/clio-agents"),
                    }
                ]
            )
            if argv[1:3] == ["plugin", "list"]
            else ""
        )
        return subprocess.CompletedProcess(argv, 0, stdout=output, stderr="")

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    with pytest.raises(subprocess.TimeoutExpired):
        runner.run(args, "missing-reader", 1)
    folder = args.output / "missing-reader-1"
    assert not (folder / "profile/.credentials.json").is_symlink()
    assert original.read_text() == "test credential; never sent to a model"
    assert (folder / "runtime.log").read_bytes() == b"partial trace"
    assert not (folder / "result.json").exists()
    assert any(c[1:3] == ["plugin", "uninstall"] for c in commands)
