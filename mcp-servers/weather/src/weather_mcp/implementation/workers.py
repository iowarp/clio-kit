"""Run the Earth2Studio worker in its own environment.

torch and the model packages are large, so they live in a locked
environment beside the worker script under ``weather_mcp/workers/earth2/``;
the first call builds it with ``uv sync --frozen`` (gigabytes, minutes) and
later calls reuse it. A request is a JSON file; the worker prints one JSON
line with its result.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from importlib import resources
from pathlib import Path
from typing import Any

from .common import WeatherToolError

WORKERS = {
    "earth2": {"package": "earth2studio 0.20.0", "licence": "Apache-2.0"},
}


def worker_dir(name: str) -> Path:
    return Path(str(resources.files("weather_mcp").joinpath("workers", name)))


def _uv() -> str:
    uv = os.environ.get("WEATHER_MCP_UV") or shutil.which("uv")
    if not uv:
        raise WeatherToolError(
            "backend_unavailable",
            "the Earth2Studio worker is built with uv, which is not on PATH "
            "(set WEATHER_MCP_UV to its path)",
        )
    return uv


def _environment_root(name: str) -> Path:
    """Where the worker's virtual environment lives (outside the package)."""
    base = os.environ.get("WEATHER_MCP_WORKER_ENVS")
    root = (
        Path(base).expanduser()
        if base
        else Path(tempfile.gettempdir()) / "weather-mcp-workers"
    )
    return root / name


def run_worker(name: str, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Build the worker environment if needed, run it, return its JSON result."""
    if name not in WORKERS:
        raise WeatherToolError("bad_input", f"unknown worker {name!r}")
    project = worker_dir(name)
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(_environment_root(name))
    env.pop("VIRTUAL_ENV", None)
    uv = _uv()
    sync = subprocess.run(
        [uv, "sync", "--frozen", "--project", str(project), "--quiet"],
        capture_output=True,
        text=True,
        env=env,
        timeout=1800,
        check=False,
    )
    if sync.returncode != 0:
        raise WeatherToolError(
            "backend_unavailable",
            f"could not build the {WORKERS[name]['package']} worker environment: {sync.stderr.strip()[-600:]}",
        )
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as handle:
        json.dump(request, handle)
        request_path = handle.name
    try:
        done = subprocess.run(
            [
                uv,
                "run",
                "--frozen",
                "--no-sync",
                "--project",
                str(project),
                "python",
                "-I",
                str(project / "worker.py"),
                request_path,
            ],
            capture_output=True,
            text=True,
            env=env,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise WeatherToolError(
            "timeout",
            f"the {WORKERS[name]['package']} worker did not finish in {timeout_s:.0f} s",
        ) from exc
    finally:
        Path(request_path).unlink(missing_ok=True)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if lines and json.loads(lines[-1]).get("status") == "error":
        raise WeatherToolError(
            "worker_failed",
            f"{WORKERS[name]['package']}: {json.loads(lines[-1])['error']}",
        )
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-12:]
        raise WeatherToolError(
            "worker_failed",
            f"{WORKERS[name]['package']} worker failed: " + " | ".join(tail),
        )
    payload = json.loads(lines[-1])
    payload["worker"] = {"name": name, **WORKERS[name]}
    return payload
