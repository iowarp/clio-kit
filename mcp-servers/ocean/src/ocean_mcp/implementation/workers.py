"""Run the copyleft workers (marEx, py-eddy-tracker) in their own environments.

The server never imports marEx (AGPL-3.0-or-later) or py-eddy-tracker
(GPL-3.0). Each lives in a locked environment beside its worker script under
``ocean_mcp/workers/<name>/``; the first call builds it with ``uv sync
--frozen`` and later calls reuse it. A request is a JSON file; the worker
prints one JSON line with its result.
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

from .common import OceanToolError

WORKERS = {
    "marex": {"package": "marEx 5.0.0", "licence": "AGPL-3.0-or-later"},
    "eddy": {"package": "pyEddyTracker 3.6.1", "licence": "GPL-3.0"},
}


def worker_dir(name: str) -> Path:
    return Path(str(resources.files("ocean_mcp").joinpath("workers", name)))


def _uv() -> str:
    uv = os.environ.get("OCEAN_MCP_UV") or shutil.which("uv")
    if not uv:
        raise OceanToolError(
            "backend_unavailable",
            "the marEx and py-eddy-tracker workers are built with uv, which is not on PATH "
            "(set OCEAN_MCP_UV to its path)",
        )
    return uv


def _environment_root(name: str) -> Path:
    """Where the worker's virtual environment lives (outside the package)."""
    base = os.environ.get("OCEAN_MCP_WORKER_ENVS")
    root = (
        Path(base).expanduser()
        if base
        else Path(tempfile.gettempdir()) / "ocean-mcp-workers"
    )
    return root / name


def run_worker(name: str, request: dict[str, Any], timeout_s: float) -> dict[str, Any]:
    """Build the worker environment if needed, run it, return its JSON result."""
    if name not in WORKERS:
        raise OceanToolError("bad_input", f"unknown worker {name!r}")
    project = worker_dir(name)
    env = dict(os.environ)
    env["UV_PROJECT_ENVIRONMENT"] = str(_environment_root(name))
    env.pop("VIRTUAL_ENV", None)
    # The worker pins its own interpreter (.python-version); the server's must not leak in.
    env.pop("UV_PYTHON", None)
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
        raise OceanToolError(
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
        raise OceanToolError(
            "timeout",
            f"the {WORKERS[name]['package']} worker did not finish in {timeout_s:.0f} s",
        ) from exc
    finally:
        Path(request_path).unlink(missing_ok=True)
    lines = [ln for ln in done.stdout.splitlines() if ln.startswith("{")]
    if done.returncode != 0 or not lines:
        tail = (done.stderr or done.stdout).strip().splitlines()[-12:]
        raise OceanToolError(
            "worker_failed",
            f"{WORKERS[name]['package']} worker failed: " + " | ".join(tail),
        )
    payload = json.loads(lines[-1])
    payload["worker"] = {"name": name, **WORKERS[name]}
    return payload
