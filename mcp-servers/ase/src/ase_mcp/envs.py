"""Worker environments: the server's own one for EMT/NWChem/ORCA, one per extra otherwise.

The clio-kit launcher syncs the server with ``uv sync --frozen --no-dev`` and no
extras, so xTB, MACE and UMA (heavy, and MACE/UMA conflict over e3nn) live in their own
environment, built on first use from the ``pyproject.toml`` and ``uv.lock`` shipped
inside this package: ``uv sync --frozen --no-dev --no-install-project --extra <x>``
into ``<cache>/envs/<x>-<lock hash>``. A file lock makes concurrent first uses build it
once; a ``ready.json`` marker records a finished build. The worker imports this package
from a copy of its source (``<cache>/code/<hash>``) via PYTHONPATH, so the server's own
site-packages never leak into the worker.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from importlib import resources
from pathlib import Path
from typing import Any

from .errors import AseError

ENV_DIR_VAR = "ASE_MCP_ENV_DIR"
PACKAGE_DIR = Path(__file__).resolve().parent

# What a first build downloads, measured on Linux x86-64 with CPU torch (approximate).
APPROX_SIZE = {
    "xtb": "about 30 MB",
    "mace": "about 1.3 GB (CPU PyTorch)",
    "uma": "about 2 GB (CPU PyTorch, fairchem)",
}


def cache_root() -> Path:
    if os.environ.get(ENV_DIR_VAR):
        return Path(os.environ[ENV_DIR_VAR]).expanduser()
    from platformdirs import user_cache_dir

    return Path(user_cache_dir("clio-kit")) / "ase-mcp"


def project_files() -> dict[str, bytes]:
    """The server's pyproject.toml and uv.lock: package data, or the source checkout."""
    try:
        data = resources.files("ase_mcp").joinpath("_project")
        files = {name: data.joinpath(name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
        return files
    except (FileNotFoundError, NotADirectoryError, OSError):
        pass
    checkout = PACKAGE_DIR.parents[1]  # src/ase_mcp -> server folder
    try:
        return {name: (checkout / name).read_bytes() for name in ("pyproject.toml", "uv.lock")}
    except OSError as exc:
        raise AseError(
            "env_build_failed",
            "The server package lacks its pyproject.toml and uv.lock, so calculator "
            "environments cannot be built. Reinstall the ase server.",
        ) from exc


def lock_hash() -> str:
    """Hash of uv.lock: it pins every package of every extra, so it names the build."""
    return hashlib.sha256(project_files()["uv.lock"]).hexdigest()[:12]


def env_dir(extra: str) -> Path:
    return cache_root() / "envs" / f"{extra}-{lock_hash()}"


def env_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def uv_executable() -> str:
    found = os.environ.get("UV") or shutil.which("uv")
    if not found:
        raise AseError(
            "uv_missing",
            "uv was not found, and it builds the calculator environments (xTB, MACE, UMA). "
            "Install uv (https://docs.astral.sh/uv/) or set UV to its path; EMT, NWChem and "
            "ORCA do not need it.",
        )
    return found


def status(extra: str) -> dict[str, Any]:
    """Whether the environment for ``extra`` is built for the current lock."""
    d = env_dir(extra)
    marker = d / "ready.json"
    if marker.is_file():
        try:
            info = json.loads(marker.read_text())
        except (OSError, ValueError):
            info = {}
        return {"prepared": True, "path": str(d / "venv"), **info}
    others = (
        sorted(p.name for p in (cache_root() / "envs").glob(f"{extra}-*") if p.is_dir() and p != d)
        if (cache_root() / "envs").is_dir()
        else []
    )
    return {
        "prepared": False,
        "path": str(d / "venv"),
        "download": APPROX_SIZE.get(extra),
        "stale_builds": others,
    }


def _dir_size(path: Path) -> int:
    total = 0
    for root, _, names in os.walk(path):
        for n in names:
            try:
                total += (Path(root) / n).lstat().st_size
            except OSError:
                pass
    return total


def ensure(extra: str) -> dict[str, Any]:
    """Build the environment for ``extra`` if needed; returns its status and build info."""
    from filelock import FileLock

    d = env_dir(extra)
    marker = d / "ready.json"
    if marker.is_file():
        return {**status(extra), "built_now": False}
    uv = uv_executable()
    d.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(d) + ".lock"):
        if marker.is_file():  # another process built it while we waited
            return {**status(extra), "built_now": False}
        project = d / "project"
        project.mkdir(parents=True, exist_ok=True)
        for name, content in project_files().items():
            (project / name).write_bytes(content)
        (project / "README.md").write_text("ase-mcp worker environment\n")
        venv = d / "venv"
        log = d / "build.log"
        cmd = [
            uv,
            "sync",
            "--frozen",
            "--no-dev",
            "--no-install-project",
            "--extra",
            extra,
            "--project",
            str(project),
            "--python",
            sys.executable,
        ]
        env = {**os.environ, "UV_PROJECT_ENVIRONMENT": str(venv)}
        env.pop("VIRTUAL_ENV", None)
        t0 = time.time()
        with log.open("w", encoding="utf-8") as fh:
            fh.write("$ " + " ".join(cmd) + "\n")
            fh.flush()
            proc = subprocess.run(
                cmd, env=env, stdout=fh, stderr=subprocess.STDOUT, cwd=str(project), check=False
            )
        if proc.returncode != 0:
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-15:]
            raise AseError(
                "env_build_failed",
                f"Building the {extra} environment failed (uv exit {proc.returncode}). "
                f"Last lines of {log}:\n" + "\n".join(tail),
                log=str(log),
            )
        info = {
            "extra": extra,
            "lock_hash": lock_hash(),
            "python": sys.version.split()[0],
            "built_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "build_seconds": round(time.time() - t0, 1),
            "size_bytes": _dir_size(venv),
        }
        marker.write_text(json.dumps(info, indent=2))
        return {**status(extra), "built_now": True}


def code_dir() -> Path:
    """A copy of this package's source for PYTHONPATH in extra environments."""
    h = hashlib.sha256()
    sources = sorted(PACKAGE_DIR.glob("*.py"))
    for p in sources:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    target = cache_root() / "code" / h.hexdigest()[:12]
    pkg = target / "ase_mcp"
    if not (pkg / "worker.py").is_file():
        tmp = target.with_name(target.name + f".tmp{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        (tmp / "ase_mcp").mkdir(parents=True)
        for p in sources:
            shutil.copy2(p, tmp / "ase_mcp" / p.name)
        try:
            tmp.rename(target)
        except OSError:  # another process won the race; its copy is identical
            shutil.rmtree(tmp, ignore_errors=True)
    return target


def worker_command(extra: str | None, job_file: Path) -> tuple[list[str], dict[str, str]]:
    """The command and environment that run one job."""
    env = dict(os.environ)
    if extra is None:
        python = sys.executable
        pythonpath = str(PACKAGE_DIR.parent)
    else:
        python = str(env_python(env_dir(extra) / "venv"))
        pythonpath = str(code_dir())
        env.pop("VIRTUAL_ENV", None)
    env["PYTHONPATH"] = pythonpath
    env.setdefault("PYTHONUNBUFFERED", "1")
    return [python, "-m", "ase_mcp.worker", str(job_file)], env
