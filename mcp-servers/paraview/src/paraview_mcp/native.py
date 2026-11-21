"""The node half on this Windows machine, run without a POSIX shell.

Elsewhere the front prepares and starts the node half with short shell
scripts, over SSH or on this machine. Windows has no such shell to count on, so
when ParaView runs here each step is done in Python instead. Each one reports
what its shell script would, so the attach logic is the same either way. A
connector prepare line is a shell line and is not run here.
"""

from __future__ import annotations

import io
import json
import os
import shutil
import signal
import subprocess
import sys
import tarfile
from pathlib import Path
from typing import Any

from . import config
from .transport import Completed

CREATE_NEW_PROCESS_GROUP = 0x00000200
CREATE_NO_WINDOW = 0x08000000


def wanted() -> bool:
    return os.name == "nt"


def _read(path: Path) -> str | None:
    try:
        return path.read_text("utf-8").strip()
    except OSError:
        return None


def probe(folder: str) -> dict[str, str]:
    """The probe script's markers: the folder, pvpython on PATH and the records."""
    root = Path(folder)
    root.mkdir(parents=True, exist_ok=True)
    scenes = root / "scenes"
    if not scenes.is_dir():
        older = sorted(
            (p for p in root.parent.glob("*/scenes") if p.parent != root),
            key=lambda p: [
                int(x) if x.isdigit() else 0 for x in p.parent.name.split(".")
            ],
            reverse=True,
        )
        if older:
            shutil.copytree(older[0], scenes)
    marks = {"DIR": str(root), "PVPYTHON": shutil.which("pvpython") or "", "NS": ""}
    for key, name in (
        ("ENV", "env.json"),
        ("CODE", "code/HASH"),
        ("NODE", "node.json"),
    ):
        text = _read(root / name)
        if text:
            marks[key] = text
    return marks


def _run(command: list[str], timeout: float, cwd: str | None = None) -> Completed:
    try:
        done = subprocess.run(
            command,
            capture_output=True,
            timeout=timeout,
            check=False,
            cwd=cwd,
            stdin=subprocess.DEVNULL,
            creationflags=CREATE_NO_WINDOW if os.name == "nt" else 0,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return Completed(1, "", f"{type(exc).__name__}: {exc}")
    return Completed(
        done.returncode,
        done.stdout.decode("utf-8", "replace"),
        done.stderr.decode("utf-8", "replace"),
    )


def interrogate(pvpython: str, pyinfo: str) -> Completed:
    """The version and Python facts, as the shell interrogation prints them."""
    version = _run([pvpython, "--version"], 180)
    info = _run([pvpython, "--dr", "-c", pyinfo], 180)
    first = (version.stdout.strip().splitlines() or [""])[0]
    lines = [f"CLIO_VERSION={first}"]
    lines += [line for line in info.stdout.splitlines() if line.startswith("CLIO_")]
    return Completed(info.code, "\n".join(lines), version.stderr + info.stderr)


def version_of(pvpython: str) -> str:
    """The ParaView version a pvpython reports, such as ``5.13.1``."""
    done = _run([pvpython, "--version"], 180)
    first = (done.stdout.strip().splitlines() or [""])[0]
    return first.rsplit(" ", 1)[-1]


def build(folder: str, pyinfo: dict[str, Any], pvpython: str | None) -> Completed:
    """Install the trame packages into ``folder/site`` for pvpython's Python."""
    from .attach import platform_tags

    pip_plats, uv_plat = platform_tags(pyinfo)
    pyver = str(pyinfo["python"])
    target = Path(folder) / f"site.building.{os.getpid()}"
    shutil.rmtree(target, ignore_errors=True)
    target.mkdir(parents=True)
    packages = list(config.NODE_PACKAGES)
    pip = [
        "install", "--quiet", "--disable-pip-version-check", "--no-input",
        "--timeout", "20", "--retries", "1", "--target", str(target),
        "--python-version", pyver,
        *(arg for p in pip_plats for arg in ("--platform", p)),
        "--only-binary=:all:", *packages,
    ]  # fmt: skip
    attempts: list[list[str]] = []
    uv = shutil.which("uv")
    cache = Path(folder) / "uv-cache"
    if uv:
        # A confined host may not grant uv's own profile cache; a cache in this
        # folder keeps every staged file in territory this server may write.
        attempts.append(
            [uv, "pip", "install", "--quiet", "--target", str(target),
             "--cache-dir", str(cache), "--link-mode", "copy",
             "--python-version", pyver, "--python-platform", uv_plat, *packages]
        )  # fmt: skip
    attempts.append([sys.executable, "-m", "pip", *pip])
    if pvpython:
        attempts.append([pvpython, "-m", "pip", *pip])
    errors = []
    for command in attempts:
        done = _run(command, 900)
        if done.code == 0:
            site = Path(folder) / "site"
            shutil.rmtree(site, ignore_errors=True)
            target.rename(site)
            shutil.rmtree(cache, ignore_errors=True)
            return Completed(0, "CLIO_BUILD=ok", "")
        errors.append(f"{Path(command[0]).name}: {done.stderr.strip()[-300:]}")
    shutil.rmtree(target, ignore_errors=True)
    shutil.rmtree(cache, ignore_errors=True)
    return Completed(3, "CLIO_BUILD=failed", "; ".join(errors))


def write_env(folder: str, env: dict[str, Any]) -> None:
    (Path(folder) / "env.json").write_text(json.dumps(env), "utf-8")


def upload(folder: str, bundle: bytes) -> Completed:
    root = Path(folder)
    new, code = root / "code.new", root / "code"
    try:
        shutil.rmtree(new, ignore_errors=True)
        new.mkdir(parents=True)
        with tarfile.open(fileobj=io.BytesIO(bundle), mode="r:gz") as archive:
            try:
                archive.extractall(new, filter="data")
            except TypeError:  # a Python without extraction filters
                archive.extractall(new)
        shutil.rmtree(code, ignore_errors=True)
        new.rename(code)
    except OSError as exc:
        return Completed(1, "", f"{type(exc).__name__}: {exc}")
    return Completed(0, "", "")


def stop(folder: str, pid: int, slot: str) -> None:
    (Path(folder) / f"{slot}.json").unlink(missing_ok=True)
    if pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass


def start(folder: str, command: list[str], slot: str) -> None:
    """Start the node half on its own, writing its output to ``<slot>.log``."""
    root = Path(folder)
    (root / f"{slot}.json").unlink(missing_ok=True)
    with open(root / f"{slot}.log", "wb") as log:
        subprocess.Popen(
            command,
            cwd=folder,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            creationflags=CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW,
            close_fds=True,
        )


def started(folder: str, slot: str) -> tuple[dict[str, Any] | None, str | None]:
    """The node record once written, and the log's tail if the start failed."""
    root = Path(folder)
    text = _read(root / f"{slot}.json")
    record = json.loads(text) if text and text.startswith("{") else None
    log = _read(root / f"{slot}.log") or ""
    failed = any(
        mark in log
        for mark in ("Traceback", "another node half", "pvserver join failed")
    )
    return record, ("\n".join(log.splitlines()[-25:]) if failed else None)
