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
import time
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


def own_slot(base: str = "node") -> str:
    """This front's own record name, used when another holds the folder's node."""
    return f"{base}-w{os.getpid()}"


def lock_free(folder: str, slot: str) -> bool:
    """Whether no node half holds ``<slot>.lock``."""
    from .node.main import _try_lock

    path = Path(folder) / f"{slot}.lock"
    if not path.exists():
        return True
    try:
        handle = open(path, "a")  # noqa: SIM115 - closed below
    except OSError:
        return False
    try:
        if not _try_lock(handle):
            return False
        if os.name == "nt":
            import msvcrt

            # Let go at once, so the node half about to start can take it.
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
        return True
    finally:
        handle.close()


def probe(folder: str, base: str = "node") -> dict[str, str]:
    """The probe script's markers: the folder, pvpython on PATH and the records.

    Windows has no network namespaces, but a host may confine each server it
    starts in its own sandbox, from which another sandbox's node half cannot be
    signalled. Each front's ``NS`` is therefore its own, so ``node_slot`` runs
    it beside a node half whose record has gone while its lock is still held.
    """
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
    own = own_slot(base)
    marks = {
        "DIR": str(root),
        "PVPYTHON": shutil.which("pvpython") or "",
        "NS": own.removeprefix(f"{base}-"),
        "LOCKED": "0" if lock_free(folder, base) else "1",
    }
    for key, name in (
        ("ENV", "env.json"),
        ("CODE", "code/HASH"),
        ("NODE", f"{base}.json"),
        ("OWNNODE", f"{own}.json"),
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


def stop(folder: str, pid: int, slot: str, wait: float = 15.0) -> bool:
    """Stop the node half in ``slot``; whether its lock was let go within ``wait``.

    A signal does not cross into another sandbox, so the node half is also asked
    through ``<slot>.stop``, which it watches.
    """
    root = Path(folder)
    (root / f"{slot}.json").unlink(missing_ok=True)
    if not lock_free(folder, slot):
        (root / f"{slot}.stop").write_text("stop", "utf-8")
    if pid > 0:
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError:
            pass
    deadline = time.monotonic() + wait
    while not lock_free(folder, slot):
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.5)
    (root / f"{slot}.stop").unlink(missing_ok=True)
    return True


def start(folder: str, command: list[str], slot: str) -> subprocess.Popen[bytes]:
    """Start the node half on its own, writing its output to ``<slot>.log``.

    The last node half's log is kept as ``<slot>.prev.log``, so the reason it
    stopped is still there to read after this one starts.
    """
    root = Path(folder)
    (root / f"{slot}.json").unlink(missing_ok=True)
    (root / f"{slot}.lost.json").unlink(missing_ok=True)
    log_path = root / f"{slot}.log"
    if log_path.exists():
        try:
            os.replace(log_path, root / f"{slot}.prev.log")
        except OSError:
            pass
    with open(log_path, "wb") as log:
        return subprocess.Popen(
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


def log_tail(folder: str, slot: str, lines: int = 25) -> str:
    """The end of ``<slot>.log``, or empty when there is none."""
    log = _read(Path(folder) / f"{slot}.log") or ""
    return "\n".join(log.splitlines()[-lines:])


def lost_note(folder: str, slot: str) -> str:
    """The note a node half left when its pvserver went, or empty."""
    return _read(Path(folder) / f"{slot}.lost.json") or ""
