"""Serialize non-Python builds and reuse complete immutable runtime projects."""

from __future__ import annotations

import json
import platform
import shutil
import time
from collections.abc import Callable
from pathlib import Path

from clio_kit.environment_locks import _FileLock
from clio_kit.runtimes import UnsupportedRuntime, go_binary, runtime_executable


def cached_build(runtime: str, project: Path, build: Callable[[], bool]) -> bool:
    """Avoid destructive reinstalls on ordinary warm Node/Go launches.

    The project name already identifies source and lock bytes. Markers and
    locks are siblings of that project so they cannot change its source hash.
    A failed/interrupted build never receives a completion marker.
    """
    if runtime == "python":
        return build()  # uv owns Python environment synchronization.
    try:
        executable = Path(runtime_executable(runtime)).resolve()
    except UnsupportedRuntime:
        return build()  # preserve the launcher's actionable toolchain diagnostic.
    stamp = executable.stat()
    identity = {
        "runtime": runtime,
        "platform": platform.system(),
        "machine": platform.machine(),
        "toolchain": str(executable),
        "toolchain_mtime": stamp.st_mtime_ns,
        "toolchain_size": stamp.st_size,
    }
    if runtime == "node":
        node = shutil.which("node")
        if node:
            node_path = Path(node).resolve()
            identity["node"] = str(node_path)
            identity["node_mtime"] = node_path.stat().st_mtime_ns
    marker = project.parent / f".{project.name}.built.json"
    lock = _FileLock(project.parent / f".{project.name}.build.lock")
    deadline = time.monotonic() + 300
    while not lock.try_acquire():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Another process is still building {project}")
        time.sleep(0.1)
    try:
        artifact = project / "node_modules" if runtime == "node" else go_binary(project)
        try:
            if artifact.exists() and json.loads(marker.read_text()) == identity:
                return True
        except (OSError, ValueError):
            pass
        marker.unlink(missing_ok=True)
        if not build():
            return False
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps(identity))
        temporary.replace(marker)
        return True
    finally:
        lock.release()
