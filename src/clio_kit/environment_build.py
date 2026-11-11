"""Self-healing, serialized builds for cached Python MCP server environments.

An interrupted first build can leave an environment directory without an
interpreter. uv then refuses that directory on every later launch ("not a
valid Python environment"), so the server could never start again. Builds of
one environment are therefore serialized by a build lock, and the exclusive
holder discards a directory that is not a complete venv. A missing environment
is created in a temporary sibling and renamed into place only once it has an
interpreter, so an interrupted creation cannot poison the real path. uv's own
frozen sync then repairs any partially installed packages in place.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from clio_kit._env_cache_config import EmitEvent
from clio_kit.environment_locks import EnvironmentBuildLock, environment_in_use


def environment_interpreter(environment: Path) -> Path:
    """Return the interpreter path a complete venv at ``environment`` contains."""
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def is_complete_environment(environment: Path) -> bool:
    """Whether ``environment`` is a venv uv can sync into rather than refuse."""
    return (environment / "pyvenv.cfg").is_file() and environment_interpreter(
        environment
    ).is_file()


def _staging_path(environment: Path) -> Path:
    # Dot-prefixed so cache maintenance never mistakes it for a server spec.
    return (
        environment.parent / f".{environment.name}.{os.getpid()}.{time.time_ns()}.tmp"
    )


def _remove_stale_staging(environment: Path) -> None:
    """Remove leftovers of builds killed mid-way; only call under the build lock."""
    for leftover in environment.parent.glob(f".{environment.name}.*.tmp"):
        shutil.rmtree(leftover, ignore_errors=True)


def discard_partial_environment(
    cache_root: Path, environment: Path, *, server: str, emit: EmitEvent
) -> bool:
    """Remove a non-empty environment directory that is not a complete venv.

    The directory is renamed aside first, so the real path is freed atomically
    even when some file inside cannot be deleted yet. It is left alone while
    any other live launcher holds it. Raises ``OSError`` if the rename fails.
    """
    try:
        if not environment.is_dir() or not any(environment.iterdir()):
            return False
    except OSError:
        return False
    if is_complete_environment(environment):
        return False
    if environment_in_use(cache_root, environment.name, ignore_current_process=True):
        emit(
            {
                "event": "partial_environment_kept",
                "server": server,
                "environment": str(environment),
                "reason": "in_use",
            }
        )
        return False
    discarded = _staging_path(environment)
    os.replace(environment, discarded)
    emit(
        {
            "event": "partial_environment_discarded",
            "server": server,
            "environment": str(environment),
        }
    )
    shutil.rmtree(discarded, ignore_errors=True)
    return True


def create_environment_atomically(
    environment: Path,
    *,
    uv: str,
    project: Path,
    child_environment: Mapping[str, str],
) -> bool:
    """Create a missing venv in a temporary sibling and rename it into place.

    Best-effort: on any failure the environment path is left absent or as it
    was, and the frozen sync that follows creates the venv itself.
    """
    try:
        if environment.is_dir():
            if any(environment.iterdir()):
                return False
            environment.rmdir()
        elif environment.exists():
            return False
        environment.parent.mkdir(parents=True, exist_ok=True)
    except OSError:
        return False
    staging = _staging_path(environment)
    try:
        completed = subprocess.run(
            [uv, "venv", "--quiet", "--project", str(project), str(staging)],
            env=dict(child_environment),
            stdin=subprocess.DEVNULL,
            capture_output=True,
        )
        for stream in (completed.stdout, completed.stderr):
            if stream:
                sys.stderr.write(stream.decode("utf-8", errors="replace"))
        sys.stderr.flush()
        if completed.returncode != 0 or not is_complete_environment(staging):
            return False
        os.replace(staging, environment)
        return True
    except OSError:
        return False
    finally:
        if staging.exists():
            shutil.rmtree(staging, ignore_errors=True)


@contextmanager
def locked_environment_build(
    cache_root: Path,
    environment: Path,
    *,
    server: str,
    uv: str,
    project: Path,
    child_environment: Mapping[str, str],
    emit: EmitEvent,
) -> Iterator[None]:
    """Hold one environment's build lock, healing it first when exclusive.

    The caller runs its frozen sync inside this context. Enter the in-use
    marker inside it too, so a launcher waiting for the lock never looks like a
    live holder of the environment being healed.
    """
    with EnvironmentBuildLock(cache_root, environment.name) as exclusive:
        if exclusive:
            _remove_stale_staging(environment)
            try:
                discard_partial_environment(
                    cache_root, environment, server=server, emit=emit
                )
            except OSError as exc:
                emit(
                    {
                        "event": "partial_environment_discard_failed",
                        "server": server,
                        "environment": str(environment),
                        "reason": repr(exc),
                    }
                )
            create_environment_atomically(
                environment,
                uv=uv,
                project=project,
                child_environment=child_environment,
            )
        yield
