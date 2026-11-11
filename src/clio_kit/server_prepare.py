"""``clio-kit mcp-server --prepare``: build a server's environment without starting it.

CLIO builds each MCP server's environment before it starts the server confined,
because a confined server cannot always run its own first build. The build here
is the one a normal ``clio-kit mcp-server <server>`` start does; the launch is
skipped. stdout carries exactly one ``clio-kit.mcp-prepare.v1`` JSON line, and
build output and cache events go to stderr.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from typing import TYPE_CHECKING, NoReturn

import click

from clio_kit.env_cache import EnvironmentInUseMarker, default_event_emitter
from clio_kit.runtimes import UnsupportedRuntime

if TYPE_CHECKING:
    from pathlib import Path

    from clio_kit import _LockedServerRuntime

# Exit codes and ``error`` kinds are stable: CLIO maps them to its typed
# provisioning failures.
PREPARE_RESULT_SCHEMA = "clio-kit.mcp-prepare.v1"
PREPARE_EXIT_UNKNOWN_SERVER = 1
PREPARE_EXIT_BUILD_FAILED = 10
PREPARE_EXIT_ACCESS_DENIED = 11
PREPARE_EXIT_PATH_TOO_LONG = 12
PREPARE_EXIT_NETWORK = 13
PREPARE_EXIT_UV_MISSING = 14
_PREPARE_EXIT_CODES = {
    "unknown_server": PREPARE_EXIT_UNKNOWN_SERVER,
    "build_failed": PREPARE_EXIT_BUILD_FAILED,
    "access_denied": PREPARE_EXIT_ACCESS_DENIED,
    "path_too_long": PREPARE_EXIT_PATH_TOO_LONG,
    "network": PREPARE_EXIT_NETWORK,
    "uv_missing": PREPARE_EXIT_UV_MISSING,
}
_PATH_TOO_LONG_MARKERS = (
    "os error 206",
    "winerror 206",
    "filename or extension is too long",
    "file name too long",
    "os error 36",
)
_ACCESS_DENIED_MARKERS = (
    "access is denied",
    "os error 5)",
    "winerror 5]",
    "permission denied",
    "os error 13",
    "windows pe resources",
)
_NETWORK_MARKERS = (
    "failed to fetch",
    "failed to download",
    "error sending request",
    "dns error",
    "tcp connect error",
    "connection refused",
    "connection reset",
    "connection aborted",
    "operation timed out",
    "network is unreachable",
    "no such host is known",
    "could not connect",
    "os error 11001",
    "os error 10060",
    "os error 10061",
    "request failed after",
    # Component artifact downloads (urllib) report these.
    "urlopen error",
    "getaddrinfo failed",
    "name or service not known",
    "temporary failure in name resolution",
)
_WINDOWS_MAX_PATH = 260


class PrepareFailure(Exception):
    """A typed ``mcp-server --prepare`` failure; ``kind`` keys the exit code."""

    def __init__(self, kind: str, detail: str) -> None:
        super().__init__(detail)
        self.kind = kind
        self.detail = detail


def _classify_build_output(output: str) -> str:
    """Return the typed failure kind a failed build's output names."""
    lowered = output.lower()
    if any(marker in lowered for marker in _PATH_TOO_LONG_MARKERS):
        return "path_too_long"
    if any(marker in lowered for marker in _ACCESS_DENIED_MARKERS):
        return "access_denied"
    if any(marker in lowered for marker in _NETWORK_MARKERS):
        return "network"
    return "build_failed"


def _classify_os_error(exc: OSError) -> str:
    """Return the typed failure kind for an OS error raised while materializing."""
    winerror = getattr(exc, "winerror", None)
    names = [str(name) for name in (exc.filename, exc.filename2) if name]
    if winerror == 206 or exc.errno == 36:  # ERROR_FILENAME_EXCED_RANGE / ENAMETOOLONG
        return "path_too_long"
    if winerror == 3 and any(len(name) >= _WINDOWS_MAX_PATH for name in names):
        return "path_too_long"
    if isinstance(exc, PermissionError) or winerror == 5:
        return "access_denied"
    return _classify_build_output(str(exc))


def _environment_interpreter(environment: Path) -> Path:
    if os.name == "nt":
        return environment / "Scripts" / "python.exe"
    return environment / "bin" / "python"


def _discard_partial_environment(runtime: _LockedServerRuntime) -> None:
    """Remove a half-built environment uv would refuse ("not a valid Python environment").

    An environment whose build was interrupted (or denied mid-write) is left
    without a usable interpreter; every later ``uv sync`` then fails on it. It
    is only removed when no live launcher holds it.
    """
    from clio_kit.environment_locks import environment_in_use

    environment = runtime.environment
    try:
        if not environment.is_dir() or not any(environment.iterdir()):
            return
    except OSError:
        return
    if (environment / "pyvenv.cfg").is_file() and _environment_interpreter(
        environment
    ).is_file():
        return
    if environment_in_use(runtime.cache_root, environment.name):
        return
    default_event_emitter(
        {
            "event": "partial_environment_discarded",
            "server": runtime.server,
            "environment": str(environment),
        }
    )
    shutil.rmtree(environment)


def _emit_prepare_result(payload: dict[str, object]) -> None:
    click.echo(json.dumps({"schema": PREPARE_RESULT_SCHEMA, **payload}, sort_keys=True))


def _prepare_failed(server: str, kind: str, detail: str) -> NoReturn:
    detail = detail.strip()
    click.echo(
        f"clio-kit mcp-server --prepare {server}: error={kind}: {detail}", err=True
    )
    _emit_prepare_result(
        {"status": "error", "server": server, "error": kind, "detail": detail[-2000:]}
    )
    sys.exit(_PREPARE_EXIT_CODES[kind])


def prepare_server(server: str | None, branch: str | None) -> None:
    """Build or sync one server's environment without starting it, then exit 0."""
    import clio_kit as kit
    from clio_kit.component_store import server_project
    from clio_kit.runtime_build import cached_build

    if not server:
        raise click.UsageError("--prepare requires a server name.")
    if branch:
        raise click.UsageError(
            "--prepare cannot be combined with --branch (uvx owns that environment)."
        )
    server_command_map, dir_name_map = kit.auto_discover_mcps()
    server_lower = server.lower()
    if server_lower not in server_command_map:
        _prepare_failed(
            server,
            "unknown_server",
            f"available servers: {', '.join(sorted(server_command_map))}",
        )
    entry_command = server_command_map[server_lower]
    started = time.monotonic()
    child_environment = kit.subprocess_env_with_github_https_rewrite()
    try:
        server_path = server_project(dir_name_map[server_lower], kit.get_servers_path())
        if not server_path.exists():
            # An installed console script has no clio-kit-owned environment to build.
            _emit_prepare_result(
                {"status": "not_applicable", "server": server_lower, "seconds": 0.0}
            )
            return
        runtime = kit._locked_server_runtime(server_path, child_environment)
        if runtime.runtime == "python":
            _discard_partial_environment(runtime)
        runtime.environment.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _prepare_failed(
            server_lower, _classify_os_error(exc), f"{type(exc).__name__}: {exc}"
        )
    except click.ClickException as exc:
        _prepare_failed(server_lower, "build_failed", exc.format_message())
    except (ValueError, UnsupportedRuntime) as exc:
        _prepare_failed(server_lower, _classify_build_output(str(exc)), str(exc))

    builds: list[subprocess.CompletedProcess[bytes]] = []

    def build() -> bool:
        completed = kit._run_locked_build(
            runtime.runtime, runtime.project, entry_command, child_environment
        )
        builds.append(completed)
        return completed.returncode == 0

    toolchain_kind = "uv_missing" if runtime.runtime == "python" else "build_failed"
    uv = kit.uv_command()
    with EnvironmentInUseMarker(runtime.cache_root, runtime.environment.name):
        try:
            built = cached_build(runtime.runtime, runtime.project, build)
        except UnsupportedRuntime as exc:
            _prepare_failed(server_lower, toolchain_kind, str(exc))
        except FileNotFoundError as exc:
            hint = "; install uv: https://github.com/astral-sh/uv"
            _prepare_failed(
                server_lower,
                toolchain_kind,
                f"build tool not found ({exc})"
                + (hint if runtime.runtime == "python" else ""),
            )
        except OSError as exc:
            _prepare_failed(
                server_lower, _classify_os_error(exc), f"{type(exc).__name__}: {exc}"
            )
        if not built:
            if not builds:
                _prepare_failed(server_lower, "build_failed", "build did not run")
            completed = builds[-1]
            output = b"\n".join((completed.stdout or b"", completed.stderr or b""))
            text = output.decode("utf-8", errors="replace")
            _prepare_failed(
                server_lower,
                _classify_build_output(text),
                f"{runtime.runtime} build exited {completed.returncode}: {text[-1500:]}",
            )
        kit._maintain_cache_after_build(runtime, uv)
    _emit_prepare_result(
        {
            "status": "prepared",
            "server": server_lower,
            "environment": str(runtime.environment),
            "project": str(runtime.project),
            "seconds": round(time.monotonic() - started, 3),
        }
    )
