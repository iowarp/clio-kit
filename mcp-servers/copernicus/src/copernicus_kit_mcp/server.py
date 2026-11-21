"""Run copernicus-mcp 0.7.2 exactly as published (BSD-3-Clause).

Nothing of copernicus-mcp is copied here: ``pyproject.toml`` pins the release and
``uv.lock`` locks its tree, which the launcher installs into this server's own
environment. This module only starts that package's own command, ``copernicus-mcp``,
from the same environment with the ``serve`` command (stdio) it is served with, so its tools, resources and
everything they return reach the client unchanged, and the server reports
copernicus-mcp's own name and version. Arguments after the server name pass through.

The upstream command runs as its own program (``exec`` on POSIX, a child process
on Windows, where ``exec`` cannot keep the client's stdio), so this wrapper never
imports it. ``server.mcp`` describes it for the repository's metadata checks
by asking the running upstream command (see :mod:`copernicus_kit_mcp.metadata`).
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .metadata import UpstreamServer

#: The upstream distribution this server runs, as pinned in pyproject.toml.
UPSTREAM = "copernicus-mcp"
#: Its console script, installed beside this one.
SCRIPT = "copernicus-mcp"
#: What the upstream command is always given first.
ARGUMENTS: tuple[str, ...] = ("serve",)


def upstream_script() -> Path:
    """The upstream console script of this environment."""
    suffix = ".exe" if os.name == "nt" else ""
    return Path(sysconfig.get_path("scripts")) / f"{SCRIPT}{suffix}"


def command(argv: Sequence[str]) -> list[str]:
    """The command that serves copernicus-mcp over stdio with ``argv`` appended."""
    return [str(upstream_script()), *ARGUMENTS, *argv]


def main() -> None:
    """Replace this process with the upstream server (POSIX) or run it (Windows)."""
    argv = command(sys.argv[1:])
    if not Path(argv[0]).is_file():
        raise SystemExit(
            f"{SCRIPT} is not installed in {sys.prefix}; rebuild this server"
        )
    if os.name == "nt":
        raise SystemExit(subprocess.call(argv))
    os.execv(argv[0], argv)


def metadata_env() -> dict[str, str]:
    """What the upstream command needs added to its environment to list everything.

    Upstream offers its CDS tools only once a CDS key resolves; listing tools sends
    nothing to the CDS, so a placeholder key shows the full surface. Its job database
    and cache go to a fresh temporary directory rather than the home directory.
    """
    state = Path(tempfile.mkdtemp(prefix="copernicus-mcp-metadata-"))
    return {
        "CDSAPI_KEY": "00000000-0000-0000-0000-000000000000",
        "COPERNICUS_MCP_CACHE_DIR": str(state / "cache"),
        "COPERNICUS_MCP_STATE_DB": str(state / "state.sqlite"),
    }


def __getattr__(name: str) -> UpstreamServer:
    """``mcp``: the upstream server's advertised metadata, read on first use."""
    if name != "mcp":
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from .metadata import UpstreamServer

    return UpstreamServer(command([]), metadata_env())


if __name__ == "__main__":
    main()
