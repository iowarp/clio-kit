"""Run ltspice-mcp 0.8.0 exactly as published (GPL-3.0-or-later).

Nothing of ltspice-mcp is copied here: ``pyproject.toml`` pins the release and
``uv.lock`` locks its tree, which the launcher installs into this server's own
environment. This module only starts that package's own command, ``ltspice-mcp``,
from the same environment with its stdio transport, so its tools, resources and
everything they return reach the client unchanged, and the server reports
ltspice-mcp's own name and version. Arguments after the server name pass through.

The upstream command runs as its own program (``exec`` on POSIX, a child process
on Windows, where ``exec`` cannot keep the client's stdio), so this wrapper never
imports it. The repository's metadata checks describe it by asking the
running upstream command what it advertises (``command`` and ``metadata_env``;
see ``scripts/upstream_metadata.py``).
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

#: The upstream distribution this server runs, as pinned in pyproject.toml.
UPSTREAM = "ltspice-mcp"
#: Its console script, installed beside this one.
SCRIPT = "ltspice-mcp"
#: What the upstream command is always given first.
ARGUMENTS: tuple[str, ...] = ()


def upstream_script() -> Path:
    """The upstream console script of this environment."""
    suffix = ".exe" if os.name == "nt" else ""
    return Path(sysconfig.get_path("scripts")) / f"{SCRIPT}{suffix}"


def command(argv: Sequence[str]) -> list[str]:
    """The command that serves ltspice-mcp over stdio with ``argv`` appended."""
    return [str(upstream_script()), *ARGUMENTS, *argv]


def main() -> None:
    """Replace this process with the upstream server (POSIX) or run it (Windows)."""
    argv = command(sys.argv[1:])
    if not Path(argv[0]).is_file():
        raise SystemExit(f"{SCRIPT} is not installed in {sys.prefix}; rebuild this server")
    if os.name == "nt":
        raise SystemExit(subprocess.call(argv))
    os.execv(argv[0], argv)


def metadata_env() -> dict[str, str]:
    """What the upstream command needs added to its environment to list everything."""
    return {}


if __name__ == "__main__":
    main()
