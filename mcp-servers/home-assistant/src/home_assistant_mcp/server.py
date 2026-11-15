"""Run ha-mcp 9.2.0 exactly as published (MIT).

Nothing of ha-mcp is copied here: ``pyproject.toml`` pins the release and
``uv.lock`` locks its tree, which the launcher installs into this server's own
environment. This module only starts that package's own command, ``ha-mcp``,
from the same environment with its default stdio transport, so its tools, resources and
everything they return reach the client unchanged, and the server reports
ha-mcp's own name and version. Arguments after the server name pass through.

The upstream command runs as its own program (``exec`` on POSIX, a child process
on Windows, where ``exec`` cannot keep the client's stdio), so this wrapper never
imports it.
"""

from __future__ import annotations

import os
import subprocess
import sys
import sysconfig
from collections.abc import Sequence
from pathlib import Path

#: The upstream distribution this server runs, as pinned in pyproject.toml.
UPSTREAM = "ha-mcp"
#: Its console script, installed beside this one.
SCRIPT = "ha-mcp"
#: What the upstream command is always given first.
ARGUMENTS: tuple[str, ...] = ()


def upstream_script() -> Path:
    """The upstream console script of this environment."""
    suffix = ".exe" if os.name == "nt" else ""
    return Path(sysconfig.get_path("scripts")) / f"{SCRIPT}{suffix}"


def command(argv: Sequence[str]) -> list[str]:
    """The command that serves ha-mcp over stdio with ``argv`` appended."""
    return [str(upstream_script()), *ARGUMENTS, *argv]


def main() -> None:
    """Replace this process with the upstream server (POSIX) or run it (Windows)."""
    argv = command(sys.argv[1:])
    if not Path(argv[0]).is_file():
        raise SystemExit(f"{SCRIPT} is not installed in {sys.prefix}; rebuild this server")
    if os.name == "nt":
        raise SystemExit(subprocess.call(argv))
    os.execv(argv[0], argv)


if __name__ == "__main__":
    main()
