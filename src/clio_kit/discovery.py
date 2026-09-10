"""Find the MCP servers a checkout or wheel ships, and how to start each one.

Discovery used to work by string-matching ``-mcp =`` inside each server's
``pyproject.toml`` and stripping that suffix to get a name. That is fragile in
two directions: any unrelated line containing ``-mcp =`` wins, because the
first match is taken, and it can only ever find Python projects, since a Go or
TypeScript server has no ``pyproject.toml`` to match against.

A generated ``clio-server.toml`` in each server directory states the same facts
outright -- what the server is called, what runtime starts it, and which lock
file pins it. The Python fallback below stays for a tree generated before the
descriptors existed, and for the source checkouts of forks that have not
regenerated.

Which lock pins which runtime is not restated here: it is a property of the
runtime, held once in :mod:`clio_kit.runtimes` beside the commands that build
and start it. A second copy would be a second thing to keep true.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from clio_kit.runtimes import lock_file_name, supported_runtimes

try:
    import tomllib
except ImportError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib  # type: ignore[import-not-found,no-redef]

DESCRIPTOR_NAME = "clio-server.toml"

# The runtimes a descriptor may claim. Every one of them is startable: the
# launcher builds and starts each from its own lock. What this repository
# currently *ships* is a separate question, and a narrower one -- see
# CONTRIBUTING.md on contributing a server in another language.
RUNTIME_LOCKS: dict[str, str] = {
    runtime: lock_file_name(runtime) for runtime in supported_runtimes()
}


def read_server_descriptor(server_dir: Path) -> dict[str, Any] | None:
    """Read one server's descriptor, or None when it has none.

    ``lock`` is optional and never selects anything: which file pins a runtime
    is a property of the runtime. It is checked rather than ignored because
    CONTRIBUTING.md shows it in the descriptor a contributor copies, so one
    naming a lock its runtime does not use is a belief worth contradicting
    now rather than at launch. ``version`` is free-form provenance.
    """
    descriptor_path = server_dir / DESCRIPTOR_NAME
    if not descriptor_path.is_file():
        return None
    with open(descriptor_path, "rb") as handle:
        data = tomllib.load(handle)

    name = data.get("name")
    runtime = data.get("runtime")
    entry = data.get("entry")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{descriptor_path} needs a name")
    if runtime not in RUNTIME_LOCKS:
        raise ValueError(
            f"{descriptor_path} declares runtime {runtime!r}; "
            f"expected one of {sorted(RUNTIME_LOCKS)}"
        )
    if not isinstance(entry, str) or not entry:
        raise ValueError(f"{descriptor_path} needs an entry")
    declared_lock = data.get("lock")
    expected_lock = RUNTIME_LOCKS[runtime]
    if declared_lock is not None and declared_lock != expected_lock:
        raise ValueError(
            f"{descriptor_path} declares lock {declared_lock!r}, but a "
            f"{runtime} server is pinned by {expected_lock!r}; the lock "
            "follows from the runtime and cannot be chosen per server"
        )
    return {
        "name": name,
        "runtime": runtime,
        "entry": entry,
        # Carried for the manifest generator rather than the launcher: a
        # non-Python server has no pyproject.toml to read these from, and the
        # descriptor is the only file this repository and a go module agree on.
        "description": data.get("description", ""),
        "version": data.get("version", ""),
    }


def _entry_point_from_pyproject(server_dir: Path) -> str | None:
    """Recover a Python server's console script from its project metadata.

    Kept for trees generated before descriptors existed. Unlike the string
    match it replaces, this parses the file and reads ``[project.scripts]``, so
    a stray ``-mcp =`` elsewhere in the document cannot win.
    """
    pyproject = server_dir / "pyproject.toml"
    if not pyproject.is_file():
        return None
    try:
        with open(pyproject, "rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError):
        return None
    scripts = data.get("project", {}).get("scripts", {})
    if not isinstance(scripts, dict):
        return None
    for script_name in scripts:
        if isinstance(script_name, str) and script_name.endswith("-mcp"):
            return script_name
    return None


def is_servers_root(path: Path) -> bool:
    """Return whether a directory holds at least one embedded server."""
    if not path.is_dir():
        return False
    return any(path.glob(f"*/{DESCRIPTOR_NAME}")) or any(path.glob("*/pyproject.toml"))


def discover_servers_in(servers_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """Map each server name to its entry command and to its directory name.

    Two maps rather than one because a server's name and its directory are
    allowed to differ, and callers need both: the name is what a user types,
    the directory is where the locked project lives.
    """
    entry_commands: dict[str, str] = {}
    directories: dict[str, str] = {}
    if not servers_path.exists():
        return entry_commands, directories

    for item in sorted(servers_path.iterdir()):
        if not item.is_dir() or item.name.startswith("."):
            continue
        try:
            descriptor = read_server_descriptor(item)
        except (ValueError, OSError):
            # A malformed descriptor must not take the whole catalogue down
            # with it; the rest of the servers still start.
            continue
        if descriptor is not None:
            entry_commands[descriptor["name"]] = descriptor["entry"]
            directories[descriptor["name"]] = item.name
            continue

        entry_point = _entry_point_from_pyproject(item)
        if entry_point:
            entry_commands[entry_point.removesuffix("-mcp").lower()] = entry_point
            directories[entry_point.removesuffix("-mcp").lower()] = item.name
    return entry_commands, directories
