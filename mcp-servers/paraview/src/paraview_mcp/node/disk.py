"""What is on disk where the data is, read with the standard library alone.

The node half reads a dataset's files to follow a source that is still being
written: what a path or pattern matches, and each file's size and time. In a
built-in session the data is on the node half's machine and these run there. When the node half joins a
pvserver the data is on the server's machine, so the same functions run in the
server's Python (see ``datafs``). Nothing here imports ParaView or another node
module, so this file's source runs on its own.

Shared file systems: a machine that looked a path up before another machine
made it can be told it is not there for a minute or more (NFS caches the miss).
Listing the folder that holds it brings the path in at once. So every operation
run through ``dispatch`` first lists the folders above its path and the path's
own folder (``refresh``), and a pvserver's other ranks list them too, since each
rank's reader opens the files. A rerun that removes and remakes its output is
the usual case. Listing costs little on a local disk. A lookup that still finds
nothing (the path was made between the listing and the look) lists the folder
holding it once more and tries again (``_there``, ``_open``).

A file's size and time can be as old (``os.stat`` answers from the cache, which
some clusters keep for minutes), while opening a file checks it with the server.
So sizes and times are read through an open file (``_fresh``): otherwise a
growing source looks unchanged for minutes.
"""

from __future__ import annotations

import glob
import os
import time
from typing import Any

#: Seconds since its last change before a file a growing pattern matches is read.
#: A writer can take many seconds over one file (a NetCDF history gathered from
#: every rank, a large VTK piece), and a reader that opens it part way fails or
#: reads it short. ``PARAVIEW_FILE_SETTLE`` sets it; 0 reads every file at once.
FILE_SETTLE = float(os.environ.get("PARAVIEW_FILE_SETTLE") or 5.0)


def expand(path: str) -> str:
    """``path`` with ``~`` and variables expanded, made absolute, on this machine."""
    return os.path.abspath(os.path.expanduser(os.path.expandvars(path)))


def is_pattern(path: str) -> bool:
    return any(ch in path for ch in "*?[")


def refresh(path: str, depth: int = 2) -> list[str]:
    """List the ``depth`` folders above ``path``, top down, then ``path`` itself when
    it is a folder, so this machine sees what another machine has made there since
    it last looked. For a pattern, ``path`` is the folder before its first pattern
    part. Returns the folders listed; one that cannot be listed is passed over."""
    if is_pattern(path):
        while is_pattern(path) and os.path.dirname(path) != path:
            path = os.path.dirname(path)
        path = path or os.curdir
    path = os.path.abspath(path)
    folders = []
    above = os.path.dirname(path)
    for _ in range(depth):
        folders.append(above)
        if os.path.dirname(above) == above:
            break
        above = os.path.dirname(above)
    folders.reverse()
    listed = []
    for folder in [*folders, path]:
        if folder == path and not os.path.isdir(path):
            continue
        try:
            os.listdir(folder)
        except OSError:
            continue
        listed.append(folder)
    return listed


def refresh_for(op: str, args: dict[str, Any]) -> list[str]:
    """``refresh`` the path an operation is about to look at, if it has one."""
    path = args.get("path")
    if op == "expand" or not isinstance(path, str) or not path:
        return []
    return refresh(path)


def _relist(path: str) -> None:
    """List the folder holding ``path``, so a machine that cached ``path`` as missing
    looks it up again."""
    try:
        os.listdir(os.path.dirname(os.path.abspath(path)))
    except OSError:
        pass


def _there(path: str, test: Any = None) -> bool:
    """``test(path)`` (``os.path.exists`` by default); when it says no, asked once
    more after listing the folder that holds ``path``."""
    test = test or os.path.exists
    if test(path):
        return True
    _relist(path)
    return bool(test(path))


def _open(path: str, flags: int = os.O_RDONLY) -> int:
    """``os.open``; on "no such file", once more after listing the folder that holds
    ``path``."""
    try:
        return os.open(path, flags)
    except FileNotFoundError:
        _relist(path)
        return os.open(path, flags)


def _fresh(path: str) -> os.stat_result:
    """``path``'s size and times as they are now: read through an open handle, which
    a shared file system checks with the server, where ``os.stat`` may answer from
    what this machine saw a while ago. A path that cannot be opened (a folder on
    some systems, no read permission) falls back to ``os.stat``."""
    try:
        handle = _open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
    except FileNotFoundError:
        raise
    except OSError:
        return os.stat(path)
    try:
        return os.fstat(handle)
    finally:
        os.close(handle)


def _stat(path: str) -> tuple[int, int] | None:
    try:
        stat = _fresh(path)
    except OSError:
        return None
    return (stat.st_size, stat.st_mtime_ns)


def disk_key(path: str) -> Any:
    """What is on disk for ``path``, to tell when a source has grown: each matched
    file (or each entry of a folder) with its size and modification time."""
    if is_pattern(path):
        return tuple((name, _stat(name)) for name in sorted(glob.glob(path)))
    if os.path.isdir(path):
        try:
            names = sorted(os.listdir(path))
        except OSError:
            return None
        return tuple((name, _stat(os.path.join(path, name))) for name in names)
    return _stat(path)


def series(path: str) -> dict[str, Any]:
    """A pattern's ``disk_key`` and this machine's clock (nanoseconds), so the
    caller can tell which files changed in the last moments (``fresh_files``)."""
    return {"key": disk_key(path), "now": time.time_ns()}


def fresh_files(key: Any, now: int, settle: float = FILE_SETTLE) -> set[str]:
    """The files of a pattern's ``disk_key`` that changed less than ``settle``
    seconds before ``now``, by the clock of the machine that looked: a writer may
    be part way through them. A time further ahead of that clock than ``settle``
    says nothing (the clocks disagree) and holds no file."""
    fresh = set()
    for name, stat in key or ():
        if stat is not None and -settle < (now - stat[1]) / 1e9 < settle:
            fresh.add(name)
    return fresh


def files(path: str) -> list[str] | str | None:
    """The file (or folder) ``path`` names, or the sorted files its pattern matches;
    None when there is none."""
    if not is_pattern(path):
        return path if _there(path) else None
    return sorted(glob.glob(path)) or None


def source_readiness(
    path: str, pattern: bool, settle: float = FILE_SETTLE
) -> tuple[str | None, Any]:
    """For a growing source about to be opened: why it cannot be read yet (or None),
    and what is on disk. A pattern can be read once one file it matches has stopped
    changing (``FILE_SETTLE``)."""
    key = disk_key(path)
    if pattern:
        if not key:
            # Nothing matches yet: what the run writes beside it says it goes on.
            return "no file matches it yet", ("absent", _beside(path))
        if len(fresh_files(key, time.time_ns(), settle)) == len(key):
            return "its first file is still being written", key
        return None, key
    if not _there(path):
        return "it is not there yet", ("absent", _beside(path))
    return None, key


def _beside(path: str) -> Any:
    """What is in the folder a source that is not there yet will be in."""
    parent = os.path.dirname(path.rstrip("/"))
    while is_pattern(parent):
        parent = os.path.dirname(parent)
    return disk_key(parent) if parent and os.path.isdir(parent) else None


def dispatch(op: str, args: dict[str, Any]) -> Any:
    """Run one of these functions by name, with a JSON-friendly result, after
    listing the folders on the way to its path (see ``refresh``)."""
    if op == "refresh":
        return refresh(args["path"])
    refresh_for(op, args)
    if op == "expand":
        return expand(args["path"])
    if op == "disk_key":
        return disk_key(args["path"])
    if op == "files":
        return files(args["path"])
    if op == "series":
        return series(args["path"])
    if op == "source_readiness":
        return list(
            source_readiness(
                args["path"],
                args["pattern"],
                float(args.get("settle", FILE_SETTLE)),
            )
        )
    raise ValueError(f"no disk operation {op!r}")
