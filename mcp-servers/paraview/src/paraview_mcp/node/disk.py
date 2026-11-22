"""What is on disk where the data is, read with the standard library alone.

The node half reads a dataset's files to follow a source that is still being
written: what a path or pattern matches, each file's size and time, and whether
an ADIOS2 BP5 folder holds whole steps yet. In a built-in session the data is on
the node half's machine and these run there. When the node half joins a
pvserver the data is on the server's machine, so the same functions run in the
server's Python (see ``datafs``). Nothing here imports ParaView or another node
module, so this file's source runs on its own.

ADIOS2 BP5: ParaView's ADIOS2 reader takes the process down when it opens or
re-reads a BP5 folder whose index names a step that the metadata does not hold
yet, and it waits forever for data that is not on disk. A writer adds a step's
data, then its metadata, then its index record, but a reader on another machine
of a shared file system can see them land in any order. So a growing BP5 folder
goes to the reader only when its files agree: at least one whole step in the
index (``md.idx``), the metadata (``md.0``) holding the last one, the
meta-metadata (``mmd.0``) written, and each data file (``data.N``) reaching into
the step's last block.

The index does not give the size of a step's last block, so a data file that
reaches into it may still be short. ``Readiness.clear`` says the data reaches a
whole block estimate past it. Otherwise the data holds every step but the last
(it reaches into the last, so past the end of the one before), and
``write_index`` gives a reader a copy of the folder whose index stops there; the
last step is read once the folder stops changing for a moment. A writer that adds
a step every second rarely stops, and its data ends where its last step ends.

Other formats (BP4, a single file) are not checked: ``check`` reports them ready.

Shared file systems: a machine that looked a path up before another machine
made it can be told it is not there for a minute or more (NFS caches the miss).
Listing the folder that holds it brings the path in at once. So every operation
run through ``dispatch`` first lists the folders above its path and the path's
own folder (``refresh``), and a pvserver's other ranks list them too, since each
rank's reader opens the files. A rerun that removes and remakes its output is
the usual case. Listing costs little on a local disk. A file's size and time can
be as old, too (``os.stat`` answers from the cache, which some clusters keep
for minutes), while opening a file checks it with the server.
So sizes and times are read through an open file (``_fresh``): otherwise a
growing source looks unchanged for minutes, and an index read whole runs ahead
of the metadata size beside it.
"""

from __future__ import annotations

import glob
import itertools
import os
import struct
from dataclasses import asdict, dataclass
from typing import Any

HEADER = 64
BP_VERSION_AT = 37
STEP = b"s"
WRITER_MAP = b"w"


@dataclass(frozen=True)
class Readiness:
    reason: str | None = None
    clear: bool = True
    steps: int = 0


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


def _fresh(path: str) -> os.stat_result:
    """``path``'s size and times as they are now: read through an open handle, which
    a shared file system checks with the server, where ``os.stat`` may answer from
    what this machine saw a while ago. A path that cannot be opened (a folder on
    some systems, no read permission) falls back to ``os.stat``."""
    try:
        handle = os.open(path, os.O_RDONLY | getattr(os, "O_NONBLOCK", 0))
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


def files(path: str) -> list[str] | str | None:
    """The file (or folder) ``path`` names, or the sorted files its pattern matches;
    None when there is none."""
    if not is_pattern(path):
        return path if os.path.exists(path) else None
    return sorted(glob.glob(path)) or None


def is_bp(path: str, reader: str | None = None) -> bool:
    """Whether ``path`` is (or will be) an ADIOS2 BP folder."""
    if reader and "ADIOS2" in reader:
        return True
    if os.path.isfile(os.path.join(path, "md.idx")):
        return True
    return path.rstrip("/").lower().endswith((".bp", ".bp4", ".bp5"))


def _size(path: str) -> int:
    try:
        return _fresh(path).st_size
    except OSError:
        return -1


def _records(index: bytes) -> list[tuple[bytes, tuple[int, ...]]]:
    """The whole records after the header; a record still being written is left out,
    as the reader leaves it out."""
    records = []
    pos = HEADER
    while pos + 9 <= len(index):
        kind = index[pos : pos + 1]
        (length,) = struct.unpack_from("<Q", index, pos + 1)
        end = pos + 9 + length
        if end > len(index) or length % 8:
            break
        records.append((kind, struct.unpack_from(f"<{length // 8}Q", index, pos + 9)))
        pos = end
    return records


def _starts(record: tuple[int, ...], writers: int) -> list[int] | None:
    """Where each writer's last block of a step starts in its data file. After the
    metadata position, size and flush count, each writer has a position and size
    per flush and then the start of its last block."""
    per = 2 * record[2] + 1
    if len(record) < 3 + per * writers:
        return None
    return [record[3 + w * per + per - 1] for w in range(writers)]


def check(path: str) -> Readiness:
    """Whether the BP5 folder at ``path`` can be read now, and if not, why not."""
    if not os.path.isdir(path):
        return Readiness("the folder is not there yet", False)
    try:
        with open(os.path.join(path, "md.idx"), "rb") as handle:
            index = handle.read()
    except FileNotFoundError:
        return Readiness("no step is written yet (there is no md.idx)", False)
    except OSError as exc:
        return Readiness(f"its index cannot be read ({exc})", False)
    if len(index) < HEADER:
        return Readiness("its index header is not written whole yet", False)
    if index[BP_VERSION_AT] != 5:
        return Readiness()
    writers = 0
    subfile_of: list[int] = []
    steps: list[tuple[int, ...]] = []
    for kind, values in _records(index):
        if kind == WRITER_MAP and len(values) >= 3:
            writers = values[0]
            subfile_of = list(values[3 : 3 + writers])
        elif kind == STEP and len(values) >= 3:
            steps.append(values)
    if not writers:
        return Readiness("its index has no writer map yet", False)
    if not steps:
        return Readiness("no whole step is in its index yet", False)
    last, count = steps[-1], len(steps)
    if _size(os.path.join(path, "md.0")) < last[0] + last[1]:
        return Readiness(f"its metadata (md.0) does not hold step {count} yet", False)
    if _size(os.path.join(path, "mmd.0")) <= 0:
        return Readiness("its meta-metadata (mmd.0) is not written yet", False)
    starts = _starts(last, writers)
    if starts is None:
        return Readiness(steps=count)
    before = _starts(steps[-2], writers) if count > 1 else None
    clear = True
    for subfile in sorted(
        {subfile_of[w] if w < len(subfile_of) else 0 for w in range(writers)}
    ):
        mine = [
            w
            for w in range(writers)
            if (subfile_of[w] if w < len(subfile_of) else 0) == subfile
        ]
        positions = sorted(starts[w] for w in mine)
        gaps = [b - a for a, b in itertools.pairwise(positions) if b > a]
        if not gaps and before is not None:
            gaps = [starts[w] - before[w] for w in mine if starts[w] > before[w]]
        need = positions[-1] + (min(gaps) if gaps else 1)
        size = _size(os.path.join(path, f"data.{subfile}"))
        if size < need:
            return Readiness(
                f"its data (data.{subfile}) does not hold step {count} yet",
                False,
                count,
            )
        if not gaps or size < positions[-1] + 2 * max(gaps):
            clear = False
    return Readiness(None, clear, count)


def whole_steps(state: Readiness) -> int:
    """How many steps the data surely holds: all, when it clearly holds the last;
    otherwise all but the last."""
    return state.steps if state.clear else max(state.steps - 1, 0)


def _cut(index: bytes, steps: int) -> int:
    """Where the index ends after its first ``steps`` step records."""
    pos, seen = HEADER, 0
    while pos + 9 <= len(index):
        kind = index[pos : pos + 1]
        (length,) = struct.unpack_from("<Q", index, pos + 1)
        end = pos + 9 + length
        if end > len(index) or length % 8:
            break
        if kind == STEP:
            if seen == steps:
                break
            seen += 1
        pos = end
    return pos


def write_index(path: str, folder: str, steps: int) -> str:
    """Make ``folder`` show the BP5 output at ``path`` up to its first ``steps``
    steps: links to its files, and its index cut after that step. A reader given
    ``folder`` cannot reach a step whose data may not be on disk yet. Returns the
    folder, expanded on this machine."""
    folder = expand(folder)
    os.makedirs(folder, exist_ok=True)
    for name in os.listdir(path):
        link = os.path.join(folder, name)
        if name != "md.idx" and not os.path.lexists(link):
            os.symlink(os.path.join(path, name), link)
    with open(os.path.join(path, "md.idx"), "rb") as handle:
        index = handle.read()
    part = f"{folder.rstrip('/')}.idx"
    with open(part, "wb") as handle:
        handle.write(index[: _cut(index, steps)])
    os.replace(part, os.path.join(folder, "md.idx"))
    return folder


def source_readiness(
    path: str, reader: str | None, pattern: bool
) -> tuple[str | None, bool, Any]:
    """For a growing source about to be opened: why it cannot be read yet (or None),
    whether it can be read without waiting for it to settle, and what is on disk."""
    key = disk_key(path)
    if pattern:
        return (None if key else "no file matches it yet"), True, key
    if not os.path.exists(path):
        return "it is not there yet", False, key
    if not is_bp(path, reader):
        return None, True, key
    state = check(path)
    return state.reason, state.clear, key


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
    if op == "is_bp":
        return is_bp(args["path"], args.get("reader"))
    if op == "check":
        return asdict(check(args["path"]))
    if op == "write_index":
        return write_index(args["path"], args["folder"], int(args["steps"]))
    if op == "source_readiness":
        return list(source_readiness(args["path"], args.get("reader"), args["pattern"]))
    raise ValueError(f"no disk operation {op!r}")
