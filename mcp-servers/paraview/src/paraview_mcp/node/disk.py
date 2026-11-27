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
meta-metadata (``mmd.0``) written, and each data file (``data.N``) holding the
step's last block.

The index gives where each writer's last block of a step starts, not its size, and
a writer starts each block on a stripe boundary, so the space between blocks says
little about them. The step's metadata in ``md.0`` gives each writer's data size
(``_block_ends``), so the last step is read once every data file reaches its end
(``Readiness.clear``). Until then the data holds every step but the last
(``Readiness.short``): ``write_index`` gives a reader a copy of the folder whose
index stops there, and the last step is read once its data is all there, however
long that takes. Metadata laid out in a way this does not know (an ADIOS2 that
writes it otherwise) gives no size: the last step is then read once its data
reaches into its last block and the folder stops changing for a moment.

Other formats (BP4, a single file) are not checked: ``check`` reports them ready.

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
growing source looks unchanged for minutes, and an index read whole runs ahead
of the metadata size beside it.
"""

from __future__ import annotations

import glob
import os
import shutil
import struct
import time
from dataclasses import asdict, dataclass
from typing import Any

HEADER = 64
BP_VERSION_AT = 37
#: Where a BP index header says whether its writer still has the output open, by BP
#: version, as ADIOS2's own ``adios2_deactivate_bp`` reads it. Closing clears it.
ACTIVE_AT = {4: 38, 5: 39}
#: The entry ``disk_key`` adds for a BP folder: whether its writer still writes.
WRITER_ACTIVE = "writer_active"
STEP = b"s"
WRITER_MAP = b"w"
#: Seconds since its last change before a file a growing pattern matches is read.
#: A writer can take many seconds over one file (a NetCDF history gathered from
#: every rank, a large VTK piece), and a reader that opens it part way fails or
#: reads it short. ``PARAVIEW_FILE_SETTLE`` sets it; 0 reads every file at once.
FILE_SETTLE = float(os.environ.get("PARAVIEW_FILE_SETTLE") or 5.0)


@dataclass(frozen=True)
class Readiness:
    reason: str | None = None
    clear: bool = True
    steps: int = 0
    #: The metadata says the last step's data is not all on disk yet: it is not
    #: read however long the folder stays still.
    short: bool = False


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


def _read(path: str) -> bytes:
    with os.fdopen(_open(path), "rb") as handle:
        return handle.read()


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


def writer_active(path: str) -> bool | None:
    """Whether the ADIOS2 writer of the BP4 or BP5 folder ``path`` still has it open:
    a run that ends closes its output, and the index header says so. None when
    ``path`` is no such folder or its header is not written whole yet."""
    try:
        handle = _open(os.path.join(path, "md.idx"))
    except OSError:
        return None
    try:
        header = os.read(handle, HEADER)
    except OSError:
        return None
    finally:
        os.close(handle)
    if len(header) < HEADER:
        return None
    at = ACTIVE_AT.get(header[BP_VERSION_AT])
    return None if at is None else bool(header[at])


def writing(key: Any) -> bool | None:
    """What a ``disk_key`` says of a BP folder's writer (``writer_active``)."""
    if isinstance(key, (list, tuple)) and key:
        last = key[-1]
        if (
            isinstance(last, (list, tuple))
            and len(last) == 2
            and last[0] == WRITER_ACTIVE
        ):
            return bool(last[1])
    return None


def disk_key(path: str) -> Any:
    """What is on disk for ``path``, to tell when a source has grown: each matched
    file (or each entry of a folder) with its size and modification time. A BP
    folder's key ends with whether its writer still has it open (``writing``)."""
    if is_pattern(path):
        return tuple((name, _stat(name)) for name in sorted(glob.glob(path)))
    if os.path.isdir(path):
        try:
            names = sorted(os.listdir(path))
        except OSError:
            return None
        key = tuple((name, _stat(os.path.join(path, name))) for name in names)
        active = writer_active(path) if "md.idx" in names else None
        return key if active is None else (*key, (WRITER_ACTIVE, active))
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
    if not _there(path, os.path.isdir):
        return Readiness("the folder is not there yet", False)
    try:
        index = _read(os.path.join(path, "md.idx"))
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
    ends = _block_ends(path, last, writers, starts)
    clear, short = ends is not None, False
    for subfile in sorted(
        {subfile_of[w] if w < len(subfile_of) else 0 for w in range(writers)}
    ):
        mine = [
            w
            for w in range(writers)
            if (subfile_of[w] if w < len(subfile_of) else 0) == subfile
        ]
        size = _size(os.path.join(path, f"data.{subfile}"))
        # Every earlier step ends before the first block of the last one.
        if size < min(starts[w] for w in mine) or (
            ends is None and size <= max(starts[w] for w in mine)
        ):
            return Readiness(
                f"its data (data.{subfile}) does not hold step {count} yet",
                False,
                count,
            )
        if ends is not None and size < max(ends[w] for w in mine):
            clear, short = False, True
    if short and count == 1:
        return Readiness(
            f"its data does not hold step {count} whole yet", False, count, True
        )
    return Readiness(None, clear, count, short)


def _block_ends(
    path: str, record: tuple[int, ...], writers: int, starts: list[int]
) -> list[int] | None:
    """Where each writer's data of the step in ``record`` ends in its data file, or
    None when the metadata cannot say.

    The index gives where a writer's last block starts, not its size, and a writer
    starts each block on a stripe boundary, so the space between blocks is no
    measure of one. Each writer's metadata block in ``md.0`` holds the size: it is
    an FFS record (the format ID that ``mmd.0`` lists, the record's 8-byte length,
    padding to 8) whose base holds ``BitFieldCount``, ``BitField`` and then
    ``DataBlockSize``, the step's data of that writer, earlier flushes included."""
    md_pos, md_size = record[0], record[1]
    try:
        ids = _format_ids(os.path.join(path, "mmd.0"))
        fd = _open(os.path.join(path, "md.0"))
        try:
            os.lseek(fd, md_pos, os.SEEK_SET)
            block = b""
            while len(block) < md_size:
                part = os.read(fd, md_size - len(block))
                if not part:
                    break
                block += part
        finally:
            os.close(fd)
    except OSError:
        return None
    head = 8 + 16 * writers
    if not ids or len(block) < head:
        return None
    sizes = struct.unpack_from(f"<{writers}Q", block, 8)
    per = 2 * record[2] + 1
    pos, ends = head, []
    for w in range(writers):
        mine = block[pos : pos + sizes[w]]
        pos += sizes[w]
        flushed = sum(record[3 + w * per + 2 * f + 1] for f in range(record[2]))
        if not sizes[w]:
            ends.append(starts[w])
            continue
        id_len = next((len(i) for i in ids if mine.startswith(i)), 0)
        header = (id_len + 8 + 7) // 8 * 8
        if not id_len or len(mine) < header + 24:
            return None
        (length,) = struct.unpack_from("<Q", mine, id_len)
        (data_size,) = struct.unpack_from("<Q", mine, header + 16)
        if length != len(mine) - header or data_size < flushed:
            return None
        ends.append(starts[w] + data_size - flushed)
    return ends


def _format_ids(path: str) -> list[bytes]:
    """The FFS format IDs in a BP5 meta-metadata file: per format, the ID's length,
    the description's length, the ID, the description."""
    data = _read(path)
    ids, pos = [], 0
    while pos + 16 <= len(data):
        id_len, info_len = struct.unpack_from("<QQ", data, pos)
        if pos + 16 + id_len + info_len > len(data):
            break
        ids.append(data[pos + 16 : pos + 16 + id_len])
        pos += 16 + id_len + info_len
    return ids


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
    folder, expanded on this machine.

    The folder outlives the run: a later run of the same scene finds links into
    an output that is gone or another run's, and a reader given those segfaults
    the server. So every link is pointed at ``path`` again, and anything that is
    not in ``path`` goes."""
    folder = expand(folder)
    os.makedirs(folder, exist_ok=True)
    names = set(os.listdir(path))
    for name in os.listdir(folder):
        link = os.path.join(folder, name)
        if name != "md.idx" and (
            name not in names
            or not os.path.islink(link)
            or os.readlink(link) != os.path.join(path, name)
        ):
            if os.path.isdir(link) and not os.path.islink(link):
                shutil.rmtree(link)
            else:
                os.remove(link)
    for name in names:
        link = os.path.join(folder, name)
        if name != "md.idx" and not os.path.lexists(link):
            os.symlink(os.path.join(path, name), link)
    index = _read(os.path.join(path, "md.idx"))
    part = f"{folder.rstrip('/')}.idx"
    with open(part, "wb") as handle:
        handle.write(index[: _cut(index, steps)])
    os.replace(part, os.path.join(folder, "md.idx"))
    return folder


def source_readiness(
    path: str, reader: str | None, pattern: bool, settle: float = FILE_SETTLE
) -> tuple[str | None, bool, Any]:
    """For a growing source about to be opened: why it cannot be read yet (or None),
    whether it can be read without waiting for it to settle (a BP5 folder can once
    one step in it is surely whole), and what is on disk. A pattern can be read
    once one file it matches has stopped changing (``FILE_SETTLE``)."""
    key = disk_key(path)
    if pattern:
        if not key:
            # Nothing matches yet: what the run writes beside it says it goes on.
            return "no file matches it yet", True, ("absent", _beside(path))
        if len(fresh_files(key, time.time_ns(), settle)) == len(key):
            return "its first file is still being written", True, key
        return None, True, key
    if not _there(path):
        return "it is not there yet", False, ("absent", _beside(path))
    if not is_bp(path, reader):
        return None, True, key
    state = check(path)
    return state.reason, state.clear or whole_steps(state) > 0, key


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
    if op == "is_bp":
        return is_bp(args["path"], args.get("reader"))
    if op == "check":
        return asdict(check(args["path"]))
    if op == "grown":
        return [disk_key(args["path"]), asdict(check(args["path"]))]
    if op == "write_index":
        return write_index(args["path"], args["folder"], int(args["steps"]))
    if op == "source_readiness":
        return list(
            source_readiness(
                args["path"],
                args.get("reader"),
                args["pattern"],
                float(args.get("settle", FILE_SETTLE)),
            )
        )
    raise ValueError(f"no disk operation {op!r}")
