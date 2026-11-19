"""Whether an ADIOS2 BP5 output that is still being written can be read yet.

ParaView's ADIOS2 reader takes the process down when it opens or re-reads a BP5
folder whose index names a step that the metadata does not hold yet, and it waits
forever for data that is not on disk. A writer adds a step's data, then its
metadata, then its index record, but a reader on another machine of a shared
file system can see them land in any order. So a growing BP5 folder goes to the
reader only when its files agree: at least one whole step in the index
(``md.idx``), the metadata (``md.0``) holding the last one, the meta-metadata
(``mmd.0``) written, and each data file (``data.N``) reaching into the step's
last block.

The index does not give the size of a step's last block, so a data file that
reaches into it may still be short. ``Readiness.clear`` says the data reaches a
whole block estimate past it; otherwise a caller waits for the folder to stop
changing for a moment before reading it.

Other formats (BP4, a single file) are not checked: ``check`` reports them ready.
"""

from __future__ import annotations

import itertools
import os
import struct
from dataclasses import dataclass
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


def is_bp(path: str, reader: str | None = None) -> bool:
    """Whether ``path`` is (or will be) an ADIOS2 BP folder."""
    if reader and "ADIOS2" in reader:
        return True
    if os.path.isfile(os.path.join(path, "md.idx")):
        return True
    return path.rstrip("/").lower().endswith((".bp", ".bp4", ".bp5"))


def _size(path: str) -> int:
    try:
        return os.path.getsize(path)
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
        return Readiness()
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


def source_readiness(
    path: str, reader: str | None, pattern: bool
) -> tuple[str | None, bool, Any]:
    """For a growing source about to be opened: why it cannot be read yet (or None),
    whether it can be read without waiting for it to settle, and what is on disk."""
    from .sources import disk_key

    key = disk_key(path)
    if pattern:
        return (None if key else "no file matches it yet"), True, key
    if not os.path.exists(path):
        return "it is not there yet", False, key
    if not is_bp(path, reader):
        return None, True, key
    state = check(path)
    return state.reason, state.clear, key
