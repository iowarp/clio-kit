"""Saved slices: each slice keeps its G-code, toolpaths and a record, by slice id.

A slice id is ``s`` and ten hex digits. Its folder holds ``record.json`` (what was
sliced, with which presets, the slicer's estimates, warnings and the per-layer index),
``slice.gcode`` (the slicer's output, plain text), ``layers.bin`` (the toolpaths the
view draws, see :mod:`.gcode.parser`), ``settings.ini`` (the flattened presets the
slicer loaded) and ``slicer.log``. The view's last layer and visible types are saved in
the record, so reopening the view shows what the person last saw.

Folders live in clio-kit's data folder (``platformdirs.user_data_dir("clio-kit")``,
under ``slicer``), which a host that confines its servers lets them write;
``SLICER_MCP_HOME`` overrides it. Nothing here expires.
"""

from __future__ import annotations

import json
import os
import re
import secrets
import tempfile
import time
from pathlib import Path
from typing import Any

from .errors import SlicerError

SLICE_ID = re.compile(r"^s[0-9a-f]{10}$")
ENV_HOME = "SLICER_MCP_HOME"
RECORD = "record.json"


def home() -> Path:
    """The server's own folder: saved slices and the slicer's pinned settings folder."""
    configured = os.environ.get(ENV_HOME, "").strip()
    if configured:
        return Path(configured).expanduser()
    import platformdirs

    return Path(platformdirs.user_data_dir("clio-kit", appauthor=False)) / "slicer"


def datadir(kind: str) -> Path:
    """The settings folder the slicer runs with, so the person's own is never touched."""
    path = home() / "datadir" / kind
    path.mkdir(parents=True, exist_ok=True)
    return path


def new_slice_id() -> str:
    return "s" + secrets.token_hex(5)


def slice_dir(slice_id: str) -> Path:
    if not SLICE_ID.match(slice_id or ""):
        raise SlicerError(
            "slice_id_invalid", f"{slice_id!r} is not a slice id (it looks like s0123456789)."
        )
    return home() / "slices" / slice_id


def save_record(record: dict[str, Any]) -> None:
    """Write a slice record atomically.

    Raises:
        SlicerError: ``slice_not_saved`` when the folder cannot be written.
    """
    folder = slice_dir(str(record.get("slice_id")))
    record = {**record, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        folder.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=folder, prefix=".record-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(tmp, folder / RECORD)
    except OSError as exc:
        raise SlicerError(
            "slice_not_saved", f"The slice could not be saved in {folder}: {exc}"
        ) from exc


def load_record(slice_id: str) -> dict[str, Any]:
    """Read a slice record.

    Raises:
        SlicerError: ``slice_id_invalid``, ``slice_not_found`` or ``slice_unreadable``.
    """
    path = slice_dir(slice_id) / RECORD
    if not path.is_file():
        raise SlicerError(
            "slice_not_found",
            f"No saved slice {slice_id}. Slice the model again with slice.",
        )
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SlicerError(
            "slice_unreadable", f"The saved slice {slice_id} cannot be read: {exc}"
        ) from exc
    if not isinstance(record, dict) or not isinstance(record.get("layers"), list):
        raise SlicerError("slice_unreadable", f"The saved slice {slice_id} is malformed.")
    return record


def remember_view(slice_id: str, view: dict[str, Any]) -> dict[str, Any]:
    """Save the view's layer and visible types in the record; returns the record."""
    record = load_record(slice_id)
    record["view"] = {**(record.get("view") or {}), **view}
    save_record(record)
    return record


__all__ = [
    "ENV_HOME",
    "SLICE_ID",
    "datadir",
    "home",
    "load_record",
    "new_slice_id",
    "remember_view",
    "save_record",
    "slice_dir",
]
