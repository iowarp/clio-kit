"""Saved globe views, so a view reopens with the layers it showed.

Each ``show_on_globe`` call saves its view as one small JSON record, named by a view id
(``g`` and ten hex digits). A layer read from a file is saved as its path, so reopening
reads the file again; inline GeoJSON is saved as given. Edits a person or the agent
makes in the view (add, remove, restyle a layer) update the record.

Records live in clio-kit's own data folder (``platformdirs.user_data_dir("clio-kit")``,
under ``geo/globe``), which a host that confines the servers it launches lets them
write; ``GEO_MCP_GLOBE_DIR`` overrides it.
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

VIEW_ID = re.compile(r"^g[0-9a-f]{10}$")


class GlobeStoreError(ValueError):
    """A view record cannot be read or written; ``reason`` is a stable code."""

    def __init__(self, reason: str, message: str) -> None:
        super().__init__(message)
        self.reason = reason


def store_dir() -> Path:
    """Where view records are kept."""
    configured = os.environ.get("GEO_MCP_GLOBE_DIR")
    if configured:
        return Path(configured).expanduser()
    import platformdirs

    return Path(platformdirs.user_data_dir("clio-kit", appauthor=False)) / "geo" / "globe"


def new_view_id() -> str:
    return "g" + secrets.token_hex(5)


def _path(view_id: str) -> Path:
    if not VIEW_ID.match(view_id or ""):
        raise GlobeStoreError(
            "view_id_invalid",
            f"{view_id!r} is not a globe view id (it looks like g0123456789).",
        )
    return store_dir() / f"{view_id}.json"


def save_view(record: dict[str, Any]) -> None:
    """Write a view record atomically.

    Raises:
        GlobeStoreError: ``view_not_saved`` when the folder cannot be written.
    """
    target = _path(str(record.get("id")))
    record = {**record, "saved_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=target.parent, prefix=".view-", suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(record, handle)
        os.replace(tmp, target)
    except OSError as exc:
        raise GlobeStoreError(
            "view_not_saved", f"The globe view could not be saved in {target.parent}: {exc}"
        ) from exc


def load_view(view_id: str) -> dict[str, Any]:
    """Read a view record.

    Raises:
        GlobeStoreError: ``view_id_invalid``, ``view_not_found`` or ``view_unreadable``.
    """
    target = _path(view_id)
    if not target.is_file():
        raise GlobeStoreError(
            "view_not_found",
            f"No saved globe view {view_id}. Show the layers again with show_on_globe.",
        )
    try:
        record = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise GlobeStoreError(
            "view_unreadable", f"The saved globe view {view_id} cannot be read: {exc}"
        ) from exc
    if not isinstance(record, dict) or not isinstance(record.get("layers"), list):
        raise GlobeStoreError("view_unreadable", f"The saved globe view {view_id} is malformed.")
    return record


__all__ = [
    "VIEW_ID",
    "GlobeStoreError",
    "load_view",
    "new_view_id",
    "save_view",
    "store_dir",
]
