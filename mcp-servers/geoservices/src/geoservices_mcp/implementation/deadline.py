"""An overall time limit for one read, so a failing server cannot hold a tool call forever.

GDAL and the point-cloud readers time out each HTTP request and retry dropped
connections, but a windowed read is hundreds or thousands of requests. A server that
keeps resetting connections made one ``read_raster_window`` call run for over 40
minutes with no result. Here the read runs in a worker thread that writes to a staging
file, and the call returns a typed failure when the limit passes. The worker cannot be
interrupted inside GDAL, so it finishes in the background; when it does, its staging
file is deleted instead of being moved to the output path, so no file appears after the
agent was told the read failed.
"""

from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

from .common import GeoServiceError

#: Default limit for one read, in seconds (``GEOSERVICES_READ_DEADLINE_S``).
DEFAULT_READ_DEADLINE_S = 600


def read_deadline_s() -> float:
    """The configured limit for one read, in seconds."""
    raw = os.environ.get("GEOSERVICES_READ_DEADLINE_S", "")
    try:
        value = float(raw) if raw else DEFAULT_READ_DEADLINE_S
    except ValueError as exc:
        raise GeoServiceError(
            f"GEOSERVICES_READ_DEADLINE_S must be a number of seconds, not {raw!r}"
        ) from exc
    if value <= 0:
        raise GeoServiceError("GEOSERVICES_READ_DEADLINE_S must be above 0")
    return value


def staging_path(out: Path) -> Path:
    """Where a read writes before it is moved to ``out`` (same suffix, so the writer
    picks the same format)."""
    return out.with_name(f".partial-{out.name}")


async def run_with_deadline(
    tool: str, out: Path, read: Callable[[Path], dict[str, Any]]
) -> dict[str, Any]:
    """Run ``read(staging)`` in a thread; move its file to ``out`` if it finishes in time.

    Raises:
        GeoServiceError: the read did not finish within :func:`read_deadline_s`.
    """
    deadline = read_deadline_s()
    staging = staging_path(out)
    abandoned = threading.Event()
    lock = threading.Lock()

    def run() -> dict[str, Any]:
        result = read(staging)
        with lock:
            if abandoned.is_set():
                staging.unlink(missing_ok=True)
                return result
            staging.replace(out)
        result["output_path"] = str(out)
        return result

    try:
        return await asyncio.wait_for(asyncio.to_thread(run), timeout=deadline)
    except (TimeoutError, asyncio.TimeoutError):  # distinct classes on Python 3.10
        with lock:
            abandoned.set()
            staging.unlink(missing_ok=True)
        raise GeoServiceError(
            f"{tool} did not finish within {deadline:g} s: the server is slow or keeps "
            "dropping connections. Read a smaller region or fewer tiles, use the dataset's "
            "VRT or a lower resolution, or try another source."
        ) from None
