"""Find the slicer already installed on this machine. This server never downloads one.

The slicer is resolved in this order, and the first answer is final:

1. ``SLICER_BIN``: the full path (or the command name) of PrusaSlicer or OrcaSlicer.
   When it is set and does not name a program here, the answer is ``slicer_missing``
   naming that value; the search below is not tried (no silent fallback).
2. ``PATH``: ``prusa-slicer``, ``PrusaSlicer``, then ``orca-slicer``, ``OrcaSlicer``.
3. The places the official installers put them on macOS and Windows.

The slicers are separate programs (all AGPL-3.0); this server runs the unmodified
program as a separate process and reads the files it writes.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from ..errors import SlicerError

ENV_BIN = "SLICER_BIN"
ENV_PROFILES = "SLICER_PROFILES"

PRUSA = "prusaslicer"
ORCA = "orcaslicer"
NAMES = {PRUSA: "PrusaSlicer", ORCA: "OrcaSlicer"}

#: Command names searched on PATH, in order of preference.
PATH_NAMES = ("prusa-slicer", "PrusaSlicer", "orca-slicer", "OrcaSlicer")
#: Where the official installers put the programs, when they are not on PATH.
KNOWN_LOCATIONS = (
    "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
    "/Applications/Original Prusa Drivers/PrusaSlicer.app/Contents/MacOS/PrusaSlicer",
    "/Applications/OrcaSlicer.app/Contents/MacOS/OrcaSlicer",
    r"C:\Program Files\Prusa3D\PrusaSlicer\prusa-slicer-console.exe",
    r"C:\Program Files\OrcaSlicer\orca-slicer.exe",
)

MISSING_HELP = (
    "This server slices with PrusaSlicer or OrcaSlicer already installed on this "
    "machine; it never downloads a slicer. Install PrusaSlicer (prusa-slicer) or "
    "OrcaSlicer (orca-slicer), or set SLICER_BIN to the full path of the slicer "
    "program (for example /usr/bin/prusa-slicer, or "
    "/Applications/PrusaSlicer.app/Contents/MacOS/PrusaSlicer on a Mac), then try again."
)


@dataclass(frozen=True)
class Slicer:
    """An installed slicer this server can run."""

    kind: str  # PRUSA or ORCA
    binary: Path
    version: str
    profiles: Path | None

    @property
    def name(self) -> str:
        return NAMES[self.kind]

    def describe(self) -> str:
        return f"{self.name} {self.version}".strip()


def _kind_from_name(path: Path) -> str | None:
    lowered = path.name.lower()
    if "prusa" in lowered:
        return PRUSA
    if "orca" in lowered:
        return ORCA
    return None


def _probe(binary: Path) -> tuple[str | None, str]:
    """Ask the program what it is: (kind, version) from its ``--help`` banner."""
    try:
        done = subprocess.run(
            [str(binary), "--help"],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
            stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.SubprocessError):
        return None, ""
    banner = (done.stdout or done.stderr or "").strip().splitlines()
    first = banner[0] if banner else ""
    match = re.match(r"(PrusaSlicer|OrcaSlicer)[-\s]*([0-9][\w.+-]*)", first)
    if not match:
        return _kind_from_name(binary), ""
    kind = PRUSA if match.group(1) == "PrusaSlicer" else ORCA
    return kind, match.group(2).split("+")[0]


def _profile_folders(kind: str, binary: Path) -> list[Path]:
    real = binary.resolve()
    here = real.parent
    product = NAMES[kind]
    return [
        here / "resources" / "profiles",
        here.parent / "resources" / "profiles",
        here.parent / "Resources" / "profiles",
        here.parent / "share" / product / "profiles",
        here.parent / "share" / product.lower() / "profiles",
        Path("/usr/share") / product / "profiles",
        Path("/usr/share") / ("prusa-slicer" if kind == PRUSA else "orca-slicer") / "profiles",
    ]


def find_profiles(kind: str, binary: Path) -> Path | None:
    """The folder of vendor presets the slicer ships, or ``SLICER_PROFILES``."""
    configured = os.environ.get(ENV_PROFILES, "").strip()
    if configured:
        path = Path(configured).expanduser()
        return path if path.is_dir() else None
    pattern = "*.ini" if kind == PRUSA else "*.json"
    for folder in _profile_folders(kind, binary):
        if folder.is_dir() and any(folder.glob(pattern)):
            return folder
    return None


@lru_cache(maxsize=8)
def _describe(path: str, mtime: int) -> tuple[str | None, str]:
    del mtime
    return _probe(Path(path))


def _make(binary: Path) -> Slicer | None:
    try:
        mtime = binary.stat().st_mtime_ns
    except OSError:
        return None
    kind, version = _describe(str(binary), mtime)
    if kind is None:
        return None
    return Slicer(kind, binary, version, find_profiles(kind, binary))


def locate() -> Slicer:
    """The installed slicer to use.

    Raises:
        SlicerError: ``slicer_missing`` when none is found, or ``SLICER_BIN`` names a
            program that is not here or is not PrusaSlicer or OrcaSlicer.
    """
    configured = os.environ.get(ENV_BIN, "").strip()
    if configured:
        found = shutil.which(os.path.expanduser(configured))
        if not found:
            raise SlicerError(
                "slicer_missing",
                f"SLICER_BIN is set to {configured!r}, which is not a program on this "
                f"machine. {MISSING_HELP}",
                looked_for=[configured],
            )
        slicer = _make(Path(found))
        if slicer is None:
            raise SlicerError(
                "slicer_missing",
                f"SLICER_BIN names {found}, which did not answer as PrusaSlicer or "
                f"OrcaSlicer. {MISSING_HELP}",
                looked_for=[configured],
            )
        return slicer
    candidates: list[Path] = []
    for name in PATH_NAMES:
        found = shutil.which(name)
        if found:
            candidates.append(Path(found))
    candidates.extend(Path(p) for p in KNOWN_LOCATIONS if Path(p).is_file())
    for candidate in candidates:
        slicer = _make(candidate)
        if slicer is not None:
            return slicer
    raise SlicerError(
        "slicer_missing",
        "No 3D-printing slicer was found on this machine (looked for "
        + ", ".join(PATH_NAMES)
        + f" on PATH and in the usual install folders). {MISSING_HELP}",
        looked_for=list(PATH_NAMES),
    )


__all__ = ["ENV_BIN", "ENV_PROFILES", "NAMES", "ORCA", "PRUSA", "Slicer", "find_profiles", "locate"]
