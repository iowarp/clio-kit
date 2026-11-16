"""Where outputs go and how input files are found.

Runs land in ``chemistry/graspa/<run-name>/`` under ``output_dir`` when given, else
``$CLIO_KIT_ARTIFACTS`` (CLIO sets it to the workspace), else the current directory.
A taken run name gets ``-2``, ``-3`` …; nothing is overwritten.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .errors import GraspaError

ARTIFACTS_ENV = "CLIO_KIT_ARTIFACTS"
SUBFOLDER = ("chemistry", "graspa")


def output_root(output_dir: str | None) -> Path:
    if output_dir:
        base = Path(output_dir).expanduser()
    elif os.environ.get(ARTIFACTS_ENV):
        base = Path(os.environ[ARTIFACTS_ENV]).expanduser()
    else:
        base = Path.cwd()
    return base.resolve().joinpath(*SUBFOLDER)


def slug(text: str, fallback: str = "run") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "-", text.strip()).strip("-.")
    return cleaned[:60] or fallback


def new_run_dir(output_dir: str | None, run_name: str) -> Path:
    root = output_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = slug(run_name)
    candidate, n = root / name, 2
    while candidate.exists():
        candidate = root / f"{name}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def existing_file(path: str, what: str = "file") -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists() and os.environ.get(ARTIFACTS_ENV):
        alt = Path(os.environ[ARTIFACTS_ENV]).expanduser() / p
        if alt.exists():
            p = alt
    if not p.is_file():
        raise GraspaError(
            "not_found",
            f"The {what} {path!r} does not exist. Give the path of an existing file "
            "(absolute, or relative to the workspace).",
            path=str(path),
        )
    return p.resolve()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, data: dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    tmp.replace(path)
