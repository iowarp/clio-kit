"""Parts: build123d scripts run in their own process, with their exports kept together.

A part is a folder ``<parts_dir>/<name>`` holding the script (``part.py``), what it
made (``part.stl``, ``part.step``), the measurements (``build.json``) and the view
the person last saw (``view.json``). The folder path is the part's id.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

#: Seconds a script may run before it is stopped (CAD_BUILD_TIMEOUT overrides).
BUILD_TIMEOUT = 600.0


class PartError(Exception):
    """A failure with a short code the agent can act on."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message

    def text(self) -> str:
        return f"{self.message} [{self.code}]"


def parts_dir() -> Path:
    env = os.environ.get("CAD_PARTS_DIR")
    return Path(env).expanduser() if env else Path.home() / ".clio" / "cad" / "parts"


def slug(name: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-.")
    if not text:
        raise PartError("bad_name", "Give the part a name with letters or digits.")
    return text[:80]


def part_folder(name: str, out_dir: str = "") -> Path:
    base = Path(out_dir).expanduser() if out_dir else parts_dir()
    return (base / slug(name)).resolve()


def resolve(part: str) -> Path:
    """A part folder from its path or its name in the default parts folder."""
    path = Path(part).expanduser()
    if not path.is_absolute():
        path = parts_dir() / slug(part)
    if path.suffix.lower() in (".stl", ".step", ".json", ".py"):
        path = path.parent
    if not (path / "build.json").is_file():
        raise PartError("part_not_found", f"No built part at {path}; build it with build_part.")
    return path.resolve()


def _timeout() -> float:
    try:
        return float(os.environ.get("CAD_BUILD_TIMEOUT", BUILD_TIMEOUT))
    except ValueError:
        return BUILD_TIMEOUT


def build(name: str, code: str, out_dir: str = "") -> dict[str, Any]:
    """Run ``code`` as the part's script and return its record."""
    if not code.strip():
        raise PartError("code_missing", "build_part needs build123d code that sets result.")
    folder = part_folder(name, out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    for stale in ("part.stl", "part.step", "build.json"):
        (folder / stale).unlink(missing_ok=True)
    (folder / "part.py").write_text(code, encoding="utf-8")
    timeout = _timeout()
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "cad_mcp.runner", str(folder)],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise PartError(
            "build_timeout",
            f"The script ran past {timeout:g} s and was stopped; simplify it or raise "
            "CAD_BUILD_TIMEOUT.",
        ) from exc
    built = folder / "build.json"
    if not built.is_file():
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-12:]
        raise PartError(
            "build_crashed",
            "The build process stopped without a result:\n" + "\n".join(tail),
        )
    result = json.loads(built.read_text(encoding="utf-8"))
    if not result.get("ok"):
        raise PartError(
            "script_failed",
            f"The script failed: {result.get('error')}\n{result.get('traceback', '')}".rstrip(),
        )
    return record(folder)


def record(folder: Path) -> dict[str, Any]:
    data = json.loads((folder / "build.json").read_text(encoding="utf-8"))
    data.pop("ok", None)
    stl = folder / "part.stl"
    view_file = folder / "view.json"
    return {
        "part": str(folder),
        "name": folder.name,
        "stl": str(stl),
        "step": str(folder / "part.step"),
        "script": str(folder / "part.py"),
        "triangles": stl_triangles(stl),
        "view": json.loads(view_file.read_text(encoding="utf-8")) if view_file.is_file() else {},
        **data,
    }


def stl_triangles(path: Path) -> int:
    with path.open("rb") as handle:
        head = handle.read(84)
    if len(head) == 84 and not head[:5].lower().startswith(b"solid"):
        return int(np.frombuffer(head[80:84], "<u4")[0])
    return len(read_stl(path)) // 9


def read_stl(path: Path) -> np.ndarray:
    """The triangle corners of an STL as a flat float32 array (x, y, z per corner)."""
    raw = path.read_bytes()
    if len(raw) >= 84:
        count = int(np.frombuffer(raw[80:84], "<u4")[0])
        if 84 + 50 * count == len(raw):
            dtype = np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")])
            return np.frombuffer(raw[84:], dtype, count)["v"].reshape(-1).astype(np.float32)
    corners = re.findall(rb"vertex\s+(\S+)\s+(\S+)\s+(\S+)", raw)
    return np.array(corners, dtype=np.float32).reshape(-1)


def remember_view(folder: Path, view: dict[str, Any]) -> None:
    (folder / "view.json").write_text(json.dumps(view), encoding="utf-8")


def listing(out_dir: str = "") -> list[dict[str, Any]]:
    base = Path(out_dir).expanduser() if out_dir else parts_dir()
    if not base.is_dir():
        return []
    found = []
    for built in sorted(base.glob("*/build.json")):
        try:
            data = json.loads(built.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("ok"):
            found.append(
                {
                    "part": str(built.parent),
                    "name": built.parent.name,
                    "size": data.get("size"),
                    "volume": data.get("volume"),
                }
            )
    return found
