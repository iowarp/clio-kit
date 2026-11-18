"""Run one build123d script in its own process and export what it builds.

``python -m cad_mcp.runner <part_dir>`` executes ``<part_dir>/part.py`` with
``from build123d import *`` already done, takes the shape it leaves in ``result``
(or ``part``, or the last shape or BuildPart it made), and writes ``part.stl`` and
``part.step`` next to it with ``build.json`` (size, volume, area, solids,
validity). A failure goes to ``build.json`` as ``{"ok": false, "error": ...}``
with the traceback.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any


def _pick(namespace: dict[str, Any]) -> Any:
    import build123d as bd

    for key in ("result", "part"):
        value = namespace.get(key)
        if value is not None:
            return value.part if isinstance(value, bd.BuildPart) else value
    found = None
    for key, value in namespace.items():
        if key.startswith("_"):
            continue
        if isinstance(value, bd.BuildPart):
            found = value.part
        elif isinstance(value, bd.Shape) and not isinstance(value, (bd.Vertex, bd.Edge)):
            found = value
    return found


def run(part_dir: Path) -> dict[str, Any]:
    import build123d as bd

    code = (part_dir / "part.py").read_text(encoding="utf-8")
    namespace: dict[str, Any] = {"__name__": "__part__"}
    exec("from build123d import *", namespace)  # noqa: S102 - the script expects it
    exec(compile(code, str(part_dir / "part.py"), "exec"), namespace)  # noqa: S102
    shape = _pick(namespace)
    if shape is None:
        raise ValueError(
            "The script made no shape: assign the finished part to result "
            "(for example result = extrude(sketch, 1))."
        )
    if not isinstance(shape, bd.Shape):
        raise TypeError(f"result is a {type(shape).__name__}, not a build123d shape.")
    box = shape.bounding_box()
    size = [float(box.size.X), float(box.size.Y), float(box.size.Z)]
    extent = max(size) or 1.0
    bd.export_stl(shape, str(part_dir / "part.stl"), tolerance=extent * 5e-4, angular_tolerance=0.1)
    bd.export_step(shape, str(part_dir / "part.step"))
    solids = shape.solids()
    return {
        "lower": [float(box.min.X), float(box.min.Y), float(box.min.Z)],
        "upper": [float(box.max.X), float(box.max.Y), float(box.max.Z)],
        "size": size,
        "volume": float(shape.volume),
        "area": float(shape.area),
        "solids": len(solids),
        "valid": bool(valid() if callable(valid := shape.is_valid) else valid),
        "shape_type": type(shape).__name__,
        "build123d": getattr(bd, "__version__", ""),
    }


def main() -> None:
    part_dir = Path(sys.argv[1])
    try:
        record: dict[str, Any] = {"ok": True, **run(part_dir)}
    except BaseException as exc:  # noqa: BLE001 - report every failure of the script
        record = {
            "ok": False,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(limit=8),
        }
    (part_dir / "build.json").write_text(json.dumps(record, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
