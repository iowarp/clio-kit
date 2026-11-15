"""export_animation renders the scene's time steps and encodes them.

    pvpython animation_check.py OUT_DIR ffmpeg|vtk

``vtk`` hides ffmpeg from PATH, so VTK's OGG Theora writer is used. Prints
``ANIMATION=<json>``.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node.engine import CommandError, Engine


def main(out: Path, encoder: str) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    for step in range(3):
        simple.SaveData(
            str(out / f"wave_{step}.vti"),
            proxy=simple.Wavelet(
                WholeExtent=[0, 8, 0, 8, 0, 8], Maximum=200 + 40 * step
            ),
        )
    engine = Engine(out / "folder")
    engine.run("open_view", {"path": str(out / "wave_*.vti"), "new": True})
    engine.run("set_timestep", {"index": 1})
    if encoder == "vtk":
        os.environ["PATH"] = ""
    result = engine.run(
        "export_animation",
        {
            "path": str(out / "clip.mp4"),
            "width": 321,
            "height": 240,
            "fps": 2,
            "labels": ["one", "two", "three"],
        },
    )
    report: dict[str, Any] = {
        "format": result["format"],
        "frames": result["frames"],
        "width": result["width"],
        "exists": Path(result["path"]).is_file()
        and Path(result["path"]).stat().st_size > 0,
        "suffix": Path(result["path"]).suffix,
        "time_index_after": engine.time_summary()["index"],
        "sources_left": sorted(n for (n, _) in simple.GetSources()),
    }
    try:
        engine.run(
            "export_animation", {"path": str(out / "bad.mp4"), "labels": ["one"]}
        )
    except CommandError as exc:
        report["bad_labels"] = exc.code
    return report


if __name__ == "__main__":
    print("ANIMATION=" + json.dumps(main(Path(sys.argv[1]), sys.argv[2])), flush=True)
