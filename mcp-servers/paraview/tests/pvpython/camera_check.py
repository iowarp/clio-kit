"""A scene's camera comes back exactly after a restart; run with pvpython.

    pvpython camera_check.py OUT_DIR build|restore

``build`` opens a wavelet, sets a camera off the fitted one (moved and zoomed in)
and saves the scene; ``restore`` reopens it in a fresh process and renders once,
as the view does. Each prints ``CAMERA=<json>``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node import manifest
from paraview_mcp.node.engine import Engine


def rounded(camera: dict[str, Any]) -> dict[str, Any]:
    return {
        key: [round(v, 4) for v in value]
        if isinstance(value, list)
        else value
        if isinstance(value, bool)
        else round(value, 4)
        for key, value in camera.items()
    }


def main(out: Path, mode: str) -> dict[str, Any]:
    engine = Engine(out / "folder")
    if mode == "build":
        out.mkdir(parents=True, exist_ok=True)
        data = out / "wave.vti"
        simple.SaveData(str(data), proxy=simple.Wavelet())
        engine.run("open_view", {"path": str(data), "new": True})
        engine.run(
            "set_camera",
            {
                "position": [30.0, 20.0, 90.0],
                "focal_point": [2.0, 1.0, 0.0],
                "zoom": 1.7,
            },
        )
        engine.scene["camera"] = engine.read_camera()
        manifest.write(engine.root, engine.scene)
        (out / "scene").write_text(engine.scene["scene"])
        return rounded(engine.read_camera())
    engine.run("open_view", {"scene": (out / "scene").read_text()})
    simple.Render(engine.view)  # the view's first render in this process
    return rounded(engine.read_camera())


if __name__ == "__main__":
    print("CAMERA=" + json.dumps(main(Path(sys.argv[1]), sys.argv[2])), flush=True)
