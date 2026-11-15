"""A scene's colour maps survive a restart and do not leak between scenes.

    pvpython colors_check.py OUT_DIR build|restore

``build`` colours a first scene's field with Jet, opens a second scene (which
must start with ParaView's own colour map, not Jet), then builds a scene with
Viridis, a legend and opacity. ``restore`` reopens it in a fresh process.
Each prints ``COLORS=<json>``.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node.engine import Engine


def state(engine: Engine) -> dict[str, Any]:
    lut = simple.GetColorTransferFunction("RTData")
    points = [round(v, 3) for v in lut.RGBPoints]
    reps = {}
    for item in engine.scene["objects"]:
        rep = simple.GetDisplayProperties(engine.proxies[item["id"]], engine.view)
        reps[item["id"]] = [list(rep.ColorArrayName), round(float(rep.Opacity), 3)]
    return {
        "points": hashlib.sha1(str(points).encode()).hexdigest(),
        "range": [points[0], points[-4]],
        "preset": str(lut.NameOfLastPresetApplied),
        "recorded": {
            o["id"]: (o.get("display") or {}).get("colormap")
            for o in engine.scene["objects"]
        },
        "reps": reps,
    }


def main(out: Path, mode: str) -> dict[str, Any]:
    data = out / "wave.vti"
    engine = Engine(out / "folder")
    if mode == "build":
        out.mkdir(parents=True, exist_ok=True)
        simple.SaveData(str(data), proxy=simple.Wavelet())
        other = out / "other.vti"
        simple.SaveData(
            str(other), proxy=simple.Wavelet(WholeExtent=[0, 8, 0, 8, 0, 8])
        )
        engine.run("open_view", {"path": str(other), "new": True})
        engine.run(
            "set_representation",
            {
                "object": "o1",
                "representation": "Surface",
                "color_by": "RTData",
                "colormap": "Jet",
            },
        )
        engine.run("open_view", {"path": str(data), "new": True})
        fresh = str(simple.GetColorTransferFunction("RTData").NameOfLastPresetApplied)
        engine.run(
            "create_filter",
            {
                "filter": "Contour",
                "properties": {"ContourBy": ["POINTS", "RTData"], "Isosurfaces": [150]},
            },
        )
        engine.run(
            "set_representation",
            {"object": "o2", "colormap": "Viridis", "opacity": 0.7, "scalar_bar": True},
        )
        engine.run(
            "create_filter",
            {
                "filter": "Slice",
                "input": "o1",
                "properties": {"SliceType.Normal": [0, 0, 1]},
            },
        )
        engine.run(
            "set_representation",
            {"object": "o3", "color_by": "RTData", "colormap": "Inferno"},
        )
        # The field's one legend stays on (it is the Contour's) after the Slice changes.
        lut = simple.GetColorTransferFunction("RTData")
        legend = simple.GetScalarBar(lut, engine.view)
        (out / "scene").write_text(engine.scene["scene"])
        return {
            "preset_after_new_scene": fresh,
            "legend": bool(legend.Visibility),
            **state(engine),
        }
    engine.run("open_view", {"scene": (out / "scene").read_text()})
    legend = simple.GetScalarBar(simple.GetColorTransferFunction("RTData"), engine.view)
    return {"legend": bool(legend.Visibility), **state(engine)}


if __name__ == "__main__":
    print("COLORS=" + json.dumps(main(Path(sys.argv[1]), sys.argv[2])), flush=True)
