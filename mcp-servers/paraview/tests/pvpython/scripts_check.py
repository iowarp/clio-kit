"""Scene scripts build inside the scene, leave nothing when they fail, and replay.

    pvpython scripts_check.py OUT_DIR

Prints ``SCRIPTS=<json>`` with what each step left in the scene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node.engine import CommandError, Engine

BUILD = """
other = OpenDataFile(r"{other}")
Show(other, view)
Hide(other, view)
shell = Contour(Input=objects["o1"], ContourBy=["POINTS", "RTData"], Isosurfaces=[150])
shown = Show(shell, view)
ColorBy(shown, None)
shown.DiffuseColor = [1.0, 0.0, 0.0]
"""


def sources() -> int:
    return len(simple.GetSources())


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    first, other, third = out / "first.vti", out / "other.vti", out / "third.vti"
    for path in (first, other, third):
        simple.SaveData(str(path), proxy=simple.Wavelet(WholeExtent=[0, 8, 0, 8, 0, 8]))
    engine = Engine(out / "folder")
    opened = engine.run("open_view", {"path": str(first), "new": True})
    report: dict[str, Any] = {"scene": opened["scene"]}
    built = engine.run(
        "run_scene_script", {"code": BUILD.format(other=other), "label": "shell"}
    )
    report["added"] = [
        {"id": o["id"], "kind": o["kind"], "type": o["type"], "display": o["display"]}
        for o in built["added"]
    ]
    report["scripts"] = built["scene"]["scripts"]
    before = sources()
    try:
        engine.run(
            "run_scene_script",
            {"code": "Contour(Input=objects['o1'])\nraise ValueError('no')"},
        )
    except CommandError as exc:
        report["failed"] = exc.code
    report["sources_after_failure"] = sources() - before
    views = len(simple.GetViews())
    try:
        engine.run("run_scene_script", {"code": "CreateRenderView()"})
    except CommandError as exc:
        report["own_view"] = exc.code
    report["views_after_refusal"] = len(simple.GetViews()) - views
    shell = next(o["id"] for o in report["added"] if o["type"] == "Contour")
    engine.run("remove_scene_object", {"object": shell})
    # Another scene, then this one again: its scripts replay.
    engine.run("open_view", {"path": str(third), "new": True})
    restored = engine.run("open_view", {"scene": report["scene"]})
    report["restored"] = [
        {"id": o["id"], "kind": o["kind"], "visible": o["display"].get("visible", True)}
        for o in restored["objects"]
    ]
    report["restored_scripts"] = restored["scripts"]
    return report


if __name__ == "__main__":
    print("SCRIPTS=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
