"""Drive the view's panel against a real engine; run with pvpython.

    pvpython panel_check.py OUT_DIR

Prints one JSON object: what each panel action did to the scene.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node import filters
from paraview_mcp.node.engine import Engine
from paraview_mcp.node.workbench import Workbench


class State(dict):
    """The few trame state behaviours the workbench uses."""

    def __getattr__(self, name: str) -> Any:
        return self.get(name)

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value

    def __enter__(self) -> State:  # noqa: PYI034 - a test stand-in
        return self

    def __exit__(self, *_: object) -> None:
        return None


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    source = simple.Gradient(Input=simple.Wavelet(), ScalarArray=["POINTS", "RTData"])
    source.ResultArrayName = "Gradient"
    data = out / "wavelet_gradient.vti"
    simple.SaveData(str(data), proxy=source)
    simple.Delete(source)

    engine = Engine(out / "folder")
    state = State(clio_index=0)
    bench = Workbench(engine, state)
    engine.listeners.append(bench.refresh)
    engine.run("open_view", {"path": str(data), "new": True})
    report: dict[str, Any] = {
        "offered": [f["name"] for f in state.clio_offered if not f["disabled"]]
    }

    made = {}
    for entry in filters.CATALOG:
        bench.select("o1")
        bench.open_form(entry["name"])
        values = dict(state.clio_form)
        if entry["name"] == "Calculator":
            values["Function"] = "RTData*2"
        bench.apply_form(values)
        made[entry["name"]] = {"error": state.clio_error, "active": state.clio_active}
    report["made"] = made
    report["objects"] = [(o["id"], o["type"]) for o in engine.scene["objects"]]

    bench.select("o1")
    bench.display("color_by", "Gradient")
    bench.display("component", 2)
    bench.display("scalar_bar", True)
    bench.display("rescale", True)
    bench.display("colormap", "Viridis")
    bench.display("opacity", 0.6)
    bench.display("representation", "Surface With Edges")
    report["display"] = dict(engine.scene["objects"][0]["display"])
    bench.display("colormap", "No Such Map")
    report["bad_colormap"] = state.clio_error
    report["display_after_refusal"] = engine.scene["objects"][0]["display"]["colormap"]

    bench.camera("direction", "-z")
    bench.camera("parallel", True)
    report["camera"] = engine.read_camera()
    bench.set_visible("o2", False)
    bench.remove("o3")
    report["after_remove"] = [o["id"] for o in engine.scene["objects"]]
    report["visible_o2"] = engine.scene["objects"][1]["display"]["visible"]
    report["changes"] = engine.changes.recent()
    report["scene_state_changes"] = engine.run("scene_state", {})["changes"]
    manifest = json.loads(
        (
            out / "folder" / "scenes" / engine.scene["scene"] / "manifest.json"
        ).read_text()
    )
    report["manifest_objects"] = [o["id"] for o in manifest["objects"]]
    report["manifest_parallel"] = manifest["camera"]["parallel_projection"]
    return report


if __name__ == "__main__":
    print("PANEL_REPORT=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
