"""A field's default colour range represents it: the 2nd to 98th percentile of the
values shown, not a few extremes.

    pvpython robust_range_check.py OUT_DIR

Writes a field that is a smooth ramp from 0 to 1 except for a small "body" of
points at 100, colours it, and prints ``ROBUST=<json>`` with the colour map's end
points by default, the numpy percentiles they should be, what a scene script that
sets a range keeps, what a script that only colours gets, and Rescale's full range.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy
from paraview import simple

from paraview_mcp.node.engine import Engine

N = 20


def values() -> list[float]:
    out = []
    for z in range(N):
        for y in range(N):
            for x in range(N):
                body = x < 4 and y < 4 and z < 5
                out.append(100.0 if body else x / (N - 1))
    return out


def write(path: Path) -> None:
    extent = f"0 {N - 1} 0 {N - 1} 0 {N - 1}"
    path.write_text(
        f"""<?xml version="1.0"?>
<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">
<ImageData WholeExtent="{extent}" Origin="0 0 0" Spacing="1 1 1">
<Piece Extent="{extent}"><PointData Scalars="w">
<DataArray type="Float32" Name="w" format="ascii">{" ".join(f"{v:g}" for v in values())}</DataArray>
</PointData></Piece></ImageData></VTKFile>
"""
    )


def ends(field: str) -> list[float]:
    points = simple.GetColorTransferFunction(field).RGBPoints
    return [round(points[0], 6), round(points[-4], 6)]


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    write(out / "body.vti")
    engine = Engine(out / "robust-folder")
    engine.run("open_view", {"path": str(out / "body.vti"), "new": True})
    engine.run(
        "set_representation",
        {"object": "o1", "representation": "Surface", "color_by": "w"},
    )
    shown = dict(engine.require_scene()["objects"][0]["display"])
    default = ends("w")
    # The range is measured on the object's data, every value of it.
    expected = [
        round(float(v), 6)
        for v in numpy.percentile(numpy.asarray(values()), [2.0, 98.0])
    ]
    engine.run(
        "run_scene_script",
        {
            "code": "lut = GetColorTransferFunction('w')\n"
            "lut.RescaleTransferFunction(0.0, 0.5)\n",
            "label": "range",
        },
    )
    by_script = ends("w")
    kept = engine.require_scene()["objects"][0]["display"].get("color_range")
    engine.run("set_representation", {"object": "o1", "color_by": "none"})
    engine.run(
        "run_scene_script",
        {
            "code": "c = Calculator(Input=objects['o1'], ResultArrayName='v')\n"
            "c.Function = 'w'\n"
            "rep = Show(c, view)\n"
            "ColorBy(rep, ('POINTS', 'v'))\n"
            "rep.RescaleTransferFunctionToDataRange(True, False)\n"
            "Hide(objects['o1'], view)\n",
            "label": "colour",
        },
    )
    made = engine.require_scene()["objects"][-1]
    made_display = dict(made["display"])
    coloured = ends("v")
    engine.run("set_representation", {"object": made["id"], "rescale": True})
    whole = ends("v")
    state = engine.run("scene_state", {})
    return {
        "default": default,
        "expected": expected,
        "display": {k: shown.get(k) for k in ("auto_range", "data_range")},
        "by_script": by_script,
        "kept": kept,
        "script_colour": coloured,
        "script_display": {
            k: made_display.get(k) for k in ("auto_range", "color_range")
        },
        "rescaled": whole,
        "rescaled_mode": engine.require_scene()["objects"][-1]["display"].get(
            "range_mode"
        ),
        "objects": [o.get("display", {}) for o in state.get("objects", [])],
    }


if __name__ == "__main__":
    print("ROBUST=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
