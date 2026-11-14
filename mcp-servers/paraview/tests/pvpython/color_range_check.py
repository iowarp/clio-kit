"""A colour range set by hand reaches the field's colour map and opacity ramp.

    pvpython color_range_check.py OUT_DIR

Shows a wavelet as a volume with ``color_range`` [100, 200], steps nothing else, and
prints ``RANGE=<json>`` with the colour map's and the opacity ramp's end points,
then gives a contour of it one plain colour and prints that colour.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node.engine import Engine


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    data = out / "range.vti"
    simple.SaveData(str(data), proxy=simple.Wavelet())
    engine = Engine(out / "range-folder")
    engine.run("open_view", {"path": str(data), "new": True})
    engine.run(
        "set_representation",
        {
            "object": "o1",
            "representation": "Volume",
            "color_by": "RTData",
            "color_range": [100.0, 200.0],
        },
    )
    lut = simple.GetColorTransferFunction("RTData").RGBPoints
    pwf = simple.GetOpacityTransferFunction("RTData").Points
    engine.run(
        "create_filter",
        {
            "filter": "Contour",
            "properties": {"ContourBy": ["POINTS", "RTData"], "Isosurfaces": [150]},
        },
    )
    engine.run("set_representation", {"object": "o2", "solid_color": "#ff8000"})
    shell = simple.GetDisplayProperties(engine.proxies["o2"], engine.view)
    return {
        "color": [round(lut[0], 3), round(lut[-4], 3)],
        "opacity": [round(pwf[0], 3), round(pwf[-4], 3)],
        "solid": [round(v, 2) for v in shell.DiffuseColor],
        "solid_coloured_by": str(shell.ColorArrayName[1]),
    }


if __name__ == "__main__":
    print("RANGE=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
