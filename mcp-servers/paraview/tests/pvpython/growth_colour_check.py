"""A followed source's colour maps: one nobody fixed grows with each new step, one
set by hand stays.

    pvpython growth_colour_check.py OUT_DIR

Opens a growing wavelet series, colours a slice by RTData with its own range and the
reader's outline by a second field with a range set by hand, writes a step with
twice the values, checks growth and prints ``GROWTH=<json>`` with both colour maps'
end points before and after.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from paraview import simple

from paraview_mcp.node.engine import Engine


def write(path: Path, scale: float) -> None:
    calc = simple.Calculator(Input=simple.Wavelet(), ResultArrayName="RTData")
    calc.Function = f"RTData*{scale}"
    twin = simple.Calculator(Input=calc, ResultArrayName="Twin")
    twin.Function = "RTData"
    simple.SaveData(str(path), proxy=twin)


def ends(field: str) -> list[float]:
    points = simple.GetColorTransferFunction(field).RGBPoints
    return [round(points[0], 1), round(points[-4], 1)]


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    write(out / "w_0.vti", 1.0)
    engine = Engine(out / "growth-folder")
    engine.run(
        "open_view",
        {
            "path": str(out / "w_*.vti"),
            "new": True,
            "growing": True,
            "descriptor": {"display": {"array": "RTData", "representation": "Slice"}},
        },
    )
    engine.run(
        "set_representation",
        {"object": "o1", "color_by": "Twin", "color_range": [0.0, 100.0]},
    )
    before = {"free": ends("RTData"), "fixed": ends("Twin")}
    write(out / "w_1.vti", 2.0)
    grown = engine.check_growth() or {}
    return {
        "before": before,
        "after": {"free": ends("RTData"), "fixed": ends("Twin")},
        "new_steps": grown.get("new_steps"),
    }


if __name__ == "__main__":
    print("GROWTH=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
