"""A volume is drawn on the ParaView machine; run with pvpython.

    pvpython view_mode_check.py OUT_DIR

Opens a small wavelet, shows it as a Volume, then as a Surface, and prints
``VIEW=<json>`` with the view's mode and reason after each.
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
    data = out / "wave.vti"
    simple.SaveData(str(data), proxy=simple.Wavelet(WholeExtent=[0, 8, 0, 8, 0, 8]))
    engine = Engine(out / "folder")
    engine.run("open_view", {"path": str(data), "new": True})
    seen: dict[str, Any] = {"opened": dict(engine.scene["view"])}
    for shown in ("Volume", "Surface"):
        engine.run(
            "set_representation",
            {"object": "o1", "representation": shown, "color_by": "RTData"},
        )
        seen[shown] = dict(engine.scene["view"])
    return seen


if __name__ == "__main__":
    print("VIEW=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
