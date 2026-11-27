"""A growing source opened late catches up from its first step, then follows.

    pvpython replay_check.py OUT_DIR

Writes a three-step wavelet series, opens it growing, starts the catch-up a view's
first connection starts, steps it to the end and prints ``REPLAY=<json>`` with the
shown index and the catch-up flag at each point, and whether a chosen step ends it.
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
    simple.SaveData(str(path), proxy=calc)


def shown(engine: Engine) -> list[Any]:
    summary = engine.time_summary()
    return [summary["index"], summary["catching_up"], summary["follow_latest"]]


def main(out: Path) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    for i in range(3):
        write(out / f"r_{i}.vti", 1.0 + i)
    engine = Engine(out / "replay-folder")
    engine.run(
        "open_view", {"path": str(out / "r_*.vti"), "new": True, "growing": True}
    )
    report: dict[str, Any] = {"opened": shown(engine), "armed": engine.replay_armed}
    report["started"] = engine.start_replay()
    report["first"] = shown(engine)
    report["going"] = engine.replay_tick()
    report["second"] = shown(engine)
    report["ended"] = not engine.replay_tick()
    report["last"] = shown(engine)
    report["again"] = engine.start_replay()
    engine.arm_replay()
    engine.start_replay()
    engine.run("set_timestep", {"index": 1})
    report["chosen"] = shown(engine)
    return report


if __name__ == "__main__":
    print("REPLAY=" + json.dumps(main(Path(sys.argv[1]))), flush=True)
