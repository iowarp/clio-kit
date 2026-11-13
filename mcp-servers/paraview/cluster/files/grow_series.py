"""Write a time series one step at a time, as a running simulation would.

Run with ParaView's Python on node08 (``module load paraview``)::

    python grow_series.py OUT_DIR [--interval 5] [--steps 60] [--n 32]

Each step is ``OUT_DIR/growing_NNNN.vti``, written under a hidden name and
renamed into place, so a reader never sees half a file. After each step the
index ``OUT_DIR/growing.vti.series`` (ParaView's file-series JSON) is
rewritten the same way. Existing steps are kept, so a restart continues the
series. Stop it with SIGTERM (``cluster.sh grow stop``).
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from make_datasets import _write, pulse_step


def _series_index(out: Path) -> list[dict[str, object]]:
    steps = sorted(out.glob("growing_[0-9][0-9][0-9][0-9].vti"))
    return [{"name": p.name, "time": int(p.stem.split("_")[1]) / 10.0} for p in steps]


def _write_index(out: Path) -> int:
    files = _series_index(out)
    tmp = out / ".growing.vti.series.part"
    tmp.write_text(json.dumps({"file-series-version": "1.0", "files": files}, indent=1))
    os.replace(tmp, out / "growing.vti.series")
    return len(files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument("--interval", type=float, default=5.0, help="seconds per step")
    parser.add_argument("--steps", type=int, default=60, help="last step to write")
    parser.add_argument("--n", type=int, default=32, help="grid points per axis")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    step = len(_series_index(args.out))
    while step < args.steps:
        final = args.out / f"growing_{step:04d}.vti"
        tmp = args.out / f".growing_{step:04d}.vti.part"
        _write(pulse_step(step % 12, args.n), tmp)
        os.replace(tmp, final)
        count = _write_index(args.out)
        print(f"step {step} written ({count} in the series)", flush=True)
        step += 1
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
