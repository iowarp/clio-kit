"""The person's own ParaView session on the node08 variant's pvserver.

It connects to the pvserver, opens a dataset with an isosurface, and stays
connected, so the server keeps that pipeline for any later client. When it
is ready it writes the source names to ``--ready``.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from paraview import simple


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--port", type=int, default=11111)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--ready", type=Path, required=True)
    args = parser.parse_args()
    simple.Connect("localhost", args.port)
    reader = simple.OpenDataFile(args.dataset, registrationName=Path(args.dataset).name)
    reader.UpdatePipelineInformation()
    times = list(getattr(reader, "TimestepValues", []) or [])
    contour = simple.Contour(
        Input=reader,
        ContourBy=["POINTS", "pressure"],
        Isosurfaces=[0.5],
        registrationName="pressure isosurface",
    )
    view = simple.GetActiveViewOrCreate("RenderView")
    if times:
        view.ViewTime = times[len(times) // 2]
    simple.Show(reader, view).SetRepresentationType("Outline")
    simple.Show(contour, view)
    simple.ResetCamera(view)
    simple.Render(view)
    names = sorted(name for name, _ in simple.GetSources())
    args.ready.write_text("\n".join(names) + "\n")
    print(f"holding {names} at time {view.ViewTime}", flush=True)
    while True:
        time.sleep(30)


if __name__ == "__main__":
    main()
