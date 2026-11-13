"""Small probes of node08's ParaView, used by the self-checks.

pvpython probe.py render [PNG]    render a sphere offscreen, write a PNG
pvpython probe.py sources PORT    connect to a pvserver, list its sources
"""

from __future__ import annotations

import os
import sys

from paraview import simple


def render(png: str) -> None:
    simple.Show(simple.Sphere())
    view = simple.GetActiveView()
    view.ViewSize = [320, 240]
    simple.Render(view)
    simple.SaveScreenshot(png, view, ImageResolution=[320, 240])
    print(f"rendered {png} {os.path.getsize(png)} bytes")


def sources(port: int) -> None:
    simple.Connect("localhost", port)
    print(sorted(name for name, _ in simple.GetSources()))
    simple.Disconnect()


if __name__ == "__main__":
    if sys.argv[1] == "render":
        render(sys.argv[2] if len(sys.argv) > 2 else "/tmp/clio-pv-render-check.png")
    elif sys.argv[1] == "sources":
        sources(int(sys.argv[2]))
    else:
        raise SystemExit(__doc__)
