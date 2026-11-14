"""Regenerate the test fixtures: a small tube (STL) and its G-code from PrusaSlicer.

The tube is 16 mm across, 3 mm tall, with a 6 mm hole and a 2 mm bridge-free floor:
it gives walls, solid and sparse infill, top and bottom surfaces and arcs (the MK4
presets fit arcs), in about 100 KB of G-code. Run from mcp-servers/slicer with the
slicer installed (test-only; the server itself never downloads one)::

    uv run python tests/fixtures/make_fixture.py

"""

from __future__ import annotations

import math
import struct
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def tube(outer: float = 8.0, inner: float = 3.0, height: float = 3.0, n: int = 48) -> bytes:
    triangles = []
    ring = [(math.cos(2 * math.pi * i / n), math.sin(2 * math.pi * i / n)) for i in range(n)]
    for i in range(n):
        (c0, s0), (c1, s1) = ring[i], ring[(i + 1) % n]
        o0, o1 = (outer * c0, outer * s0), (outer * c1, outer * s1)
        i0, i1 = (inner * c0, inner * s0), (inner * c1, inner * s1)
        for z, up in ((0.0, False), (height, True)):
            a, b, c, d = (*o0, z), (*o1, z), (*i1, z), (*i0, z)
            quad = [(a, b, c), (a, c, d)] if up else [(a, c, b), (a, d, c)]
            triangles += quad
        lo, hi = 0.0, height
        triangles += [((*o0, lo), (*o1, lo), (*o1, hi)), ((*o0, lo), (*o1, hi), (*o0, hi))]
        triangles += [((*i0, lo), (*i1, hi), (*i1, lo)), ((*i0, lo), (*i0, hi), (*i1, hi))]
    body = b"".join(struct.pack("<12fH", 0, 0, 0, *a, *b, *c, 0) for a, b, c in triangles)
    return b"fixture tube".ljust(80, b" ") + struct.pack("<I", len(triangles)) + body


def main() -> int:
    (HERE / "tube.stl").write_bytes(tube())
    sys.path.insert(0, str(HERE.parents[1] / "src"))
    import os
    import shutil
    import tempfile

    os.environ["SLICER_MCP_HOME"] = tempfile.mkdtemp(prefix="slicer-fixture-")
    from slicer_mcp.slicing import slice_model

    record = slice_model(str(HERE / "tube.stl"), "Prusa MK4", "PLA", "0.2 mm")
    shutil.copyfile(record["gcode"], HERE / "tube_mk4_pla_0.20.gcode")
    shutil.rmtree(os.environ["SLICER_MCP_HOME"])
    print(record["estimates"], len(record["layers"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
