"""Whether a rendered picture shows anything.

A render that returns is not proof of a picture: a pvserver that went away, a
camera that looks at nothing or an empty pipeline all give a frame of plain
background. The test renders the same frame again with the pipeline hidden
(scalar bars and axes stay) and counts the pixels that differ, so it holds for
any background, gradient or theme. It needs only VTK, not numpy.
"""

from __future__ import annotations

import os
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

#: A pixel differs when its summed RGB difference is over this.
PIXEL_DIFFERENCE = 30
#: A frame with fewer differing pixels than this share is blank.
BLANK_BELOW = 0.005
#: About how many pixels are compared.
SAMPLES = 40_000


def compare(a: bytes, b: bytes, comps: int, stride: int = 1) -> float:
    """The share of sampled pixels that differ between two same-size pictures."""
    if len(a) != len(b) or comps < 1 or not a:
        return 1.0 if len(a) != len(b) else 0.0
    step = comps * max(1, stride)
    rgb = min(3, comps)
    total = differ = 0
    for start in range(0, len(a) - comps + 1, step):
        total += 1
        if sum(abs(a[start + c] - b[start + c]) for c in range(rgb)) > PIXEL_DIFFERENCE:
            differ += 1
    return differ / total if total else 0.0


def read_png(path: Path) -> tuple[bytes, int, int, int]:
    """The raw pixels of a PNG, with their component count, width and height."""
    from vtkmodules.vtkIOImage import vtkPNGReader

    reader = vtkPNGReader()
    reader.SetFileName(str(path))
    reader.Update()
    image = reader.GetOutput()
    scalars = image.GetPointData().GetScalars()
    width, height, _ = image.GetDimensions()
    if scalars is None:
        return b"", 0, width, height
    return bytes(memoryview(scalars)), scalars.GetNumberOfComponents(), width, height


def measure(
    render: Callable[[Path], None],
    representations: list[Any],
    folder: Path,
    picture: Path | None = None,
) -> dict[str, Any]:
    """How much of a frame the pipeline draws.

    ``render(path)`` writes the frame at a fixed size; ``picture``, when given, is
    that frame already rendered. The pipeline's ``representations`` are hidden
    for the second render and shown again after it.
    """
    started = time.perf_counter()
    shown = [r for r in representations if r.Visibility]
    if not shown:
        return {
            "fraction": 0.0,
            "blank": True,
            "seconds": round(time.perf_counter() - started, 2),
            "note": "nothing in the scene is visible",
        }
    folder.mkdir(parents=True, exist_ok=True)
    made: list[Path] = []

    def scratch() -> Path:
        handle, name = tempfile.mkstemp(suffix=".png", dir=folder)
        os.close(handle)
        made.append(Path(name))
        return made[-1]

    try:
        if picture is None:
            picture = scratch()
            render(picture)
        empty = scratch()
        for rep in shown:
            rep.Visibility = 0
        try:
            render(empty)
        finally:
            for rep in shown:
                rep.Visibility = 1
        a, comps, width, height = read_png(picture)
        b, comps_b, width_b, height_b = read_png(empty)
    finally:
        for path in made:
            try:
                path.unlink()
            except OSError:
                pass
    if (comps, width, height) != (comps_b, width_b, height_b) or not a or not b:
        return {
            "fraction": None,
            "blank": None,
            "seconds": round(time.perf_counter() - started, 2),
            "note": "the test frames could not be read back",
        }
    stride = max(1, (width * height) // SAMPLES)
    fraction = compare(a, b, comps, stride)
    return {
        "fraction": round(fraction, 4),
        "blank": fraction < BLANK_BELOW,
        "seconds": round(time.perf_counter() - started, 2),
    }
