"""Write the cluster fixture's small datasets.

Run with ParaView's Python (it needs only ``vtkmodules`` and NumPy)::

    python make_datasets.py OUT_DIR            # the standard set
    python make_datasets.py OUT_DIR --large N  # one N^3 wavelet, for the
                                               # remote render threshold

The standard set:

* ``wavelet.vti``: a 41^3 analytic wavelet with the point field ``RTData``
  (range about 34 to 286), for "show an isosurface of RTData".
* ``pulse_series.pvd`` with ``pulse_series/pulse_NNNN.vti``: twelve time
  steps (t = 0.0 to 1.1) of a 32^3 grid with ``pressure`` (a Gaussian pulse
  moving along x, peak 1.0), ``temperature`` and the vector ``velocity``.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkImagingCore import vtkRTAnalyticSource
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

PULSE_STEPS = 12
PULSE_N = 32


def _write(image: vtkImageData, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = vtkXMLImageDataWriter()
    writer.SetFileName(str(path))
    writer.SetInputData(image)
    writer.SetDataModeToBinary()
    writer.SetCompressorTypeToZLib()
    if writer.Write() != 1:
        raise RuntimeError(f"could not write {path}")


def wavelet(half_extent: int) -> vtkImageData:
    source = vtkRTAnalyticSource()
    source.SetWholeExtent(
        -half_extent, half_extent, -half_extent, half_extent, -half_extent, half_extent
    )
    source.Update()
    return source.GetOutput()


def _array(values: np.ndarray, name: str):
    array = numpy_to_vtk(np.ascontiguousarray(values), deep=True)
    array.SetName(name)
    return array


def pulse_step(step: int, n: int = PULSE_N) -> vtkImageData:
    """One time step of the moving pulse on an n^3 grid over [0, 1]^3."""
    t = step / 10.0
    axis = np.linspace(0.0, 1.0, n)
    z, y, x = np.meshgrid(axis, axis, axis, indexing="ij")
    cx = 0.2 + 0.6 * (step / max(PULSE_STEPS - 1, 1))
    r2 = (x - cx) ** 2 + (y - 0.5) ** 2 + (z - 0.5) ** 2
    pressure = np.exp(-r2 / 0.02)
    temperature = 300.0 + 50.0 * pressure + 5.0 * np.sin(2.0 * math.pi * (x + t))
    velocity = np.stack([np.full_like(x, 0.6), -(z - 0.5), (y - 0.5)], axis=-1).reshape(-1, 3)
    image = vtkImageData()
    image.SetDimensions(n, n, n)
    image.SetSpacing(1.0 / (n - 1), 1.0 / (n - 1), 1.0 / (n - 1))
    point_data = image.GetPointData()
    point_data.AddArray(_array(pressure.ravel(), "pressure"))
    point_data.AddArray(_array(temperature.ravel(), "temperature"))
    point_data.AddArray(_array(velocity, "velocity"))
    point_data.SetActiveScalars("pressure")
    return image


def write_standard(out: Path) -> None:
    _write(wavelet(20), out / "wavelet.vti")
    entries = []
    for step in range(PULSE_STEPS):
        name = f"pulse_series/pulse_{step:04d}.vti"
        _write(pulse_step(step), out / name)
        entries.append(f'    <DataSet timestep="{step / 10.0:.1f}" file="{name}"/>')
    (out / "pulse_series.pvd").write_text(
        '<?xml version="1.0"?>\n<VTKFile type="Collection" version="0.1">\n'
        "  <Collection>\n" + "\n".join(entries) + "\n  </Collection>\n</VTKFile>\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("out", type=Path)
    parser.add_argument(
        "--large", type=int, metavar="N", help="write only wavelet_large.vti, an N^3 wavelet"
    )
    args = parser.parse_args()
    if args.large:
        _write(wavelet(args.large // 2), args.out / "wavelet_large.vti")
    else:
        write_standard(args.out)
    print(f"datasets written to {args.out}")


if __name__ == "__main__":
    main()
