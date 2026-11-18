"""Read an STL body and turn it into the voxel mask the patched solver reads.

The mask voxels sit on solver points (or on a grid ``refine`` times finer), so the
solver's nearest-voxel lookup is exact for ``iibm = 1``. A finer mask helps
``iibm = 2``, which also probes points between grid points.
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .errors import CaseError

_RECORD = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])


@dataclass(frozen=True)
class Mesh:
    """Triangles as an (n, 3, 3) float64 array."""

    triangles: np.ndarray

    @property
    def lower(self) -> np.ndarray:
        return self.triangles.reshape(-1, 3).min(axis=0)

    @property
    def upper(self) -> np.ndarray:
        return self.triangles.reshape(-1, 3).max(axis=0)

    @property
    def size(self) -> np.ndarray:
        return self.upper - self.lower

    def volume(self) -> float:
        a, b, c = self.triangles[:, 0], self.triangles[:, 1], self.triangles[:, 2]
        return float(abs(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0)

    def transformed(self, scale: float, rotate_z_deg: float, shift: np.ndarray) -> Mesh:
        """Scale about the origin, turn about +z, then shift."""
        angle = math.radians(rotate_z_deg)
        c, s = math.cos(angle), math.sin(angle)
        turn = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
        points = (self.triangles.reshape(-1, 3) * scale) @ turn.T + shift
        return Mesh(points.reshape(-1, 3, 3))


def read_stl(path: str | Path) -> Mesh:
    """Read a binary or ASCII STL file."""
    file = Path(path)
    if not file.is_file():
        raise CaseError("stl_missing", f"No STL file at {file}.")
    data = file.read_bytes()
    if len(data) >= 84:
        count = struct.unpack_from("<I", data, 80)[0]
        if count > 0 and 84 + 50 * count == len(data):
            rows = np.frombuffer(data, dtype=_RECORD, count=count, offset=84)
            return Mesh(rows["v"].astype(np.float64))
    text = data.decode("ascii", errors="replace")
    values = [
        [float(x) for x in line.split()[1:4]]
        for line in text.splitlines()
        if line.strip().startswith("vertex")
    ]
    if not values or len(values) % 3:
        raise CaseError("stl_unreadable", f"{file} is neither a binary nor an ASCII STL.")
    return Mesh(np.asarray(values, dtype=np.float64).reshape(-1, 3, 3))


def write_stl(mesh: Mesh, path: str | Path) -> None:
    """Write a binary STL (the body as the solver sees it)."""
    tri = mesh.triangles.astype(np.float32)
    normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.divide(normals, lengths, out=np.zeros_like(normals), where=lengths > 0)
    rows = np.zeros(len(tri), dtype=_RECORD)
    rows["normal"] = normals
    rows["v"] = tri
    with open(path, "wb") as out:
        out.write(b"clio xcompact3d body".ljust(80, b" "))
        out.write(struct.pack("<I", len(tri)))
        out.write(rows.tobytes())


@dataclass(frozen=True)
class Mask:
    """Voxel mask ``solid[k, j, i]`` (written x fastest), with origin and spacing."""

    solid: np.ndarray
    origin: tuple[float, float, float]
    spacing: tuple[float, float, float]

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        nz, ny, nx = self.solid.shape
        return nx, ny, nz

    def write(self, run_dir: Path) -> None:
        nx, ny, nz = self.shape_xyz
        o, d = self.origin, self.spacing
        (run_dir / "clio-geometry.txt").write_text(
            f"{nx} {ny} {nz}\n{o[0]!r} {o[1]!r} {o[2]!r}\n{d[0]!r} {d[1]!r} {d[2]!r}\n",
            encoding="ascii",
        )
        (run_dir / "clio-geometry.bin").write_bytes(self.solid.astype(np.int8).tobytes())


def voxelise(
    mesh: Mesh,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    shape: tuple[int, int, int],
) -> np.ndarray:
    """Mark the voxel centres inside a closed mesh, by ray parity along +x.

    ``shape`` is (nx, ny, nz); the result is indexed ``[k, j, i]``. Each (y, z)
    column casts one ray. The rays are nudged off the grid by an irrational
    fraction of a spacing, so none passes exactly through an edge or a vertex.
    """
    nx, ny, nz = shape
    x0, y0, z0 = origin
    dx, dy, dz = spacing
    nudge_y = dy * 1e-6 * math.sqrt(2.0)
    nudge_z = dz * 1e-6 * math.sqrt(3.0)
    hits: dict[tuple[int, int], list[float]] = {}
    for a, b, c in mesh.triangles:
        j0 = max(0, math.ceil((min(a[1], b[1], c[1]) - y0 - nudge_y) / dy))
        j1 = min(ny - 1, math.floor((max(a[1], b[1], c[1]) - y0 - nudge_y) / dy))
        k0 = max(0, math.ceil((min(a[2], b[2], c[2]) - z0 - nudge_z) / dz))
        k1 = min(nz - 1, math.floor((max(a[2], b[2], c[2]) - z0 - nudge_z) / dz))
        if j0 > j1 or k0 > k1:
            continue
        v0y, v0z = b[1] - a[1], b[2] - a[2]
        v1y, v1z = c[1] - a[1], c[2] - a[2]
        det = v0y * v1z - v1y * v0z
        if det == 0.0:
            continue
        grid_j, grid_k = np.meshgrid(np.arange(j0, j1 + 1), np.arange(k0, k1 + 1), indexing="ij")
        jj, kk = grid_j.ravel(), grid_k.ravel()
        wy = y0 + jj * dy + nudge_y - a[1]
        wz = z0 + kk * dz + nudge_z - a[2]
        u = (wy * v1z - v1y * wz) / det
        v = (v0y * wz - wy * v0z) / det
        inside = (u >= 0.0) & (v >= 0.0) & (u + v <= 1.0)
        if not inside.any():
            continue
        xs = a[0] + u * (b[0] - a[0]) + v * (c[0] - a[0])
        for j, k, x in zip(jj[inside], kk[inside], xs[inside], strict=True):
            hits.setdefault((int(j), int(k)), []).append(float(x))
    solid = np.zeros((nz, ny, nx), dtype=bool)
    xi = x0 + np.arange(nx) * dx
    for (j, k), xs in hits.items():
        xs.sort()
        for enter, leave in zip(xs[0::2], xs[1::2], strict=False):
            solid[k, j] |= (xi >= enter) & (xi <= leave)
    return solid


def mask_for(
    mesh: Mesh,
    grid_spacing: tuple[float, float, float],
    refine: int = 1,
    span: tuple[float, float] | None = None,
) -> Mask:
    """Voxelise the body around its bounding box on the solver grid.

    The mask origin is a multiple of the spacing, so with ``refine = 1`` the voxel
    centres are solver points. With ``span`` (z0, z1) the body is a section
    extruded through a periodic span: its middle z-plane is voxelised once and
    repeated over every z-plane from z0 to z1.
    """
    if refine < 1:
        raise CaseError("bad_refine", "refine must be 1 or more.")
    sx, sy, sz = (d / refine for d in grid_spacing)
    spacing = (sx, sy, sz)
    lower, upper = mesh.lower, mesh.upper

    def axis(i: int, step: float) -> tuple[float, int]:
        start = (math.floor(lower[i] / step) - 1) * step
        stop = (math.ceil(upper[i] / step) + 1) * step
        return start, round((stop - start) / step) + 1

    ox, nx = axis(0, sx)
    oy, ny = axis(1, sy)
    if span is None:
        oz, nz = axis(2, sz)
        return Mask(voxelise(mesh, (ox, oy, oz), spacing, (nx, ny, nz)), (ox, oy, oz), spacing)
    middle = 0.5 * (lower[2] + upper[2])
    plane = voxelise(mesh, (ox, oy, middle), spacing, (nx, ny, 1))
    z0, z1 = span
    nz = round((z1 - z0) / sz) + 1
    return Mask(np.repeat(plane, nz, axis=0), (ox, oy, z0), spacing)
