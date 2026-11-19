"""Case sizing, the immersed-body mask and reading a run back, without the solver."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest

from xcompact3d_mcp.case import (
    DIFFUSION_LIMIT,
    CaseRequest,
    decomposition,
    load_case,
    nice,
    nice_dt,
    prepare_case,
)
from xcompact3d_mcp.errors import CaseError
from xcompact3d_mcp.geometry import Mesh, read_stl, voxelise, write_stl
from xcompact3d_mcp.results import case_status, read_forces

FACES = [
    (0, 2, 1), (0, 3, 2), (4, 5, 6), (4, 6, 7), (0, 1, 5), (0, 5, 4),
    (2, 3, 7), (2, 7, 6), (1, 2, 6), (1, 6, 5), (0, 4, 7), (0, 7, 3),
]  # fmt: skip


def box(lower: tuple[float, float, float], upper: tuple[float, float, float]) -> Mesh:
    (x0, y0, z0), (x1, y1, z1) = lower, upper
    corners = np.array(
        [
            [x0, y0, z0], [x1, y0, z0], [x1, y1, z0], [x0, y1, z0],
            [x0, y0, z1], [x1, y0, z1], [x1, y1, z1], [x0, y1, z1],
        ]
    )  # fmt: skip
    return Mesh(np.array([[corners[i] for i in face] for face in FACES], dtype=float))


@pytest.fixture
def plate(tmp_path: Path) -> Path:
    """A 4 x 1 x 1 block, an extruded section 4 units long."""
    path = tmp_path / "plate.stl"
    write_stl(box((0.0, 0.0, 0.0), (4.0, 1.0, 1.0)), path)
    return path


def test_fft_friendly_sizes() -> None:
    assert nice(97, periodic=False) == 97  # 96 = 2^5 3
    assert nice(98, periodic=False) == 101  # 100 = 2^2 5^2
    assert nice(98, periodic=True) == 100
    assert nice(3, periodic=True) == 8


def test_round_time_steps() -> None:
    assert nice_dt(0.0042) == pytest.approx(0.004)
    assert nice_dt(0.0049) == pytest.approx(0.004)
    assert nice_dt(0.05) == pytest.approx(0.05)
    assert nice_dt(0.0012) == pytest.approx(0.001)


def test_decomposition_is_close_to_square_and_fits() -> None:
    assert decomposition(20, 361, 160, 12) == (5, 4)
    assert decomposition(1, 8, 8, 8) == (1, 1)
    with pytest.raises(CaseError) as error:
        decomposition(64, 8, 8, 2)
    assert error.value.reason == "too_many_ranks"


def test_voxelised_box_matches_its_volume(tmp_path: Path) -> None:
    mesh = box((0.1, 0.1, 0.1), (0.9, 0.5, 0.7))
    stl = tmp_path / "box.stl"
    write_stl(mesh, stl)
    again = read_stl(stl)
    assert again.volume() == pytest.approx(0.8 * 0.4 * 0.6, rel=1e-5)
    d = 0.02
    # Centres half a spacing off the faces, so each axis holds a whole count of them.
    solid = voxelise(again, (d / 2, d / 2, d / 2), (d, d, d), (50, 50, 50))
    assert solid.shape == (50, 50, 50)  # [k, j, i]
    assert solid.sum() * d**3 == pytest.approx(again.volume(), rel=0.01)
    assert solid[25, 15, 25] and not solid[25, 40, 25]


def test_prepare_case_respects_stability_and_writes_a_runnable_case(
    plate: Path, tmp_path: Path
) -> None:
    run = tmp_path / "run"
    record = prepare_case(
        CaseRequest(
            stl=str(plate), run_dir=str(run), reynolds=1000, angle_deg=10,
            points_per_length=12, end_time=6, snapshots=6, nprocs=4,
        )
    )  # fmt: skip
    plan = record["plan"]
    assert plan["span"] == "periodic"
    assert plan["scale"] == pytest.approx(0.25)
    assert plan["resolution"]["cfl"] <= 0.1 + 1e-12
    assert plan["resolution"]["diffusion_number"] <= DIFFUSION_LIMIT + 1e-12
    assert plan["ilast"] == plan["ioutput"] * plan["snapshots"]
    for name in ("input.i3d", "run.sh", "clio-vtk.xml", "clio-geometry.txt", "body.stl"):
        assert (run / name).is_file(), name
    g = plan["grid"]
    assert (run / "clio-geometry.bin").stat().st_size == math.prod(plan["body"]["mask_shape"])
    vtk = (run / "clio-vtk.xml").read_text()
    # ADIOS2 row-major order: the slowest (z) axis first.
    assert f'WholeExtent="0 {g["nz"] - 1} 0 {g["ny"] - 1} 0 {g["nx"] - 1}"' in vtk
    text = (run / "input.i3d").read_text()
    assert f"nx = {g['nx']}" in text and f"ilast = {plan['ilast']}" in text
    view = record["view"]
    assert view["path"].endswith("data.bp5") and view["growing"]
    assert view["follow"]["expected_steps"] == 6
    job = record["relay_job"]
    assert job["arguments"] == [str(run.resolve() / "run.sh")]
    assert job["resources"] == {"node_count": 1, "process_count": 4}
    assert job["launcher"] == "single"
    assert load_case(str(run))["plan"] == plan


def test_prepare_case_refuses_a_run_with_results(plate: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    request = CaseRequest(stl=str(plate), run_dir=str(run), points_per_length=8, nprocs=1)
    prepare_case(request)
    (run / "data.bp5").mkdir()
    with pytest.raises(CaseError) as error:
        prepare_case(request)
    assert error.value.reason == "run_dir_has_results"
    prepare_case(request, overwrite=True)


def test_status_and_forces_from_a_finished_run(plate: Path, tmp_path: Path) -> None:
    run = tmp_path / "run"
    plan = prepare_case(
        CaseRequest(stl=str(plate), run_dir=str(run), points_per_length=8, nprocs=1)
    )["plan"]
    last = plan["ilast"]
    (run / "xcompact3d.log").write_text(
        f" Time step = {last}/ {last}, Time unit = {plan['end_time']:.4f}\n"
        " Time for this time step (s):  0.05\n"
    )
    status = case_status(str(run))
    assert status["finished"] and not status["failed"]
    assert status["snapshots_written"] == plan["snapshots"]
    t = np.linspace(0.0, 40.0, 4001)
    rows = np.column_stack([t, 1.2 + 0 * t, 0.3 * np.sin(2 * math.pi * 0.25 * t)])
    np.savetxt(run / "forces.dat", rows)
    # The kept window, t 8 to 40, holds a whole number of periods.
    forces = read_forces(str(run), discard_fraction=0.2)
    assert forces["cd_mean"] == pytest.approx(1.2)
    assert forces["cl_mean"] == pytest.approx(0.0, abs=0.01)
    assert forces["shedding_frequency"] == pytest.approx(0.25, rel=0.03)
