"""Fixtures. The stand-in executables replay gRASPA's two output formats so the server
can be tested without a GPU: ``graspa_cuda_co2_mfi_stdout.txt`` is an excerpt of the
CO2-MFI example output shipped with snurr-group/gRASPA (MIT), and the SYCL lines are
the format ChemGraph's hermetic tests use. A real run needs GRASPA_COMMAND."""

from __future__ import annotations

import stat
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(tmp_path / "ws"))
    return tmp_path / "ws"


def _exe(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def cuda_graspa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Checks the run folder like gRASPA would, then prints the recorded CO2 output."""
    text = (FIXTURES / "graspa_cuda_co2_mfi_stdout.txt").read_text()
    body = f"""
import pathlib, sys
inp = pathlib.Path("simulation.input").read_text()
for key in ("NumberOfTrialPositions", "UseChargesFromCIFFile", "UnitCells 0", "CO2"):
    assert key in inp, key
assert pathlib.Path("CO2.def").is_file() and list(pathlib.Path().glob("*.cif"))
sys.stdout.write({text!r})
"""
    exe = _exe(tmp_path / "nvc_main.x", body)
    monkeypatch.setenv("GRASPA_COMMAND", str(exe))
    return exe


@pytest.fixture
def sycl_graspa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    body = 'print("Input UnitCells [0] = 3 3 3")\nprint("Overall: Average: 54.0,")\n'
    exe = _exe(tmp_path / "sycl.out", body)
    monkeypatch.setenv("GRASPA_COMMAND", str(exe))
    return exe


@pytest.fixture
def crashing_graspa(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    body = 'import sys\nprint("PRODUCTION Cycle: 0, 3 Adsorbate Molecules")\n'
    body += 'sys.stderr.write("device failure")\nsys.exit(137)\n'
    exe = _exe(tmp_path / "bad.x", body)
    monkeypatch.setenv("GRASPA_COMMAND", str(exe))
    return exe


def _cif(path: Path, a: float, charges: bool = False) -> Path:
    """A small cubic SiO2-like P1 framework; with charges, an _atom_site_charge column."""
    sites = [
        ("Si", 0, 0, 0, 1.2),
        ("Si", 0.5, 0.5, 0.5, 1.2),
        ("O", 0.25, 0, 0, -0.6),
        ("O", 0.75, 0, 0, -0.6),
        ("O", 0, 0.25, 0.5, -0.6),
        ("O", 0.5, 0.75, 0, -0.6),
    ]
    head = [
        f"data_{path.stem}",
        f"_cell_length_a {a}",
        f"_cell_length_b {a}",
        f"_cell_length_c {a}",
        "_cell_angle_alpha 90",
        "_cell_angle_beta 90",
        "_cell_angle_gamma 90",
        "_symmetry_space_group_name_H-M 'P 1'",
        "loop_",
        "_atom_site_label",
        "_atom_site_type_symbol",
        "_atom_site_fract_x",
        "_atom_site_fract_y",
        "_atom_site_fract_z",
    ]
    if charges:
        head.append("_atom_site_charge")
    rows = [
        f"{el}{i} {el} {x} {y} {z}" + (f" {q}" if charges else "")
        for i, (el, x, y, z, q) in enumerate(sites)
    ]
    path.write_text("\n".join(head + rows) + "\n")
    return path


@pytest.fixture
def frameworks(tmp_path: Path) -> dict[str, Path]:
    d = tmp_path / "cifs"
    d.mkdir()
    return {"small": _cif(d / "small.cif", 9.0), "large": _cif(d / "large.cif", 27.0)}


@pytest.fixture
def charged_cif(tmp_path: Path) -> Path:
    return _cif(tmp_path / "charged.cif", 9.0, charges=True)
