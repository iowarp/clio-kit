"""Fixtures. ``fake_fdmnes`` is a stand-in executable that writes a recorded FDMNES
output (Cu K edge) so the server can be tested without FDMNES; the integration tests
use the real executable from FDMNES_COMMAND and are skipped without it."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

FIXTURES = Path(__file__).parent / "fixtures"
REAL = os.environ.get("FDMNES_COMMAND") or os.environ.get("FDMNES_EXE")


@pytest.fixture
def out(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("CLIO_KIT_ARTIFACTS", str(tmp_path / "ws"))
    return tmp_path / "ws"


def _script(path: Path, body: str) -> Path:
    path.write_text(f"#!{sys.executable}\n{body}")
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return path


@pytest.fixture
def fake_fdmnes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Reads fdmfile.txt like FDMNES, prints an SCF log, writes <Filout>_conv.txt."""
    conv = (FIXTURES / "cu_conv_fdmnes.txt").read_text()
    body = f"""
import pathlib, sys
inp = pathlib.Path(pathlib.Path("fdmfile.txt").read_text().split()[1]).read_text().split()
stem = inp[inp.index("Filout") + 1]
print(" Threshold: Copper K1 edge")
print(" Number of calculated non equivalent absorbing atom =    1")
print(" E_edge     =  8979.00 eV")
print(" Cluster radius = 4.00 A, nb. of atom =  19")
print(" Point group : m3m      (Oh   )")
for c in (1, 2, 3):
    print(f" Cycle   {{c}},  Fermi energy = -10.465 eV")
print(" Number of Energies =  101")
pathlib.Path(stem + "_conv.txt").write_text({conv!r})
"""
    exe = _script(tmp_path / "fake_fdmnes", body)
    monkeypatch.setenv("FDMNES_COMMAND", str(exe))
    return exe


@pytest.fixture
def failing_fdmnes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    exe = _script(
        tmp_path / "bad_fdmnes",
        'import sys\nprint(" Error in the input file")\nsys.exit(3)\n',
    )
    monkeypatch.setenv("FDMNES_COMMAND", str(exe))
    return exe


@pytest.fixture
def copper_cif(tmp_path: Path) -> Path:
    from ase.build import bulk
    from ase.io import write

    path = tmp_path / "cu.cif"
    write(str(path), bulk("Cu", "fcc", a=3.61, cubic=True))
    return path


@pytest.fixture
def feo6_xyz(tmp_path: Path) -> Path:
    from ase import Atoms
    from ase.io import write

    d = 2.0
    pos = [(0, 0, 0), (d, 0, 0), (-d, 0, 0), (0, d, 0), (0, -d, 0), (0, 0, d), (0, 0, -d)]
    path = tmp_path / "feo6.xyz"
    write(str(path), Atoms("FeO6", positions=pos))
    return path
