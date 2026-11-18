from __future__ import annotations

from pathlib import Path

import pytest

from fdmnes_mcp.errors import FdmnesError
from fdmnes_mcp.inputs import XanesSpec, read_structure, write_inputs


def _text(path: str) -> list[str]:
    return Path(path).read_text().splitlines()


def test_crystal_input_has_cell_fractions_and_element_absorber(copper_cif, tmp_path):
    got = write_inputs(read_structure(copper_cif), XanesSpec(absorber="Cu"), tmp_path, "x")
    lines = _text(got["input_file"])
    assert got["kind"] == "crystal" and got["absorber"]["Z"] == 29
    assert lines[lines.index("Z_absorber") + 1] == "29"
    cell = lines[lines.index("Crystal") + 1].split()
    assert [round(float(v), 4) for v in cell] == [3.61, 3.61, 3.61, 90, 90, 90]
    frac = lines[lines.index("Crystal") + 3].split()
    assert len(frac[1].split(".")[1]) == 10  # FDMNES needs the digits for symmetry
    assert "Green" in lines and "SCF" in lines and "Edge" not in lines
    assert (tmp_path / "fdmfile.txt").read_text() == "1\nx_inp.txt\n"


def test_molecule_is_cartesian_angstrom(feo6_xyz, tmp_path):
    spec = XanesSpec(absorber=[0], edge="L23", method="fdm", scf=False)
    got = write_inputs(read_structure(feo6_xyz), spec, tmp_path, "x")
    lines = _text(got["input_file"])
    i = lines.index("Molecule")
    assert lines[i + 1] == "1.0 1.0 1.0 90.0 90.0 90.0"
    assert lines[i + 3].split()[:2] == ["8", "2.00000000"]
    assert lines[lines.index("Absorber") + 1] == "1"  # FDMNES counts from 1
    assert lines[lines.index("Edge") + 1] == "L23"
    assert "Green" not in lines and "SCF" not in lines


@pytest.mark.parametrize(
    ("spec", "words"),
    [
        (XanesSpec(absorber="Zn"), "no Zn atom"),
        (XanesSpec(absorber="Xx"), "not an element"),
        (XanesSpec(absorber=[9]), "outside"),
        (XanesSpec(absorber="Fe", edge="Q"), "edge"),
        (XanesSpec(absorber="Fe", energy_range=(10.0, 1.0, 5.0)), "increasing"),
        (XanesSpec(absorber="Fe", extra_keywords=["End"]), "not allowed"),
    ],
)
def test_bad_input_is_refused(feo6_xyz, tmp_path, spec, words):
    with pytest.raises(FdmnesError) as err:
        write_inputs(read_structure(feo6_xyz), spec, tmp_path, "x")
    assert err.value.reason == "invalid_input" and words in str(err.value)
