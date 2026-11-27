"""Bonds, measurements and setting internal coordinates."""

from __future__ import annotations

import pytest
from ase.build import molecule

from ase_mcp import geometry
from ase_mcp.errors import AseError


def test_describe_ethanol() -> None:
    atoms = molecule("CH3CH2OH")
    d = geometry.describe(atoms, [[0, 1], [0, 1, 2], [8, 0, 1, 2]])
    assert d["formula"] == "C2H6O"
    assert d["n_bonds"] == 8 and d["n_fragments"] == 1
    kinds = [m["kind"] for m in d["measurements"]]
    assert kinds == ["distance", "angle", "dihedral"]
    assert 1.4 < d["measurements"][0]["value"] < 1.6


@pytest.mark.parametrize("target", [0.0, 60.0, -120.0, 179.0])
@pytest.mark.parametrize("indices", [[8, 0, 1, 2], [2, 1, 0, 8], [3, 2, 1, 0]])
def test_set_dihedral_moves_the_smaller_side(indices: list[int], target: float) -> None:
    atoms = molecule("CH3CH2OH")
    before = atoms.get_distance(0, 1)
    change = geometry.set_dihedral(atoms, indices, target)
    assert abs(geometry.signed_dihedral(geometry.measure(atoms, indices)["value"] - target)) < 1e-3
    assert abs(atoms.get_distance(0, 1) - before) < 1e-9  # rigid rotation
    assert len(change["moved_atoms"]) <= 5


def test_set_distance_and_angle() -> None:
    atoms = molecule("C2H6")
    geometry.set_distance(atoms, [0, 1], 1.7)
    assert geometry.measure(atoms, [0, 1])["value"] == pytest.approx(1.7, abs=1e-6)
    geometry.set_angle(atoms, [2, 0, 1], 100.0)
    assert geometry.measure(atoms, [2, 0, 1])["value"] == pytest.approx(100.0, abs=1e-4)


def test_two_fragments_distance() -> None:
    a = molecule("H2O")
    b = molecule("NH3")
    b.translate([3.0, 0, 0])
    both = a + b
    assert len(geometry.fragments(both)) == 2
    geometry.set_distance(both, [0, 3], 4.0)
    assert geometry.measure(both, [0, 3])["value"] == pytest.approx(4.0, abs=1e-6)


def test_ring_bond_and_unbonded_pair_are_refused() -> None:
    benzene = molecule("C6H6")
    with pytest.raises(AseError) as info:
        geometry.set_dihedral(benzene, [6, 0, 1, 7], 10)
    assert info.value.reason == "ring_bond"
    ethanol = molecule("CH3CH2OH")
    with pytest.raises(AseError) as info:
        geometry.set_dihedral(ethanol, [8, 2, 0, 1], 10)
    assert info.value.reason == "invalid_input"


def test_bad_indices() -> None:
    atoms = molecule("H2O")
    with pytest.raises(AseError, match="out of range"):
        geometry.measure(atoms, [0, 7])
    with pytest.raises(AseError, match="2 \\(distance\\)"):
        geometry.measure(atoms, [0])
