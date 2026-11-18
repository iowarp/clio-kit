"""Calculator, constraint and run input validation."""

from __future__ import annotations

import pytest

from ase_mcp.errors import AseError
from ase_mcp.schemas import CalculatorSpec, RunInput, parse_calculator, parse_constraints


def test_defaults_are_resolved_and_recorded() -> None:
    assert parse_calculator({"type": "xtb"}).resolved() == {
        "type": "xtb",
        "method": "GFN2-xTB",
        "accuracy": 1.0,
        "electronic_temperature": 300.0,
        "max_iterations": 250,
        "solvation": None,
    }
    mace = parse_calculator("mace").resolved()
    assert mace["model"] == "mace-off-small" and mace["dtype"] == "float64"
    nw = parse_calculator({"type": "nwchem"}).resolved()
    assert (nw["method"], nw["basis"], nw["theory"]) == ("B3LYP", "6-31G*", "dft")
    assert parse_calculator({"type": "uma"}).resolved()["task"] == "omol"


def test_settings_at_top_level_or_nested() -> None:
    a = parse_calculator({"type": "xtb", "accuracy": 0.1}).resolved()
    b = parse_calculator({"type": "xtb", "settings": {"accuracy": 0.1}}).resolved()
    assert a["accuracy"] == b["accuracy"] == 0.1


@pytest.mark.parametrize(
    "spec, words",
    [
        ({"type": "xtb", "basis_set": "x"}, "does not take"),
        ({"type": "xtb", "method": "GFN0"}, "xTB method"),
        ({"type": "emt", "model": "x"}, "emt takes no model"),
        ({"type": "xtb", "basis": "x"}, "xtb takes no basis"),
        ({"type": "dft"}, "Input should be"),
        ({"type": "mace", "temperature": 3}, "does not take"),
    ],
)
def test_bad_calculators_are_refused_not_ignored(spec: dict, words: str) -> None:
    with pytest.raises(AseError) as info:
        parse_calculator(spec)
    assert info.value.reason == "invalid_input"
    assert words in str(info.value)


def test_constraints() -> None:
    cs = parse_constraints(
        [
            {"type": "fix_atoms", "indices": [0, 1]},
            {"type": "dihedral", "indices": [2, 0, 1, 5], "value_deg": 0},
            {"type": "bond", "indices": [0, 1], "value_A": 1.5},
        ]
    )
    assert [c.type for c in cs] == ["fix_atoms", "dihedral", "bond"]
    for bad in (
        {"type": "dihedral", "indices": [0, 1, 2]},
        {"type": "bond", "indices": [0, 1], "value_deg": 3},
        {"type": "fix_atoms", "indices": [0], "value_A": 1},
        {"type": "torsion", "indices": [0, 1, 2, 3]},
    ):
        with pytest.raises(AseError) as info:
            parse_constraints([bad])
        assert info.value.reason == "invalid_input"


def test_run_input_bounds() -> None:
    spec = CalculatorSpec(type="emt")
    RunInput(structure="x.xyz", calculator=spec, multiplicity=3)
    with pytest.raises(ValueError):
        RunInput(structure="x.xyz", calculator=spec, multiplicity=0)
    with pytest.raises(ValueError):
        RunInput(structure="x.xyz", calculator=spec, fmax=0)
