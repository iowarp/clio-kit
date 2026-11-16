"""Calculator gating (typed errors that say how to fix), readiness and NWChem input."""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.build import molecule

from ase_mcp import calculators
from ase_mcp.errors import AseError
from ase_mcp.schemas import parse_calculator


@pytest.fixture
def bare_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (*calculators.NWCHEM_ENV, *calculators.ORCA_ENV, *calculators.HF_ENV):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf"))
    monkeypatch.setenv("PATH", str(tmp_path / "empty-bin"))


def resolved(spec: dict) -> dict:
    return parse_calculator(spec).resolved()


def test_nwchem_missing(bare_env: None) -> None:
    with pytest.raises(AseError) as info:
        calculators.gate(resolved({"type": "nwchem"}))
    assert info.value.reason == "binary_missing"
    assert "ASE_NWCHEM_COMMAND" in str(info.value) and "Spack" in str(info.value)


def test_orca_missing(bare_env: None) -> None:
    with pytest.raises(AseError) as info:
        calculators.gate(resolved({"type": "orca"}))
    assert info.value.reason == "binary_missing"
    assert "ORCA_COMMAND" in str(info.value) and "registration" in str(info.value)


def test_uma_without_token(bare_env: None) -> None:
    with pytest.raises(AseError) as info:
        calculators.gate(resolved({"type": "uma"}))
    assert info.value.reason == "hf_token_missing"
    assert "HF_TOKEN" in str(info.value)


def test_found_by_environment(bare_env: None, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NWCHEM_COMMAND", "/opt/nwchem/bin/nwchem")
    monkeypatch.setenv("ORCA_COMMAND", "/opt/orca/orca")
    monkeypatch.setenv("HF_TOKEN", "hf_dummy")
    for kind in ("nwchem", "orca", "uma", "emt", "xtb", "mace"):
        calculators.gate(resolved({"type": kind}))
    assert calculators.nwchem_command()["source"] == "NWCHEM_COMMAND"
    assert calculators.hf_token()["source"] == "HF_TOKEN"


def test_ir_needs_a_dipole() -> None:
    for kind in ("emt", "mace"):
        with pytest.raises(AseError) as info:
            calculators.gate(resolved({"type": kind}), "ir")
        assert info.value.reason == "dipole_unavailable"


def test_spin_and_charge_modelling() -> None:
    assert calculators.models_spin(resolved({"type": "xtb"}))
    assert not calculators.models_spin(resolved({"type": "mace"}))
    assert calculators.models_spin(resolved({"type": "mace", "model": "mace-omol-extra-large"}))
    assert not calculators.models_spin(resolved({"type": "emt"}))


def test_emt_build_notes_unmodelled_spin(tmp_path: Path) -> None:
    _, info, notes, _ = calculators.build(resolved({"type": "emt"}), -1, 3, tmp_path)
    assert info == {}
    assert any("does not model electronic spin" in n for n in notes)
    assert any("does not model the total charge" in n for n in notes)


def _nwchem_input(tmp_path: Path, spec: dict, charge: int, mult: int) -> str:
    from ase.io.nwchem.nwwriter import write_nwchem_in

    kwargs = calculators.nwchem_kwargs(resolved(spec), charge, mult, tmp_path, "nwchem")
    kwargs.pop("command")
    path = tmp_path / "in.nwi"
    with path.open("w") as fh:
        write_nwchem_in(fh, molecule("OH"), properties=["energy", "forces"], **kwargs)
    return path.read_text()


def test_nwchem_input_dft_triplet_anion(tmp_path: Path) -> None:
    text = _nwchem_input(tmp_path, {"type": "nwchem", "method": "PBE0", "basis": "def2-SVP"}, -1, 3)
    assert "charge -1" in text
    assert "mult 3" in text
    assert "xc PBE0" in text
    assert "def2-SVP" in text


def test_nwchem_input_scf_open_shell(tmp_path: Path) -> None:
    text = _nwchem_input(tmp_path, {"type": "nwchem", "theory": "scf"}, 0, 2)
    assert "nopen 1" in text and "uhf" in text


def test_nwchem_command_completion(
    bare_env: None, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("NWCHEM_COMMAND", "/opt/nwchem")
    calc, *_ = calculators.build(resolved({"type": "nwchem", "nprocs": 4}), 0, 1, tmp_path)
    assert calc.command == "mpirun -np 4 /opt/nwchem PREFIX.nwi > PREFIX.nwo"


def test_mace_model_loads_take_turns(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Parallel first runs share MACE's fixed download path, so loads hold one lock."""
    import sys
    import types

    from filelock import Timeout

    utils = types.ModuleType("mace.tools.utils")
    utils.get_cache_dir = lambda: tmp_path / "cache" / "mace"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mace", types.ModuleType("mace"))
    monkeypatch.setitem(sys.modules, "mace.tools", types.ModuleType("mace.tools"))
    monkeypatch.setitem(sys.modules, "mace.tools.utils", utils)

    first = calculators._model_download_lock()
    second = calculators._model_download_lock()
    assert Path(first.lock_file).parent == tmp_path / "cache" / "mace"
    with first, pytest.raises(Timeout):
        second.acquire(timeout=0.1)
    with second:
        pass
