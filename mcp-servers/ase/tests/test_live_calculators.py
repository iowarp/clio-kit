"""Real calculations with xTB, MACE and NWChem (slow: run with --run-slow).

The first xTB/MACE test builds that calculator's worker environment in the user cache
(``ASE_MCP_ENV_DIR`` overrides it); MACE downloads MACE-OFF23 small (7 MB) once.
NWChem runs only when an nwchem executable is configured or on PATH."""

from __future__ import annotations

from pathlib import Path

import pytest

from ase_mcp import calculators

from .helpers import call

pytestmark = pytest.mark.slow


async def _ethane_saddle_and_relax(calc: dict) -> dict:
    s = (await call("smiles_to_structure", {"smiles": "CC", "name": "ethane"})).structured_content
    ecl = (
        await call(
            "set_geometry",
            {
                "structure": s["structure"]["path"],
                "dihedral": {"indices": [2, 0, 1, 5], "value": 0},
            },
        )
    ).structured_content["structure"]["path"]
    ts = (
        await call(
            "run_ase",
            {
                "structure": ecl,
                "calculator": calc,
                "driver": "vib",
                "fmax": 0.01,
                "constraints": [{"type": "dihedral", "indices": [2, 0, 1, 5]}],
            },
        )
    ).structured_content
    disp = (
        await call("displace_along_mode", {"source": ts["results_file"], "mode_index": 0})
    ).structured_content
    relaxed = (
        await call(
            "run_ase",
            {
                "structure": disp["structures"][0]["structure"]["path"],
                "calculator": calc,
                "driver": "vib",
                "fmax": 0.01,
            },
        )
    ).structured_content
    torsion = (
        await call(
            "inspect_structure",
            {"structure": relaxed["final_structure"]["path"], "measure": [[2, 0, 1, 5]]},
        )
    ).structured_content["measurements"][0]["value"]
    return {"ts": ts, "relaxed": relaxed, "torsion": torsion}


def _check_ethane(r: dict) -> None:
    ts, relaxed = r["ts"], r["relaxed"]
    assert ts["vibrations"]["n_imaginary"] == 1
    assert 200 < ts["vibrations"]["imaginary_cm1"][0] < 400  # the methyl torsion
    assert relaxed["vibrations"]["n_imaginary"] == 0
    assert abs(abs(r["torsion"]) - 60) < 2  # staggered
    barrier = ts["energy"]["kJ_mol"] - relaxed["energy"]["kJ_mol"]
    assert 8 < barrier < 16  # experiment about 12 kJ/mol


async def test_xtb_water_opt_vib_ir(out: Path) -> None:
    s = (await call("smiles_to_structure", {"smiles": "O"})).structured_content
    r = await call(
        "run_ase",
        {"structure": s["structure"]["path"], "calculator": "xtb", "driver": "ir", "fmax": 0.01},
    )
    d = r.structured_content
    assert d["calculator"]["method"] == "GFN2-xTB"
    assert d["optimization"]["converged"]
    f = sorted(m["frequency_cm1"] for m in d["vibrations"]["modes"])
    assert 1450 < f[0] < 1650 and 3550 < f[1] < 3750 and 3550 < f[2] < 3800
    assert Path(d["ir"]["spectrum_csv"]["path"]).is_file()
    assert d["dipole"]["debye"] > 1.5


async def test_xtb_triplet_o2_thermo(out: Path) -> None:
    from ase import Atoms
    from ase.io import write

    o2 = out / "o2.xyz"
    write(str(o2), Atoms("O2", positions=[[0, 0, 0], [0, 0, 1.21]]))
    d = (
        await call(
            "run_ase",
            {"structure": str(o2), "calculator": "xtb", "driver": "thermo", "multiplicity": 3},
        )
    ).structured_content
    th = d["thermochemistry"]
    assert th["geometry"] == "linear" and th["symmetry_number"] == 2 and th["spin_S"] == 1.0
    assert d["calculator"]["models_spin"] is True
    assert not any("does not model electronic spin" in n for n in d["notes"])


async def test_xtb_ethane_rotation_barrier(out: Path) -> None:
    _check_ethane(await _ethane_saddle_and_relax({"type": "xtb"}))


async def test_mace_ethane_rotation_barrier(out: Path) -> None:
    r = await _ethane_saddle_and_relax({"type": "mace"})
    _check_ethane(r)
    assert r["ts"]["calculator"]["model"] == "mace-off-small"
    assert r["ts"]["dipole"] is None


async def test_mace_notes_unmodelled_spin(out: Path, water: Path) -> None:
    d = (
        await call("run_ase", {"structure": str(water), "calculator": "mace", "multiplicity": 3})
    ).structured_content
    assert any("does not model electronic spin" in n for n in d["notes"])


@pytest.mark.integration
@pytest.mark.skipif(not calculators.nwchem_command()["found"], reason="no NWChem here")
async def test_nwchem_water_vib(out: Path) -> None:
    s = (await call("smiles_to_structure", {"smiles": "O"})).structured_content
    d = (
        await call(
            "run_ase",
            {
                "structure": s["structure"]["path"],
                "calculator": "nwchem",
                "driver": "vib",
                "fmax": 0.01,
            },
        )
    ).structured_content
    f = sorted(m["frequency_cm1"] for m in d["vibrations"]["modes"])
    # B3LYP/6-31G* water: about 1713, 3727, 3849 cm-1.
    assert f[0] == pytest.approx(1713, abs=25) and f[2] == pytest.approx(3849, abs=40)


@pytest.mark.integration
@pytest.mark.skipif(
    not __import__("ase_mcp.envs").envs.status("uma")["prepared"],
    reason="the uma environment is not built here (about 2 GB)",
)
async def test_uma_rejected_token_is_typed(
    out: Path, water: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from .helpers import call_error

    monkeypatch.setenv("HF_TOKEN", "hf_not_a_real_token")
    text = await call_error("run_ase", {"structure": str(water), "calculator": "uma"})
    assert text.endswith("[hf_token_missing]") and "facebook/UMA" in text
