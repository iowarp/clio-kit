"""SMILES to 3D, inspect, set_geometry, trajectory_frame and displace_along_mode."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.build import molecule
from ase.io import read, write

from .helpers import call, call_error


async def test_smiles_to_structure(out: Path) -> None:
    r = await call("smiles_to_structure", {"smiles": "OCC", "name": "ethanol"})
    d = r.structured_content
    assert d["canonical_smiles"] == "CCO" and d["formula"] == "C2H6O"
    assert d["force_field"] == "MMFF94" and d["embedding"] == "ETKDGv3"
    assert d["structure"]["n_atoms"] == 9 and d["structure"]["format"] == "xyz"
    path = Path(d["structure"]["path"])
    assert path.parent == out / "chemistry" / "ase" / "structures"
    assert len(read(str(path))) == 9 and Path(d["sdf"]["path"]).is_file()
    assert len(d["structure"]["sha256"]) == 64
    assert "CCO" in r.content[0].text and str(path) in r.content[0].text
    again = await call("smiles_to_structure", {"smiles": "OCC", "name": "ethanol"})
    assert again.structured_content["structure"]["path"] != str(path)  # never overwrites


async def test_smiles_facts(out: Path) -> None:
    ion = (await call("smiles_to_structure", {"smiles": "[NH4+]"})).structured_content
    assert ion["formal_charge"] == 1
    radical = (await call("smiles_to_structure", {"smiles": "[CH3]"})).structured_content
    assert radical["radical_electrons"] == 1 and radical["multiplicity_from_smiles"] == 2
    chiral = (await call("smiles_to_structure", {"smiles": "CC(N)C(=O)O"})).structured_content
    assert any("Stereocentres" in n for n in chiral["notes"])


async def test_invalid_smiles(out: Path) -> None:
    text = await call_error("smiles_to_structure", {"smiles": "C1CC(("})
    assert text.endswith("[invalid_smiles]")


async def test_inspect_and_missing_file(out: Path, ethane: Path) -> None:
    r = await call(
        "inspect_structure", {"structure": str(ethane), "measure": [[0, 1], [2, 0, 1, 5]]}
    )
    d = r.structured_content
    assert d["formula"] == "C2H6" and d["n_bonds"] == 7
    assert [m["kind"] for m in d["measurements"]] == ["distance", "dihedral"]
    assert "C0-C1" in r.content[0].text
    text = await call_error("inspect_structure", {"structure": str(out / "nope.xyz")})
    assert text.endswith("[not_found]")


async def test_set_geometry(out: Path, ethane: Path) -> None:
    r = await call(
        "set_geometry",
        {
            "structure": str(ethane),
            "dihedral": {"indices": [2, 0, 1, 5], "value": 0},
            "distance": {"indices": [0, 1], "value": 1.6},
        },
    )
    d = r.structured_content
    new = read(d["structure"]["path"])
    assert (
        abs(new.get_dihedral(2, 0, 1, 5) % 360) < 1e-3
        or abs(new.get_dihedral(2, 0, 1, 5) % 360 - 360) < 1e-3
    )
    assert new.get_distance(0, 1) == pytest.approx(1.6, abs=1e-5)
    assert {c["kind"] for c in d["changes"]} == {"dihedral", "distance"}
    benzene = out / "benzene.xyz"
    write(str(benzene), molecule("C6H6"))
    text = await call_error(
        "set_geometry",
        {"structure": str(benzene), "dihedral": {"indices": [6, 0, 1, 7], "value": 30}},
    )
    assert text.endswith("[ring_bond]")


async def test_trajectory_frame(out: Path) -> None:
    frames = []
    for k in range(4):
        a = molecule("H2O")
        a.positions[1, 0] += 0.01 * k
        a.info = {"energy_eV": -1.0 - k, "step": k}
        frames.append(a)
    traj = out / "traj.xyz"
    write(str(traj), frames, format="extxyz")
    d = (await call("trajectory_frame", {"trajectory": str(traj), "frame": 2})).structured_content
    assert d["n_frames"] == 4 and d["frame"] == 2 and d["energy_eV"] == -3.0
    assert read(d["structure"]["path"]).positions[1, 0] == pytest.approx(frames[2].positions[1, 0])
    last = (await call("trajectory_frame", {"trajectory": str(traj)})).structured_content
    assert last["frame"] == 3
    text = await call_error("trajectory_frame", {"trajectory": str(traj), "frame": 9})
    assert text.endswith("[invalid_input]")


async def test_displace_along_mode_from_npz_json_and_animation(out: Path) -> None:
    atoms = molecule("H2O")
    modes = np.zeros((3, 3, 3))
    modes[0, 1, 0] = 1.0
    modes[1, 2, 1] = 2.0
    npz = out / "normal_modes.npz"
    np.savez(
        npz,
        numbers=atoms.numbers,
        positions=atoms.positions,
        modes=modes,
        frequencies_cm1=np.array([-300.0, 1600.0, 3700.0]),
    )
    d = (
        await call(
            "displace_along_mode",
            {"source": str(npz), "mode_index": 0, "amplitude": 0.2, "both": True},
        )
    ).structured_content
    assert d["imaginary"] is True and len(d["structures"]) == 2
    plus = read(d["structures"][0]["structure"]["path"])
    minus = read(d["structures"][1]["structure"]["path"])
    assert plus.positions[1, 0] - atoms.positions[1, 0] == pytest.approx(0.2)
    assert minus.positions[1, 0] - atoms.positions[1, 0] == pytest.approx(-0.2)
    result = out / "result.json"
    result.write_text(json.dumps({"vibrations": {"modes_file": {"path": str(npz)}}}))
    d = (
        await call("displace_along_mode", {"source": str(result), "mode_index": 1})
    ).structured_content
    moved = read(d["structures"][0]["structure"]["path"])
    assert moved.positions[2, 1] - atoms.positions[2, 1] == pytest.approx(0.3)
    text = await call_error("displace_along_mode", {"source": str(npz), "mode_index": 7})
    assert text.endswith("[invalid_input]")
