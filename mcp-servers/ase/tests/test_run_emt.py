"""run_ase end to end with EMT: the worker process, the files, the one result format.

EMT is not a molecular method; these tests check the machinery, not chemistry."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from ase.io import read

from .helpers import call, call_error


def rows(path: str) -> list[dict[str, str]]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


async def test_tools_declare_paths() -> None:
    from fastmcp import Client

    from ase_mcp.server import mcp

    async with Client(mcp) as client:
        tools = {t.name: t for t in await client.list_tools()}
    assert set(tools) == {
        "smiles_to_structure",
        "inspect_structure",
        "set_geometry",
        "run_ase",
        "displace_along_mode",
        "trajectory_frame",
        "calculators",
    }
    meta = tools["run_ase"].meta or {}
    paths = meta.get("ai.iowarp.clio/paths") or meta.get("_fastmcp", {}).get("ai.iowarp.clio/paths")
    assert paths["structure"]["access"] == "read"
    assert tools["inspect_structure"].annotations.read_only_hint is True


async def test_energy(out: Path, water: Path) -> None:
    r = await call("run_ase", {"structure": str(water), "calculator": "emt"})
    d = r.structured_content
    assert d["schema"] == "ase-mcp/run@1" and d["status"] == "completed"
    assert d["energy"]["kJ_mol"] == pytest.approx(d["energy"]["eV"] * 96.4853321, rel=1e-6)
    assert "optimization" not in d and "final_structure" not in d
    assert Path(d["results_file"]).parent == out / "chemistry" / "ase" / "water-energy-emt"
    assert json.loads(Path(d["results_file"]).read_text())["energy"] == d["energy"]
    assert "eV" in r.content[0].text and "kJ/mol" in r.content[0].text


async def test_opt_writes_trajectory_and_steps_csv(out: Path, water: Path) -> None:
    d = (
        await call(
            "run_ase",
            {"structure": str(water), "calculator": {"type": "emt"}, "driver": "opt", "fmax": 0.01},
        )
    ).structured_content
    opt = d["optimization"]
    assert opt["converged"] is True
    steps = rows(opt["steps_csv"]["path"])
    assert list(steps[0]) == ["step", "energy_eV", "fmax_eV_per_A"]
    frames = read(opt["trajectory"]["path"], index=":")
    assert len(frames) == len(steps) == opt["steps"] + 1
    assert frames[-1].info["energy_eV"] == pytest.approx(float(steps[-1]["energy_eV"]))
    assert float(steps[-1]["fmax_eV_per_A"]) < 0.01
    assert d["final_structure"]["n_atoms"] == 3


async def test_thermo_files_and_numbers(out: Path, water: Path) -> None:
    r = await call(
        "run_ase",
        {
            "structure": str(water),
            "calculator": "emt",
            "driver": "thermo",
            "multiplicity": 3,
            "run_name": "w",
        },
    )
    d = r.structured_content
    vib = d["vibrations"]
    assert vib["n_modes"] == 3
    freq = rows(vib["frequencies_csv"]["path"])
    assert list(freq[0]) == ["mode", "frequency_cm1", "imaginary", "label", "energy_meV"]
    anim = read(vib["modes"][0]["animation"], index=":")
    assert len(anim) == 20
    th = d["thermochemistry"]
    assert th["multiplicity"] == 3 and th["spin_S"] == 1.0 and th["symmetry_number"] == 2
    assert {"eV", "kJ_mol"} <= set(th["gibbs_free_energy"])
    assert any("does not model electronic spin" in n for n in d["notes"])
    text = r.content[0].text
    assert "multiplicity 3 (spin not modelled by emt)" in text and "G =" in text


async def test_constraint_holds_dihedral(out: Path, ethane: Path) -> None:
    d = (
        await call(
            "run_ase",
            {
                "structure": str(ethane),
                "calculator": "emt",
                "driver": "opt",
                "constraints": [{"type": "dihedral", "indices": [2, 0, 1, 5], "value_deg": 0}],
            },
        )
    ).structured_content
    held = d["final_constraint_values"][0]["value"]
    assert abs(held) < 0.05
    assert d["constraints"][0]["value_deg"] == pytest.approx(0.0, abs=1e-6)


async def test_typed_errors(out: Path, water: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    text = await call_error(
        "run_ase", {"structure": str(water), "calculator": "emt", "driver": "ir"}
    )
    assert text.endswith("[dipole_unavailable]")
    text = await call_error(
        "run_ase", {"structure": str(water), "calculator": {"type": "xtb", "basis": "x"}}
    )
    assert text.endswith("[invalid_input]")
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("HF_HOME", str(out / "hf"))
    text = await call_error("run_ase", {"structure": str(water), "calculator": "uma"})
    assert text.endswith("[hf_token_missing]")
    for name in ("ORCA_COMMAND", "ASE_ORCA_COMMAND"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("PATH", str(out / "empty"))
    text = await call_error("run_ase", {"structure": str(water), "calculator": "orca"})
    assert text.endswith("[binary_missing]")


async def test_worker_failure_is_typed(out: Path, tmp_path: Path) -> None:
    gold = tmp_path / "x.xyz"
    gold.write_text("1\n\nXx 0 0 0\n")
    text = await call_error("run_ase", {"structure": str(gold), "calculator": "emt"})
    assert text.endswith(("[unreadable_structure]", "[calculation_failed]"))


async def test_calculators_tool(out: Path) -> None:
    d = (await call("calculators", {})).structured_content
    kinds = {c["type"]: c for c in d["calculators"]}
    assert set(kinds) == {"emt", "xtb", "mace", "uma", "nwchem", "orca"}
    assert kinds["emt"]["state"] == "ready"
    assert "environment" in kinds["mace"]
