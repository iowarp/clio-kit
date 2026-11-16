"""The tools end to end through an in-memory MCP client."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from fdmnes_mcp.server import mcp

from .conftest import FIXTURES, REAL


async def call(name: str, args: dict) -> tuple[str, dict]:
    async with Client(mcp) as client:
        res = await client.call_tool(name, args)
    return res.content[0].text, res.structured_content


async def test_run_xanes_with_stand_in_fdmnes(fake_fdmnes, copper_cif, out):
    text, data = await call("run_xanes", {"structure": str(copper_cif), "absorber": "Cu"})
    run = Path(data["run_dir"])
    assert run.parent == out / "chemistry" / "fdmnes" and run.name == "cu-Cu-K"
    assert data["status"] == "completed"
    assert data["log"]["inequivalent_absorbers"] == 1 and data["log"]["scf_cycles"] == 3
    spec = data["spectra"][0]
    assert Path(spec["csv"]).name == "spectrum.csv"
    head = Path(spec["csv"]).read_text().splitlines()[0]
    assert head == "energy_eV,mu,photon_energy_eV"  # -10..40 eV is too short to normalize
    assert not spec["normalized"] and "extend the energy range" in spec["normalization_skipped"]
    assert json.loads((run / "result.json").read_text())["status"] == "completed"
    assert "K-edge XANES of Cu4" in text and "1 inequivalent absorbing site" in text
    assert "not normalized" in text


async def test_run_names_do_not_overwrite(fake_fdmnes, copper_cif, out):
    _, a = await call("run_xanes", {"structure": str(copper_cif), "absorber": "Cu"})
    _, b = await call("run_xanes", {"structure": str(copper_cif), "absorber": "Cu"})
    assert Path(b["run_dir"]).name == "cu-Cu-K-2" and a["run_dir"] != b["run_dir"]


async def test_failed_fdmnes_is_a_typed_error_and_keeps_the_folder(failing_fdmnes, copper_cif, out):
    with pytest.raises(ToolError, match=r"(?s)exited with code 3.*\[fdmnes_failed\]") as err:
        await call("run_xanes", {"structure": str(copper_cif), "absorber": "Cu"})
    assert "Error in the input file" in str(err.value)
    saved = json.loads((out / "chemistry/fdmnes/cu-Cu-K/result.json").read_text())
    assert saved["status"] == "failed" and saved["error"]["reason"] == "fdmnes_failed"


async def test_missing_executable(monkeypatch, copper_cif, out):
    monkeypatch.delenv("FDMNES_COMMAND", raising=False)
    monkeypatch.delenv("FDMNES_EXE", raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(ToolError, match=r"spack install fdmnes.*\[binary_missing\]"):
        await call("run_xanes", {"structure": str(copper_cif), "absorber": "Cu"})
    text, data = await call("fdmnes_status", {})
    assert not data["executable"]["found"] and "not available" in text


async def test_missing_structure(fake_fdmnes, out):
    with pytest.raises(ToolError, match=r"\[not_found\]"):
        await call("run_xanes", {"structure": "nope.cif", "absorber": "Cu"})


async def test_read_and_compare(out, tmp_path):
    import numpy as np

    e = np.arange(-40.0, 80.0, 0.5)
    mu = 0.1 + 2.0 / (1 + np.exp(-e / 1.5))
    calc = tmp_path / "calc_conv.txt"
    calc.write_text("Energy <xanes>\n" + "\n".join(f"{a} {b}" for a, b in zip(e, mu)))
    meas = tmp_path / "measured.csv"
    meas.write_text("E,mu\n" + "\n".join(f"{a + 7112},{3 * b}" for a, b in zip(e, mu)))
    text, d = await call("read_spectrum", {"path": str(meas)})
    assert d["normalized"] and abs(d["e0_eV"] - 7112) < 0.6 and "Wrote" in text
    text, d = await call(
        "compare_spectra",
        {"paths": [str(calc), str(meas)], "labels": ["FDMNES", "measured"], "align": True},
    )
    assert d["spectra"][1]["correlation_vs_first"] > 0.999
    assert Path(d["csv"]).read_text().splitlines()[0] == "energy_eV,FDMNES,measured"
    assert "R-factor" in text


@pytest.mark.integration
@pytest.mark.skipif(not REAL, reason="set FDMNES_COMMAND to the FDMNES executable")
async def test_real_fdmnes_copper_k_edge(copper_cif, out):
    _text, data = await call(
        "run_xanes",
        {
            "structure": str(copper_cif),
            "absorber": "Cu",
            "radius": 4.0,
            "energy_range": [-30, 1.0, -10, 0.5, 20, 1.0, 60],
        },
    )
    assert data["log"]["inequivalent_absorbers"] == 1
    assert data["log"]["point_group"].startswith("m3m")
    spec = data["spectra"][0]
    assert spec["normalized"] and -10 < spec["e0_eV"] < 10
    assert os.path.getsize(spec["csv"]) > 1000


@pytest.mark.integration
@pytest.mark.skipif(not REAL, reason="set FDMNES_COMMAND to the FDMNES executable")
async def test_real_fdmnes_reads_molecule_in_angstrom(feo6_xyz, out):
    _, data = await call(
        "run_xanes",
        {
            "structure": str(feo6_xyz),
            "absorber": "Fe",
            "radius": 3.0,
            "scf": False,
            "energy_range": [-20, 1.0, 40],
        },
    )
    bav = (Path(data["run_dir"]) / "xanes_bav.txt").read_text()
    assert data["log"]["point_group"].startswith("m3m")  # octahedron, Fe–O 2.0 Å
    assert "nb. of atom =   7" in bav or "nb. of atom = 7" in bav


def test_fixture_exists():
    assert (FIXTURES / "cu_conv_fdmnes.txt").is_file()
