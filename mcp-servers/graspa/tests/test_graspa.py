from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastmcp import Client
from fastmcp.exceptions import ToolError

from graspa_mcp import parse
from graspa_mcp.errors import GraspaError
from graspa_mcp.inputs import cell_replication, read_framework
from graspa_mcp.server import mcp

from .conftest import FIXTURES


async def call(name: str, args: dict) -> tuple[str, dict]:
    async with Client(mcp) as client:
        res = await client.call_tool(name, args)
    return res.content[0].text, res.structured_content


def test_cells_cover_twice_the_cutoff(frameworks):
    assert cell_replication(read_framework(frameworks["small"]), 12.8) == [3, 3, 3]
    assert cell_replication(read_framework(frameworks["large"]), 12.8) == [1, 1, 1]


def test_cuda_output_takes_the_adsorbate_block_by_name():
    text = (FIXTURES / "graspa_cuda_co2_mfi_stdout.txt").read_text()
    got = parse.read_loading(text, "CO2", None, None)
    assert got["flavour"] == "graspa"
    assert got["uptake_mol_kg"] == pytest.approx(0.44065)  # not the framework's 0.02167
    assert got["uptake_error_mol_kg"] == pytest.approx(0.00884)
    assert got["molecules_in_box"] == pytest.approx(20.3334)
    assert got["uptake_mg_per_g"] == pytest.approx(19.38806)
    assert got["unit_cells"] == [1, 1, 1]
    with pytest.raises(GraspaError, match="not N2"):
        parse.read_loading(text, "N2", None, None)


def test_sycl_output_converts_molecules_with_the_framework_mass():
    got = parse.read_loading("UnitCells 0 2 2 2\nOverall: Average: 12.011,\n", "H2O", None, 1.5015)
    assert got["flavour"] == "graspa-sycl"
    assert got["uptake_mol_kg"] == pytest.approx(12.011 / (1.5015 * 8) * 1000)


def test_no_loading_is_unreadable():
    with pytest.raises(GraspaError) as err:
        parse.read_loading("INITIALIZATION Cycle: 0\n", "CO2", [1, 1, 1], 100.0)
    assert err.value.reason == "unreadable_output"


async def test_run_graspa_end_to_end(cuda_graspa, frameworks, out):
    text, d = await call(
        "run_graspa",
        {
            "structure": str(frameworks["small"]),
            "adsorbate": "CO2",
            "pressure": 10000,
            "prod_cycles": 100,
            "init_cycles": 100,
        },
    )
    run = Path(d["run_dir"])
    assert run == out / "chemistry/graspa/small-CO2-298.15K-10000Pa"
    inp = (run / "simulation.input").read_text()
    assert "UnitCells 0 3 3 3" in inp and "UseChargesFromCIFFile no" in inp
    assert "Pressure    10000" in inp and "NumberOfProductionCycles     100" in inp
    assert d["loading"]["uptake_mol_kg"] == pytest.approx(0.44065)
    assert json.loads((run / "result.json").read_text())["status"] == "completed"
    assert "uptake 0.4406 ± 0.00884 mol/kg" in text
    assert "no framework–adsorbate electrostatics" in text


async def test_framework_charges_from_cif(cuda_graspa, charged_cif, frameworks, out):
    _, d = await call(
        "prepare_graspa",
        {"structure": str(charged_cif), "adsorbate": "CO2", "framework_charges": "cif"},
    )
    assert "UseChargesFromCIFFile yes" in (Path(d["run_dir"]) / "simulation.input").read_text()
    assert d["notes"] == []
    with pytest.raises(ToolError, match=r"has none.*\[invalid_input\]"):
        await call(
            "prepare_graspa",
            {"structure": str(frameworks["small"]), "adsorbate": "CO2", "framework_charges": "cif"},
        )


async def test_prepare_then_read_from_another_machine(sycl_graspa, frameworks, out):
    text, d = await call(
        "prepare_graspa", {"structure": str(frameworks["large"]), "adsorbate": "H2O"}
    )
    run = Path(d["run_dir"])
    assert not (run / "result.json").exists() and "GPU node" in text
    with pytest.raises(ToolError, match=r"no gRASPA output yet.*\[unreadable_output\]"):
        await call("read_graspa", {"run_dir": str(run)})
    (run / "graspa_stdout.txt").write_text("UnitCells 0 1 1 1\nOverall: Average: 6.0,\n")
    text, r = await call("read_graspa", {"run_dir": str(run / "run.json")})
    mass = r["structure"]["framework_mass_g_mol"]
    assert r["loading"]["uptake_mol_kg"] == pytest.approx(6.0 / mass * 1000)
    assert "no error bar" in text


async def test_crash_is_typed_and_recorded(crashing_graspa, frameworks, out):
    with pytest.raises(ToolError, match=r"(?s)code 137.*device failure.*\[graspa_failed\]"):
        await call("run_graspa", {"structure": str(frameworks["small"]), "adsorbate": "N2"})
    saved = next((out / "chemistry/graspa").glob("*/result.json"))
    assert json.loads(saved.read_text())["status"] == "failed"


async def test_missing_executable_points_to_prepare(monkeypatch, frameworks, out):
    for var in ("GRASPA_COMMAND", "CHEMGRAPH_GRASPA_EXECUTABLE"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("PATH", "/nonexistent")
    with pytest.raises(ToolError, match=r"prepare_graspa.*\[binary_missing\]"):
        await call("run_graspa", {"structure": str(frameworks["small"]), "adsorbate": "CO2"})
    text, d = await call("graspa_status", {})
    assert not d["executable"]["found"] and "GPU" in text


async def test_rank_by_working_capacity_excludes_incomplete(sycl_graspa, frameworks, out):
    async def point(name: str, pressure: float, molecules: float) -> None:
        _, d = await call(
            "prepare_graspa",
            {"structure": str(frameworks[name]), "adsorbate": "CO2", "pressure": pressure},
        )
        Path(d["run_dir"], "graspa_stdout.txt").write_text(f"Overall: Average: {molecules},\n")
        await call("read_graspa", {"run_dir": d["run_dir"]})

    await point("small", 100000, 40)
    await point("small", 10000, 10)
    await point("large", 100000, 30)  # no desorption run
    text, d = await call(
        "rank_frameworks",
        {
            "results": [str(out / "chemistry/graspa")],
            "adsorption": {"temperature_K": 298.15, "pressure_Pa": 100000},
            "desorption": {"temperature_K": 298.15, "pressure_Pa": 10000},
        },
    )
    assert d["ranked"] == 1 and d["top"][0]["framework"] == "small"
    assert d["top"][0]["working_capacity"] > 0
    assert d["excluded"][0]["reason"].startswith("no run at 298.15 K/10000 Pa")
    assert Path(d["csv"]).read_text().splitlines()[0] == (
        "rank,framework,uptake_ads,uptake_des,working_capacity,repeats,structure"
    )
    assert "Excluded" in text
