"""The Materials Project fetch, against a recorded-shape reply through a mock transport."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from fdmnes_mcp import materials
from fdmnes_mcp.errors import FdmnesError


def _structure(a: float, elements: list[str], abc: list[list[float]], magmom: float = 0.0):
    return {
        "lattice": {"matrix": [[a, 0, 0], [0, a, 0], [0, 0, a]]},
        "sites": [
            {"species": [{"element": e, "occu": 1}], "abc": p, "properties": {"magmom": magmom}}
            for e, p in zip(elements, abc, strict=True)
        ],
    }


ROWS = [
    {
        "material_id": "mp-2",
        "formula_pretty": "AB",
        "energy_above_hull": 0.03,
        "is_stable": False,
        "symmetry": {"symbol": "Fm-3m", "crystal_system": "Cubic"},
        "nsites": 2,
        "ordering": "NM",
        "total_magnetization": 0.0,
        "structure": _structure(4.0, ["Na", "Cl"], [[0, 0, 0], [0.5, 0.5, 0.5]]),
    },
    {
        "material_id": "mp-1",
        "formula_pretty": "AB",
        "energy_above_hull": 0.0,
        "is_stable": True,
        "symmetry": {"symbol": "Pm-3m", "crystal_system": "Cubic"},
        "nsites": 2,
        "ordering": "FM",
        "total_magnetization": 4.0,
        "structure": _structure(3.0, ["Fe", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]], 2.0),
    },
]


def _client(seen: list[httpx.Request], status: int = 200) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json={"data": ROWS} if status == 200 else {})

    return httpx.Client(transport=httpx.MockTransport(handler))


def test_formula_returns_every_candidate_by_hull_energy(monkeypatch, out):
    monkeypatch.setenv("MP_API_KEY", "k")
    seen: list[httpx.Request] = []
    got = materials.fetch("FeO", 0.1, 5, None, _client(seen))
    assert seen[0].headers["X-API-KEY"] == "k"
    assert seen[0].url.params["formula"] == "FeO"
    # The limit is applied after the API sorts by hull energy, or the ground state can be cut.
    assert seen[0].url.params["_sort_fields"] == "energy_above_hull"
    assert [m["material_id"] for m in got["materials"]] == ["mp-1", "mp-2"]
    assert got["status"] == "several"
    first = got["materials"][0]
    from ase.io import read

    atoms = read(first["cif"])
    assert Path(first["cif"]).parent == out / "chemistry/fdmnes/structures"
    assert atoms.get_chemical_formula() == "FeO" and abs(atoms.cell[0][0] - 3.0) < 1e-6


def test_an_id_is_sent_as_an_id(monkeypatch, out):
    monkeypatch.setenv("MP_API_KEY", "k")
    seen: list[httpx.Request] = []
    materials.fetch("mp-19306", 0.05, 5, None, _client(seen))
    assert seen[0].url.params["material_ids"] == "mp-19306"
    assert "formula" not in seen[0].url.params


def test_no_key_and_rejected_key_are_typed(monkeypatch, out):
    monkeypatch.delenv("MP_API_KEY", raising=False)
    with pytest.raises(FdmnesError) as err:
        materials.fetch("FeO", 0.1, 5, None, _client([]))
    assert err.value.reason == "mp_key_missing" and "connector" in str(err.value)
    monkeypatch.setenv("MP_API_KEY", "bad")
    with pytest.raises(FdmnesError) as err:
        materials.fetch("FeO", 0.1, 5, None, _client([], status=401))
    assert err.value.reason == "mp_key_rejected"


def test_a_name_is_refused_with_a_hint(monkeypatch, out):
    monkeypatch.setenv("MP_API_KEY", "k")
    with pytest.raises(FdmnesError, match="formula first"):
        materials.fetch("magnetite", 0.1, 5, None, _client([]))
