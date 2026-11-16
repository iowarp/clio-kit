"""Crystal structures from the Materials Project, as CIF files.

ChemGraph's XANES workflow fetched structures with ``mp_api.MPRester``
(``tools/xanes_core.py`` ``fetch_materials_project_data``). This is a small client of
the same REST API (``api.materialsproject.org``, ``X-API-KEY``) with no mp-api or
pymatgen dependency: the summary endpoint's structure (a pymatgen dict: lattice matrix
and fractional site coordinates) becomes an ASE ``Atoms`` written as CIF. A formula
returns every matching material up to ``max_results``, ordered by energy above the hull,
and writes each one; none is chosen for the caller.
"""

from __future__ import annotations

import os
import re
from typing import Any

import httpx

from . import records
from .errors import FdmnesError

API = "https://api.materialsproject.org/materials/summary/"
KEY_ENV = "MP_API_KEY"
FIELDS = (
    "material_id,formula_pretty,energy_above_hull,is_stable,symmetry,nsites,"
    "structure,total_magnetization,ordering"
)
MP_ID = re.compile(r"^mp-\d+$")


def _key() -> str:
    key = os.environ.get(KEY_ENV, "").strip()
    if not key:
        raise FdmnesError(
            "mp_key_missing",
            "Fetching from the Materials Project needs an API key (free, from "
            "next-gen.materialsproject.org/api after signing in). Set it in the FDMNES "
            "connector's Materials Project API key, or as MP_API_KEY.",
        )
    return key


def _get(client: httpx.Client, params: dict[str, Any]) -> list[dict[str, Any]]:
    try:
        resp = client.get(API, params={**params, "_fields": FIELDS}, headers={"X-API-KEY": _key()})
    except httpx.HTTPError as exc:
        raise FdmnesError(
            "mp_unavailable", f"The Materials Project could not be reached: {exc}."
        ) from exc
    if resp.status_code in (401, 403):
        raise FdmnesError(
            "mp_key_rejected",
            "The Materials Project rejected the API key (HTTP "
            f"{resp.status_code}); check the key in the FDMNES connector.",
        )
    if resp.status_code != 200:
        raise FdmnesError(
            "mp_unavailable",
            f"The Materials Project answered HTTP {resp.status_code}: {resp.text[:200]}",
        )
    return list(resp.json().get("data") or [])


def to_atoms(structure: dict[str, Any]) -> Any:
    """ASE Atoms from a pymatgen Structure dict (lattice matrix, fractional sites)."""
    from ase import Atoms

    cell = structure["lattice"]["matrix"]
    symbols, frac, magmoms = [], [], []
    for site in structure["sites"]:
        species = max(site["species"], key=lambda s: s.get("occu", 1.0))
        if len(site["species"]) > 1 or species.get("occu", 1.0) < 0.999:
            raise FdmnesError(
                "invalid_input",
                "This Materials Project structure has partial occupancies, which this "
                "fetch does not write; choose an ordered material.",
            )
        symbols.append(species["element"])
        frac.append(site["abc"])
        magmoms.append(float((site.get("properties") or {}).get("magmom") or 0.0))
    atoms = Atoms(symbols=symbols, scaled_positions=frac, cell=cell, pbc=True)
    if any(magmoms):
        atoms.set_initial_magnetic_moments(magmoms)
    return atoms


def fetch(
    query: str,
    energy_above_hull_max: float,
    max_results: int,
    output_dir: str | None,
    client: httpx.Client | None = None,
) -> dict[str, Any]:
    from ase.io import write

    query = query.strip()
    params: dict[str, Any] = {"deprecated": "false", "_limit": max_results}
    if MP_ID.match(query):
        params["material_ids"] = query
    elif re.fullmatch(r"[A-Z][A-Za-z0-9().]*", query):
        params["formula"] = query
        params["energy_above_hull_max"] = energy_above_hull_max
    else:
        raise FdmnesError(
            "invalid_input",
            f"{query!r}: give a formula ('Fe3O4') or a Materials Project id ('mp-19306'). "
            "For a mineral or common name, find its formula first.",
        )
    own = client is None
    client = client or httpx.Client(timeout=60)
    try:
        rows = _get(client, params)
    finally:
        if own:
            client.close()
    if not rows:
        raise FdmnesError(
            "not_found",
            f"The Materials Project has no material for {query!r}"
            + (
                f" within {energy_above_hull_max:g} eV/atom of the hull"
                if "formula" in params
                else ""
            )
            + ".",
        )
    rows.sort(key=lambda r: (r.get("energy_above_hull") or 0.0, r["material_id"]))
    folder = records.output_root(output_dir) / "structures"
    folder.mkdir(parents=True, exist_ok=True)
    materials = []
    for row in rows[:max_results]:
        atoms = to_atoms(row["structure"])
        sym = row.get("symmetry") or {}
        path = folder / f"{row['material_id']}_{records.slug(row['formula_pretty'])}.cif"
        write(str(path), atoms, format="cif")
        materials.append(
            {
                "material_id": row["material_id"],
                "formula": row["formula_pretty"],
                "space_group": sym.get("symbol"),
                "crystal_system": sym.get("crystal_system"),
                "energy_above_hull_eV_atom": row.get("energy_above_hull"),
                "is_stable": row.get("is_stable"),
                "n_sites": len(atoms),
                "magnetic_ordering": row.get("ordering"),
                "total_magnetization_muB": row.get("total_magnetization"),
                "cif": str(path),
                "sha256": records.sha256(path),
                "url": f"https://next-gen.materialsproject.org/materials/{row['material_id']}",
            }
        )
    return {
        "query": query,
        "status": "one" if len(materials) == 1 else "several",
        "materials": materials,
        "note": (
            "Ordered by energy above the hull (0 is the computed ground state); several "
            "polymorphs or magnetic orderings can match a formula, so say which one is used."
        ),
    }
