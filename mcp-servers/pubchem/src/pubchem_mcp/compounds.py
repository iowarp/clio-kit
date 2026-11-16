"""Name to compound, compound records and 3D structures from PubChem.

The name lookup never chooses for the agent: an exact name that PubChem maps to one CID
is ``resolved``; anything else (several CIDs, or none) gathers candidates from
PubChem's autocomplete dictionary, its word search and the exact lookups of the
suggested names, and returns them as ``ambiguous`` for the person to choose from.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any

from .client import AUTOCOMPLETE, PUG, PubChemClient, quote
from .errors import PubChemError

PROPERTIES = (
    "Title,IUPACName,MolecularFormula,MolecularWeight,SMILES,ConnectivitySMILES,InChIKey,"
    "Charge,HeavyAtomCount,AtomStereoCount,DefinedAtomStereoCount,UndefinedAtomStereoCount,"
    "BondStereoCount,UndefinedBondStereoCount,CovalentUnitCount"
)
SYNONYMS_PER_CANDIDATE = 6
SYNONYMS_PER_COMPOUND = 20
MAX_CANDIDATES = 25


def compound_url(cid: int) -> str:
    return f"https://pubchem.ncbi.nlm.nih.gov/compound/{cid}"


def _record(row: dict[str, Any]) -> dict[str, Any]:
    """PubChem's property row as a typed record with units in the names."""
    weight = row.get("MolecularWeight")
    return {
        "cid": int(row["CID"]),
        "title": row.get("Title"),
        "iupac_name": row.get("IUPACName"),
        "formula": row.get("MolecularFormula"),
        "molecular_weight_g_mol": float(weight) if weight is not None else None,
        # PubChem's SMILES carries stereo and isotopes; ConnectivitySMILES drops them.
        "isomeric_smiles": row.get("SMILES"),
        "canonical_smiles": row.get("ConnectivitySMILES"),
        "inchikey": row.get("InChIKey"),
        "charge": row.get("Charge"),
        "heavy_atoms": row.get("HeavyAtomCount"),
        "stereo": {
            "atom_centres": row.get("AtomStereoCount", 0),
            "defined_atom_centres": row.get("DefinedAtomStereoCount", 0),
            "undefined_atom_centres": row.get("UndefinedAtomStereoCount", 0),
            "bond_centres": row.get("BondStereoCount", 0),
            "undefined_bond_centres": row.get("UndefinedBondStereoCount", 0),
        },
        "components": row.get("CovalentUnitCount"),
        "url": compound_url(int(row["CID"])),
    }


async def properties(client: PubChemClient, cids: list[int]) -> list[dict[str, Any]]:
    """Property records for ``cids`` in one request, in the order given."""
    if not cids:
        return []
    ids = ",".join(str(c) for c in cids)
    data = await client.json(f"{PUG}/compound/cid/{ids}/property/{PROPERTIES}/JSON")
    # An unknown CID comes back as a row holding only its CID.
    rows = {
        int(r["CID"]): _record(r)
        for r in data["PropertyTable"]["Properties"]
        if r.get("MolecularFormula")
    }
    return [rows[c] for c in cids if c in rows]


async def synonyms(client: PubChemClient, cids: list[int], limit: int) -> dict[int, list[str]]:
    """The first ``limit`` synonyms of each CID (PubChem lists the most used first)."""
    if not cids:
        return {}
    ids = ",".join(str(c) for c in cids)
    try:
        data = await client.json(f"{PUG}/compound/cid/{ids}/synonyms/JSON")
    except PubChemError as exc:
        if exc.reason == "not_found":
            return {}
        raise
    return {
        int(item["CID"]): list(item.get("Synonym", []))[:limit]
        for item in data["InformationList"]["Information"]
    }


async def name_cids(client: PubChemClient, name: str, word: bool = False) -> list[int]:
    """CIDs PubChem maps ``name`` to: exactly (default) or by whole word. [] when none."""
    params = {"name_type": "word"} if word else None
    try:
        data = await client.json(f"{PUG}/compound/name/{quote(name)}/cids/JSON", params)
    except PubChemError as exc:
        if exc.reason == "not_found":
            return []
        raise
    return [int(c) for c in data.get("IdentifierList", {}).get("CID", []) if int(c) > 0]


async def autocomplete(client: PubChemClient, name: str, limit: int) -> list[str]:
    """Compound names PubChem's dictionary suggests for ``name``."""
    try:
        data = await client.json(f"{AUTOCOMPLETE}/{quote(name)}/json", {"limit": limit})
    except PubChemError as exc:
        if exc.reason == "not_found":
            return []
        raise
    return list(data.get("dictionary_terms", {}).get("compound", []))


def _closeness(term: str, name: str) -> tuple[int, int]:
    """Order suggestions: those containing the name as a word first, then fewest extra
    characters (``p-xylene`` before ``2,5-dichloro-p-xylene``)."""
    low, query = term.lower(), name.lower().strip()
    whole = re.search(rf"(?<![a-z]){re.escape(query)}(?![a-z])", low) is not None
    return (0 if whole else 1, len(low) - len(query))


async def resolve_name(client: PubChemClient, name: str, max_candidates: int = 8) -> dict[str, Any]:
    """Resolve a compound name; see the module docstring for the statuses."""
    name = name.strip()
    if not name:
        raise PubChemError("bad_request", "Give a compound name, e.g. 'p-xylene' or 'caffeine'.")
    limit = max(1, min(int(max_candidates), MAX_CANDIDATES))
    exact = await name_cids(client, name)
    if len(exact) == 1:
        record = (await properties(client, exact))[0]
        record["synonyms"] = (await synonyms(client, exact, SYNONYMS_PER_CANDIDATE)).get(
            exact[0], []
        )
        return {
            "query": name,
            "status": "resolved",
            "compound": record,
            "sources": ["exact"],
        }

    # Several CIDs or none: collect candidates, each with how it was found.
    found: dict[int, list[str]] = {}

    def add(cid: int, how: str) -> None:
        found.setdefault(cid, [])
        if how not in found[cid]:
            found[cid].append(how)

    for cid in exact:
        add(cid, "exact name")
    suggestions = sorted(
        await autocomplete(client, name, limit * 2), key=lambda t: _closeness(t, name)
    )
    for term in suggestions[:limit]:
        for cid in (await name_cids(client, term))[:3]:
            add(cid, f"suggested name '{term}'")
    word_hits = await name_cids(client, name, word=True)
    for cid in word_hits[: limit * 2]:
        if len(found) >= limit * 2:
            break
        add(cid, "name contains the word")
    if not found:
        raise PubChemError(
            "not_found",
            f"PubChem knows no compound named '{name}', and its dictionary suggests none. "
            "Check the spelling, or give a CAS number, a more specific name, or a SMILES.",
            query=name,
        )
    cids = list(found)[:limit]
    records = await properties(client, cids)
    names = await synonyms(client, cids, SYNONYMS_PER_CANDIDATE)
    for record in records:
        record["found_by"] = found[record["cid"]]
        record["synonyms"] = names.get(record["cid"], [])
    if exact:
        why = f"PubChem maps the name '{name}' to {len(exact)} different compounds."
    else:
        why = (
            f"No compound in PubChem is named exactly '{name}'. Candidates from PubChem's "
            "name suggestions and word search; found_by says how each was found."
        )
    return {
        "query": name,
        "status": "ambiguous",
        "reason": why,
        "candidates": records,
        "word_search_hits": len(word_hits),
        "suggested_names": suggestions[:limit],
        "next_step": "Ask the person which compound they mean (show title, formula and "
        "SMILES of each candidate); do not pick one for them.",
    }


async def get_compound(client: PubChemClient, cid: int) -> dict[str, Any]:
    """Properties and the first synonyms of one CID."""
    cid = _check_cid(cid)
    try:
        records = await properties(client, [cid])
    except PubChemError as exc:
        if exc.reason in {"not_found", "bad_request"}:
            raise PubChemError(
                "not_found", f"PubChem has no compound with CID {cid}.", cid=cid
            ) from None
        raise
    if not records:
        raise PubChemError("not_found", f"PubChem has no compound with CID {cid}.", cid=cid)
    record = records[0]
    record["synonyms"] = (await synonyms(client, [cid], SYNONYMS_PER_COMPOUND)).get(cid, [])
    return record


def _check_cid(cid: int | str) -> int:
    try:
        value = int(cid)
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        raise PubChemError(
            "bad_request",
            f"A CID is a positive whole number; got {cid!r}.",
            cid=str(cid),
        )
    return value


def output_dir(base: str | None, run_name: str) -> Path:
    """``<base or $CLIO_KIT_ARTIFACTS or cwd>/chemistry/pubchem/<run_name>/``, created."""
    root = Path(base or os.environ.get("CLIO_KIT_ARTIFACTS") or os.getcwd()).expanduser()
    folder = root / "chemistry" / "pubchem" / run_name
    folder.mkdir(parents=True, exist_ok=True)
    return folder.resolve()


def _slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")[:40] or "compound"


def sdf_atom_count(sdf: str) -> int:
    """Atoms in the first molecule of an SDF, from its V2000 counts line (or V3000)."""
    lines = sdf.splitlines()
    if len(lines) > 3 and "V3000" not in lines[3]:
        return int(lines[3][:3])
    match = re.search(r"M\s+V30 COUNTS (\d+)", sdf)
    return int(match.group(1)) if match else 0


async def fetch_structure(
    client: PubChemClient, cid: int, out_dir: str | None = None
) -> dict[str, Any]:
    """Write PubChem's 3D conformer of ``cid`` as SDF; when it has none, its 2D record."""
    compound = await get_compound(client, cid)
    cid = compound["cid"]
    notes: list[str] = []
    dimension = "3d"
    try:
        sdf = await client.text(f"{PUG}/compound/cid/{cid}/record/SDF", {"record_type": "3d"})
    except PubChemError as exc:
        if exc.reason != "not_found":
            raise
        dimension = "2d"
        sdf = await client.text(f"{PUG}/compound/cid/{cid}/record/SDF", {"record_type": "2d"})
        notes.append(
            "PubChem has no 3D conformer for this compound (common for salts, metals, "
            "mixtures and very large or flexible molecules). The file written is its 2D "
            "drawing: flat coordinates, not a geometry. To get 3D coordinates, embed one "
            "from the SMILES with ase.smiles_to_structure."
        )
    folder = output_dir(out_dir, f"cid{cid}-{_slug(compound['title'] or '')}")
    path = folder / f"{cid}_{dimension}.sdf"
    path.write_text(sdf, encoding="utf-8")
    structure = {
        "path": str(path),
        "format": "sdf",
        "formula": compound["formula"],
        "n_atoms": sdf_atom_count(sdf),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "dimension": dimension,
        "source": f"PubChem CID {cid} ({dimension.upper()} record)",
    }
    result = {
        "cid": cid,
        "compound": compound,
        "structure": structure,
        "has_3d": dimension == "3d",
        "fallback": None if dimension == "3d" else "2d_record",
        "notes": notes,
    }
    results_file = folder / f"{cid}_compound.json"
    results_file.write_text(json.dumps(result, indent=2), encoding="utf-8")
    result["results_file"] = str(results_file)
    return result
