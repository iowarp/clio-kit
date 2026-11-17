"""The agent-facing text of each tool: lean, in words; full records stay structured."""

from __future__ import annotations

from typing import Any


def _stereo(record: dict[str, Any]) -> str:
    s = record["stereo"]
    parts = []
    if s["atom_centres"]:
        parts.append(
            f"{s['atom_centres']} stereo centre(s), {s['undefined_atom_centres']} undefined"
        )
    if s["bond_centres"]:
        parts.append(
            f"{s['bond_centres']} stereo double bond(s), {s['undefined_bond_centres']} undefined"
        )
    return "; ".join(parts) if parts else "no stereo centres"


def _line(record: dict[str, Any]) -> str:
    mw = record["molecular_weight_g_mol"]
    return (
        f"CID {record['cid']} {record['title']} ({record['iupac_name']}): "
        f"{record['formula']}, {mw:g} g/mol, SMILES {record['isomeric_smiles']}, "
        f"{_stereo(record)}"
        + (f", {record['components']} components" if (record.get("components") or 1) > 1 else "")
        + "."
    )


def resolve_text(data: dict[str, Any]) -> str:
    if data["status"] == "resolved":
        record = data["compound"]
        lines = [f"'{data['query']}' is PubChem {_line(record)}"]
        if record["stereo"]["undefined_atom_centres"] or record["stereo"]["undefined_bond_centres"]:
            lines.append(
                "Its stereochemistry is not defined in this record; if a specific "
                "stereoisomer is meant, ask which one."
            )
        lines.append(f"InChIKey {record['inchikey']}. {record['url']}")
        return "\n".join(lines)
    count = len(data["candidates"])
    lines = [f"{data['reason']} {count} candidate{'s' if count != 1 else ''}:"]
    for record in data["candidates"]:
        lines.append(f"- {_line(record)}")
    lines.append(data["next_step"])
    return "\n".join(lines)


def compound_text(record: dict[str, Any]) -> str:
    lines = [f"PubChem {_line(record)}", f"InChIKey {record['inchikey']}. {record['url']}"]
    if record.get("synonyms"):
        lines.append("Also called: " + "; ".join(record["synonyms"][:8]) + ".")
    return "\n".join(lines)


def structure_text(data: dict[str, Any]) -> str:
    s, c = data["structure"], data["compound"]
    if data["has_3d"]:
        head = (
            f"Wrote PubChem's 3D conformer of CID {data['cid']} {c['title']} "
            f"({s['formula']}, {s['n_atoms']} atoms including hydrogens) to {s['path']}."
        )
    else:
        head = (
            f"PubChem has no 3D conformer for CID {data['cid']} {c['title']}. Wrote its 2D "
            f"record instead ({s['formula']}, {s['n_atoms']} atoms, flat coordinates) to "
            f"{s['path']}."
        )
    lines = [head, *data["notes"], f"Compound record: {data['results_file']}."]
    return "\n".join(lines)
