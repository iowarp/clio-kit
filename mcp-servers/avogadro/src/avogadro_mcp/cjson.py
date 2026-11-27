"""What the person sent from Avogadro: CJSON to XYZ, formula and selected atoms."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from ase.data import chemical_symbols


def read(doc: dict[str, Any]) -> dict[str, Any]:
    atoms = doc.get("atoms") or {}
    numbers = list((atoms.get("elements") or {}).get("number") or [])
    coords = list((atoms.get("coords") or {}).get("3d") or [])
    if len(coords) != 3 * len(numbers):
        raise ValueError("the CJSON has no 3D coordinates for every atom")
    symbols = [chemical_symbols[int(z)] for z in numbers]
    positions = [coords[i : i + 3] for i in range(0, len(coords), 3)]
    selected = [i for i, s in enumerate(atoms.get("selected") or []) if s]
    counts = Counter(symbols)
    order = (["C", "H"] if "C" in counts else []) + sorted(
        k for k in counts if k not in ("C", "H") or "C" not in counts
    )
    formula = "".join(f"{e}{counts[e] if counts[e] > 1 else ''}" for e in order if e in counts)
    unit_cell = doc.get("unitCell")
    return {
        "symbols": symbols,
        "positions": positions,
        "selected": selected,
        "formula": formula,
        "n_atoms": len(symbols),
        "unit_cell": unit_cell,
        "name": doc.get("name") or "",
    }


def write_xyz(path: Path, mol: dict[str, Any], comment: str) -> None:
    lines = [str(mol["n_atoms"]), comment.replace("\n", " ")]
    for s, (x, y, z) in zip(mol["symbols"], mol["positions"], strict=True):
        lines.append(f"{s} {x:.6f} {y:.6f} {z:.6f}")
    path.write_text("\n".join(lines) + "\n")
