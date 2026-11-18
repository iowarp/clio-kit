"""Bonds, fragments and internal coordinates of a molecule, with ASE.

Bonds come from covalent radii (``ase.neighborlist.natural_cutoffs``, scaled by 1.15
by default). Setting a distance, angle or dihedral moves the smaller of the two
fragments the bond splits the molecule into, as a rigid body, so the rest of the
molecule keeps its shape; a bond inside a ring cannot be changed this way and is
refused (``ring_bond``). Atom indices are 0-based, in file order.
"""

from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .errors import AseError

BOND_SCALE = 1.15


def bonds(atoms: Any, scale: float = BOND_SCALE) -> list[tuple[int, int, float]]:
    """Covalent bonds as ``(i, j, length_A)`` with ``i < j``."""
    from ase.neighborlist import NeighborList, natural_cutoffs

    if len(atoms) < 2:
        return []
    cutoffs = natural_cutoffs(atoms, mult=scale)
    nl = NeighborList(cutoffs, self_interaction=False, bothways=False, skin=0.0)
    nl.update(atoms)
    out: list[tuple[int, int, float]] = []
    for i in range(len(atoms)):
        neighbors, offsets = nl.get_neighbors(i)
        for j, offset in zip(neighbors, offsets):
            if any(offset):
                continue  # periodic image: not a molecular bond
            a, b = (i, int(j)) if i < j else (int(j), i)
            out.append((a, b, float(atoms.get_distance(a, b))))
    return sorted(set(out))


def adjacency(n: int, bond_list: list[tuple[int, int, float]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for i, j, _ in bond_list:
        adj[i].add(j)
        adj[j].add(i)
    return adj


def component(adj: list[set[int]], start: int, cut: tuple[int, int] | None = None) -> set[int]:
    """Atoms reachable from ``start`` without crossing the ``cut`` bond."""
    seen = {start}
    stack = [start]
    while stack:
        k = stack.pop()
        for m in adj[k]:
            if cut and {k, m} == set(cut):
                continue
            if m not in seen:
                seen.add(m)
                stack.append(m)
    return seen


def fragments(atoms: Any) -> list[list[int]]:
    """Connected fragments (molecules) as lists of atom indices."""
    adj = adjacency(len(atoms), bonds(atoms))
    left = set(range(len(atoms)))
    out = []
    while left:
        comp = component(adj, min(left))
        out.append(sorted(comp))
        left -= comp
    return out


def check_indices(atoms: Any, indices: list[int], n: int, what: str) -> list[int]:
    if len(indices) != n:
        raise AseError(
            "invalid_input",
            f"A {what} needs {n} atom indices, got {len(indices)}.",
            indices=indices,
        )
    if len(set(indices)) != n:
        raise AseError("invalid_input", f"The {what} indices must differ: {indices}.")
    for i in indices:
        if not 0 <= int(i) < len(atoms):
            raise AseError(
                "invalid_input",
                f"Atom index {i} is out of range: the structure has {len(atoms)} atoms "
                f"(indices 0 to {len(atoms) - 1}).",
            )
    return [int(i) for i in indices]


def measure(atoms: Any, indices: list[int]) -> dict[str, Any]:
    """A distance (2 indices), angle (3) or dihedral (4) of the structure."""
    n = len(indices)
    if n == 2:
        i, j = check_indices(atoms, indices, 2, "distance")
        return {
            "indices": [i, j],
            "kind": "distance",
            "value": round(float(atoms.get_distance(i, j)), 5),
            "unit": "Å",
        }
    if n == 3:
        i, j, k = check_indices(atoms, indices, 3, "angle")
        return {
            "indices": [i, j, k],
            "kind": "angle",
            "value": round(float(atoms.get_angle(i, j, k)), 4),
            "unit": "degree",
        }
    if n == 4:
        a, b, c, d = check_indices(atoms, indices, 4, "dihedral")
        return {
            "indices": [a, b, c, d],
            "kind": "dihedral",
            "value": round(signed_dihedral(float(atoms.get_dihedral(a, b, c, d))), 4),
            "unit": "degree",
        }
    raise AseError(
        "invalid_input",
        f"Measure 2 (distance), 3 (angle) or 4 (dihedral) atom indices, got {n}.",
    )


def signed_dihedral(value: float) -> float:
    """Map ASE's 0–360 degree dihedral to −180…180."""
    v = value % 360.0
    return v - 360.0 if v > 180.0 else v


def _rotate(
    positions: np.ndarray, idx: list[int], origin: np.ndarray, axis: np.ndarray, angle_rad: float
) -> None:
    axis = axis / np.linalg.norm(axis)
    p = positions[idx] - origin
    cos, sin = np.cos(angle_rad), np.sin(angle_rad)
    rotated = p * cos + np.cross(axis, p) * sin + np.outer(p @ axis, axis) * (1.0 - cos)
    positions[idx] = rotated + origin


def _split(atoms: Any, i: int, j: int, what: str) -> tuple[set[int], set[int]]:
    """The two fragments on either side of bond i–j (refuses a ring bond)."""
    adj = adjacency(len(atoms), bonds(atoms))
    if j not in adj[i] and j in component(adj, i):
        raise AseError(
            "invalid_input",
            f"Atoms {i} and {j} are not bonded to each other but belong to the same "
            f"molecule, so the {what} has no single bond to turn or stretch. The middle "
            "atoms of a dihedral, and the atoms of a distance, must be bonded (or in "
            "separate fragments).",
            indices=[i, j],
        )
    side_i = component(adj, i, cut=(i, j))
    if j in side_i:
        raise AseError(
            "ring_bond",
            f"Atoms {i} and {j} are connected through a ring (or another path), so the "
            f"{what} cannot be changed by moving one side rigidly. Choose an acyclic bond, "
            "or edit the structure another way.",
            indices=[i, j],
        )
    side_j = component(adj, j, cut=(i, j))
    return side_i, side_j


def set_distance(atoms: Any, indices: list[int], value: float) -> dict[str, Any]:
    i, j = check_indices(atoms, indices, 2, "distance")
    if value <= 0:
        raise AseError("invalid_input", f"A distance must be positive, got {value} Å.")
    side_i, side_j = _split(atoms, i, j, "distance")
    pos = atoms.get_positions()
    vec = pos[j] - pos[i]
    current = float(np.linalg.norm(vec))
    shift = (value - current) * vec / current
    if len(side_j) <= len(side_i):
        moved = sorted(side_j)
        pos[moved] += shift
    else:
        moved = sorted(side_i)
        pos[moved] -= shift
    atoms.set_positions(pos)
    return {
        "kind": "distance",
        "indices": [i, j],
        "before": round(current, 5),
        "target": value,
        "moved_atoms": moved,
    }


def set_angle(atoms: Any, indices: list[int], value: float) -> dict[str, Any]:
    a, b, c = check_indices(atoms, indices, 3, "angle")
    if not 0.0 < value < 180.0:
        raise AseError("invalid_input", f"An angle must be between 0 and 180°, got {value}.")
    side_a, _ = _split(atoms, a, b, "angle")
    side_c, _ = _split(atoms, c, b, "angle")
    pos = atoms.get_positions()
    current = float(atoms.get_angle(a, b, c))
    axis = np.cross(pos[a] - pos[b], pos[c] - pos[b])
    if np.linalg.norm(axis) < 1e-8:  # linear: any perpendicular axis
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(ref, pos[c] - pos[b])) > 0.9 * np.linalg.norm(pos[c] - pos[b]):
            ref = np.array([0.0, 1.0, 0.0])
        axis = np.cross(pos[c] - pos[b], ref)
    delta = np.radians(value - current)
    if len(side_c) <= len(side_a):
        moved = sorted(side_c)
        _rotate(pos, moved, pos[b].copy(), axis, delta)
    else:
        moved = sorted(side_a)
        _rotate(pos, moved, pos[b].copy(), axis, -delta)
    atoms.set_positions(pos)
    return {
        "kind": "angle",
        "indices": [a, b, c],
        "before": round(current, 4),
        "target": value,
        "moved_atoms": moved,
    }


def set_dihedral(atoms: Any, indices: list[int], value: float) -> dict[str, Any]:
    a, b, c, d = check_indices(atoms, indices, 4, "dihedral")
    side_b, side_c = _split(atoms, b, c, "dihedral")
    pos = atoms.get_positions()
    current = signed_dihedral(float(atoms.get_dihedral(a, b, c, d)))
    delta = signed_dihedral(value - current)
    axis = pos[c] - pos[b]
    # ASE's dihedral grows when the c side turns right-handed about b->c.
    if len(side_c) <= len(side_b):
        moved = sorted(side_c)
        _rotate(pos, moved, pos[b].copy(), axis, np.radians(delta))
    else:
        moved = sorted(side_b)
        _rotate(pos, moved, pos[b].copy(), axis, -np.radians(delta))
    atoms.set_positions(pos)
    return {
        "kind": "dihedral",
        "indices": [a, b, c, d],
        "before": round(current, 4),
        "target": signed_dihedral(value),
        "moved_atoms": moved,
    }


def describe(
    atoms: Any, measure_sets: list[list[int]] | None = None, max_bonds: int = 300
) -> dict[str, Any]:
    """Formula, composition, bonds, fragments and requested measurements."""
    bond_list = bonds(atoms)
    symbols = atoms.get_chemical_symbols()
    frags = fragments(atoms)
    pos = atoms.get_positions()
    extent = (pos.max(axis=0) - pos.min(axis=0)).tolist() if len(atoms) else [0, 0, 0]
    data: dict[str, Any] = {
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "composition": dict(sorted(Counter(symbols).items())),
        "n_bonds": len(bond_list),
        "bonds": [
            {"i": i, "j": j, "atoms": f"{symbols[i]}{i}-{symbols[j]}{j}", "length_A": round(d, 4)}
            for i, j, d in bond_list[:max_bonds]
        ],
        "bonds_truncated": len(bond_list) > max_bonds,
        "n_fragments": len(frags),
        "fragments": frags if len(frags) > 1 else None,
        "extent_A": [round(x, 3) for x in extent],
        "periodic": bool(any(atoms.pbc)),
        "bond_rule": f"covalent radii × {BOND_SCALE}",
    }
    if measure_sets:
        data["measurements"] = [measure(atoms, list(m)) for m in measure_sets]
    return data
