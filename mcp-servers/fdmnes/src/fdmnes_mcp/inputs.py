"""FDMNES input files from a structure file.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), ``tools/xanes_core.py``
``write_fdmnes_input``. Changed: the absorber is always named by the caller (ChemGraph
took the heaviest element); a molecule is written in Cartesian Å (``Molecule`` with
a = b = c = 1; ChemGraph scaled the Cartesian positions by a box length, which FDMNES
reads as fractions of that box); crystal fractions are written with 10 digits, which
FDMNES needs to find the symmetry; the edge, method, energy range and SCF are inputs.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .errors import FdmnesError

EDGES = (
    "K",
    "L1",
    "L2",
    "L3",
    "L23",
    "M1",
    "M2",
    "M3",
    "M23",
    "M4",
    "M5",
    "M45",
    "N1",
    "N2",
    "N3",
    "N23",
    "N4",
    "N5",
    "N45",
)
DEFAULT_RANGE = (-30.0, 1.0, -10.0, 0.2, 20.0, 0.5, 80.0)


@dataclass
class XanesSpec:
    """What one FDMNES run computes; every field is recorded in result.json."""

    absorber: str | int | list[int]
    edge: str = "K"
    method: str = "green"
    radius: float = 6.0
    scf: bool = True
    quadrupole: bool = False
    magnetism: bool = False
    energy_range: tuple[float, ...] = DEFAULT_RANGE
    extra_keywords: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "absorber": self.absorber,
            "edge": self.edge,
            "method": self.method,
            "radius_A": self.radius,
            "scf": self.scf,
            "quadrupole": self.quadrupole,
            "magnetism": self.magnetism,
            "energy_range_eV": list(self.energy_range),
            "extra_keywords": list(self.extra_keywords),
        }


def read_structure(path: Path) -> Any:
    from ase.io import read

    try:
        kwargs: dict[str, Any] = {"index": -1}
        if path.suffix.lower() == ".mol":
            kwargs["format"] = "sdf"
        return read(str(path), **kwargs)
    except Exception as exc:  # ASE raises many types for unreadable files
        raise FdmnesError(
            "unreadable_structure",
            f"ASE could not read {path.name}: {exc}. Give a CIF (crystals) or an XYZ, "
            "SDF/MOL, PDB or POSCAR file.",
            path=str(path),
        ) from exc


def is_crystal(atoms: Any) -> bool:
    return bool(all(atoms.pbc)) and float(atoms.cell.volume) > 1e-6


def absorber_lines(atoms: Any, absorber: str | int | list[int]) -> tuple[list[str], dict[str, Any]]:
    """``Z_absorber`` for an element (all its sites), ``Absorber`` for chosen sites."""
    from ase.data import atomic_numbers, chemical_symbols

    numbers = [int(z) for z in atoms.get_atomic_numbers()]
    if isinstance(absorber, list):
        if not absorber:
            raise FdmnesError("invalid_input", "absorber: give an element or site indices.")
        bad = [i for i in absorber if not 0 <= int(i) < len(numbers)]
        if bad:
            raise FdmnesError(
                "invalid_input",
                f"absorber sites {bad} are outside 0..{len(numbers) - 1} (0-based, file order).",
            )
        elements = sorted({chemical_symbols[numbers[i]] for i in absorber})
        lines = ["Absorber", " ".join(str(int(i) + 1) for i in absorber)]
        return lines, {"sites": [int(i) for i in absorber], "elements": elements}
    if isinstance(absorber, int) or str(absorber).isdigit():
        z = int(absorber)
    else:
        symbol = str(absorber).strip().capitalize()
        if symbol not in atomic_numbers:
            raise FdmnesError("invalid_input", f"absorber {absorber!r} is not an element.")
        z = atomic_numbers[symbol]
    count = numbers.count(z)
    if count == 0:
        present = sorted({chemical_symbols[n] for n in numbers})
        raise FdmnesError(
            "invalid_input",
            f"The structure has no {chemical_symbols[z]} atom to absorb; it has "
            f"{', '.join(present)}.",
        )
    return ["Z_absorber", str(z)], {
        "element": chemical_symbols[z],
        "Z": z,
        "sites_in_file": count,
    }


def check_spec(spec: XanesSpec) -> None:
    if spec.edge not in EDGES:
        raise FdmnesError("invalid_input", f"edge {spec.edge!r}: one of {', '.join(EDGES)}.")
    if spec.method not in ("green", "fdm"):
        raise FdmnesError("invalid_input", "method: 'green' (multiple scattering) or 'fdm'.")
    if not 1.0 <= spec.radius <= 12.0:
        raise FdmnesError("invalid_input", "radius: a cluster radius in Å between 1 and 12.")
    r = spec.energy_range
    if len(r) < 3 or len(r) % 2 == 0:
        raise FdmnesError(
            "invalid_input",
            "energy_range: Emin, step, Emax (or Emin, step, E1, step, … Emax) in eV.",
        )
    edges, steps = r[0::2], r[1::2]
    if any(b <= a for a, b in itertools.pairwise(edges)) or any(s <= 0 for s in steps):
        raise FdmnesError("invalid_input", "energy_range: increasing energies, positive steps.")
    for kw in spec.extra_keywords:
        if not kw.strip() or kw.strip().lower() in {"filout", "end"}:
            raise FdmnesError("invalid_input", f"extra keyword {kw!r} is not allowed.")


def write_inputs(atoms: Any, spec: XanesSpec, run_dir: Path, stem: str) -> dict[str, Any]:
    """Write ``fdmfile.txt`` and ``<stem>_inp.txt``; return what the input describes."""
    from ase.data import chemical_symbols

    check_spec(spec)
    abs_lines, absorber = absorber_lines(atoms, spec.absorber)
    crystal = is_crystal(atoms)
    lines = ["Filout", stem, "", "Range", " ".join(f"{v:g}" for v in spec.energy_range), ""]
    lines += ["Radius", f"{spec.radius:g}", ""]
    lines += abs_lines + [""]
    if spec.edge != "K":
        lines += ["Edge", spec.edge, ""]
    if spec.method == "green":
        lines += ["Green", ""]
    if spec.scf:
        lines += ["SCF", ""]
    if spec.quadrupole:
        lines += ["Quadrupole", ""]
    if spec.magnetism:
        lines += ["Magnetism", ""]
    for kw in spec.extra_keywords:
        lines += [kw.rstrip(), ""]
    numbers = atoms.get_atomic_numbers()
    if crystal:
        a, b, c, al, be, ga = atoms.cell.cellpar()
        lines += ["Crystal", f"{a:.10f} {b:.10f} {c:.10f} {al:.10f} {be:.10f} {ga:.10f}"]
        frac = atoms.get_scaled_positions(wrap=True)
        for z, p in zip(numbers, frac):
            lines.append(f"{int(z):3d} {p[0]:.10f} {p[1]:.10f} {p[2]:.10f}")
    else:
        lines += ["Molecule", "1.0 1.0 1.0 90.0 90.0 90.0"]
        for z, p in zip(numbers, atoms.get_positions()):
            lines.append(f"{int(z):3d} {p[0]:.8f} {p[1]:.8f} {p[2]:.8f}")
    lines += ["", "Convolution", "", "End", ""]
    inp = run_dir / f"{stem}_inp.txt"
    inp.write_text("\n".join(lines))
    (run_dir / "fdmfile.txt").write_text(f"1\n{inp.name}\n")
    return {
        "input_file": str(inp),
        "kind": "crystal" if crystal else "molecule",
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "elements": sorted({chemical_symbols[int(z)] for z in numbers}),
        "absorber": absorber,
    }
