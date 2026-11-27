"""Where outputs go, how structures are read, and the record every written file gets.

Structures travel between tools by reference: a tool reads a structure file and writes
new files, and returns a ``structure`` record ``{path, format, formula, n_atoms,
sha256}`` for each structure it writes. Outputs land in a readable folder
``chemistry/ase/<run-name>/`` under ``output_dir`` when given, else
``$CLIO_KIT_ARTIFACTS`` (CLIO sets it to the workspace), else the current directory.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Any

from .errors import AseError

ARTIFACTS_ENV = "CLIO_KIT_ARTIFACTS"
SUBFOLDER = ("chemistry", "ase")

# Extensions ASE reads and the format name the record reports.
FORMATS = {
    ".xyz": "xyz",
    ".extxyz": "extxyz",
    ".sdf": "sdf",
    ".mol": "sdf",
    ".pdb": "pdb",
    ".cif": "cif",
    ".traj": "traj",
    ".vasp": "vasp",
    ".poscar": "vasp",
    ".json": "json",
    ".gen": "gen",
    ".in": "aims",
}


def output_root(output_dir: str | None) -> Path:
    """The ``chemistry/ase`` folder outputs go into."""
    if output_dir:
        base = Path(output_dir).expanduser()
    elif os.environ.get(ARTIFACTS_ENV):
        base = Path(os.environ[ARTIFACTS_ENV]).expanduser()
    else:
        base = Path.cwd()
    return base.resolve().joinpath(*SUBFOLDER)


def slug(text: str, fallback: str = "run") -> str:
    """A short file-system-safe name from free text."""
    cleaned = re.sub(r"[^A-Za-z0-9._+-]+", "-", text.strip()).strip("-.")
    return cleaned[:60] or fallback


def new_run_dir(output_dir: str | None, run_name: str) -> Path:
    """Create a new run folder; an existing name gets ``-2``, ``-3`` … appended."""
    root = output_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    name = slug(run_name)
    candidate = root / name
    n = 2
    while candidate.exists():
        candidate = root / f"{name}-{n}"
        n += 1
    candidate.mkdir(parents=True)
    return candidate


def unique_file(folder: Path, stem: str, suffix: str) -> Path:
    """A new file path in ``folder`` that does not overwrite an existing file."""
    folder.mkdir(parents=True, exist_ok=True)
    candidate = folder / f"{slug(stem)}{suffix}"
    n = 2
    while candidate.exists():
        candidate = folder / f"{slug(stem)}-{n}{suffix}"
        n += 1
    return candidate


def structures_dir(output_dir: str | None) -> Path:
    """The folder for structures that are not part of a calculation run."""
    folder = output_root(output_dir) / "structures"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def existing_file(path: str, what: str = "structure file") -> Path:
    """Resolve a path that must exist, or raise ``not_found``."""
    p = Path(path).expanduser()
    if not p.is_absolute() and not p.exists() and os.environ.get(ARTIFACTS_ENV):
        alt = Path(os.environ[ARTIFACTS_ENV]).expanduser() / p
        if alt.exists():
            p = alt
    if not p.is_file():
        raise AseError(
            "not_found",
            f"The {what} {path!r} does not exist. Give the path of an existing file "
            "(absolute, or relative to the workspace).",
            path=str(path),
        )
    return p.resolve()


def file_format(path: Path) -> str:
    return FORMATS.get(path.suffix.lower(), path.suffix.lower().lstrip(".") or "unknown")


def read_structure(path: str | Path, index: int | str = -1) -> Any:
    """Read one frame (default the last) of a structure file with ASE."""
    from ase.io import read

    p = existing_file(str(path))
    try:
        kwargs: dict[str, Any] = {"index": index}
        if p.suffix.lower() == ".mol":
            kwargs["format"] = "sdf"
        return read(str(p), **kwargs)
    except AseError:
        raise
    except Exception as exc:  # ASE raises many types for unreadable files
        raise AseError(
            "unreadable_structure",
            f"ASE could not read {p.name}: {exc}. Supported: XYZ, extended XYZ, SDF/MOL, "
            "PDB, CIF, ASE .traj, POSCAR.",
            path=str(p),
        ) from exc


def count_frames(path: Path) -> int:
    """Number of frames in a structure file (1 for single-frame formats)."""
    frames = read_structure(path, ":")
    return len(frames) if isinstance(frames, list) else 1


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def structure_record(path: Path, atoms: Any = None) -> dict[str, Any]:
    """The record of a structure file: path, format, formula, atom count and hash."""
    if atoms is None:
        atoms = read_structure(path)
    return {
        "path": str(path),
        "format": file_format(path),
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "sha256": sha256(path),
    }


def file_record(path: Path, kind: str, **extra: Any) -> dict[str, Any]:
    """The record of a non-structure output file (CSV, JSON, trajectory)."""
    return {"path": str(path), "kind": kind, **extra}


def write_xyz(path: Path, atoms: Any, comment: str = "") -> dict[str, Any]:
    """Write an extended XYZ file (plain-XYZ readable) and return its record."""
    from ase.io import write

    clean = atoms.copy()
    clean.calc = None
    clean.constraints = []
    if comment:
        clean.info = {
            "comment": comment,
            **{k: v for k, v in clean.info.items() if isinstance(v, (int, float, str))},
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    write(str(path), clean, format="extxyz")
    return structure_record(path, clean)
