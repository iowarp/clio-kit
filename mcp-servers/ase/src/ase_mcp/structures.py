"""Structure operations that need no calculator: inspect, set internal coordinates,
displace along a normal mode, take a frame of a trajectory."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from . import geometry
from .errors import AseError
from .records import (
    count_frames,
    existing_file,
    read_structure,
    structure_record,
    structures_dir,
    unique_file,
    write_xyz,
)


def stored_energy(atoms: Any) -> float | None:
    """The energy a file stores with a frame (extended XYZ info or ASE .traj results)."""
    from ase.calculators.calculator import PropertyNotImplementedError

    energy = atoms.info.get("energy_eV", atoms.info.get("energy"))
    if energy is None and atoms.calc is not None:
        try:
            energy = float(atoms.get_potential_energy())
        except (PropertyNotImplementedError, RuntimeError):
            energy = None
    return float(energy) if energy is not None else None


def inspect(structure: str, measure: list[list[int]] | None, frame: int = -1) -> dict[str, Any]:
    path = existing_file(structure)
    atoms = read_structure(path, frame)
    data = geometry.describe(atoms, measure)
    data["structure"] = structure_record(path, atoms)
    n_frames = count_frames(path) if path.suffix.lower() in (".xyz", ".extxyz", ".traj") else 1
    data["n_frames"] = n_frames
    data["frame"] = frame if n_frames > 1 else None
    energy = stored_energy(atoms)
    data["energy_eV"] = energy
    return data


def set_internal(
    structure: str,
    distance: dict | None,
    angle: dict | None,
    dihedral: dict | None,
    name: str | None,
    output_dir: str | None,
) -> dict[str, Any]:
    if not (distance or angle or dihedral):
        raise AseError(
            "invalid_input",
            "Give at least one of distance, angle or dihedral, each as {indices, value}.",
        )
    path = existing_file(structure)
    atoms = read_structure(path)
    changes = []
    for kind, spec, fn in (
        ("distance", distance, geometry.set_distance),
        ("angle", angle, geometry.set_angle),
        ("dihedral", dihedral, geometry.set_dihedral),
    ):
        if not spec:
            continue
        value = spec.get("value", spec.get("value_A" if kind == "distance" else "value_deg"))
        if value is None or "indices" not in spec:
            raise AseError(
                "invalid_input",
                f"{kind} needs indices and value, e.g. {{indices: [0, 1, 2, 3], value: 60}}.",
            )
        changes.append(fn(atoms, list(spec["indices"]), float(value)))
    measured = [geometry.measure(atoms, c["indices"]) for c in changes]
    for c, m in zip(changes, measured):
        c["after"] = m["value"]
        c["unit"] = m["unit"]
    stem = name or f"{path.stem}-" + "-".join(
        f"{c['kind'][0]}{round(c['target'], 1)}" for c in changes
    )
    out = unique_file(structures_dir(output_dir), stem, ".xyz")
    record = write_xyz(
        out,
        atoms,
        f"from {path.name}: "
        + "; ".join(f"{c['kind']} {c['indices']} = {c['after']}" for c in changes),
    )
    return {
        "source": structure_record(path, read_structure(path)),
        "changes": changes,
        "structure": record,
    }


def _mode_from_source(source: Path, mode_index: int | None) -> tuple[Any, np.ndarray, dict]:
    """(equilibrium atoms, Cartesian mode vector, info) from a result, npz or animation."""
    from ase import Atoms

    suffix = source.suffix.lower()
    if suffix == ".json":
        try:
            result = json.loads(source.read_text(encoding="utf-8"))
            npz = Path(result["vibrations"]["modes_file"]["path"])
        except (ValueError, KeyError, TypeError) as exc:
            raise AseError(
                "invalid_input",
                f"{source.name} is not a run result with vibrations; run run_ase "
                "with driver vib, thermo or ir first.",
            ) from exc
        return _mode_from_source(npz, mode_index)
    if suffix == ".npz":
        data = np.load(source)
        modes = data["modes"]
        if mode_index is None:
            raise AseError("invalid_input", "Give mode_index (see frequencies.csv).")
        if not 0 <= mode_index < len(modes):
            raise AseError("invalid_input", f"mode_index must be 0 to {len(modes) - 1}.")
        atoms = Atoms(numbers=data["numbers"], positions=data["positions"])
        freq = float(data["frequencies_cm1"][mode_index])
        return (
            atoms,
            modes[mode_index],
            {"mode": mode_index, "frequency_cm1": round(freq, 2), "imaginary": freq < 0},
        )
    # An animation (multi-frame XYZ): frame 0 is the equilibrium, frame n/4 the peak.
    from ase.io import read

    frames = read(str(source), index=":")
    if not isinstance(frames, list) or len(frames) < 4:
        raise AseError(
            "invalid_input",
            f"{source.name} is not a mode animation (needs "
            "the results JSON, normal_modes.npz or a mode_NNN.xyz).",
        )
    eq = frames[0]
    vec = frames[len(frames) // 4].get_positions() - eq.get_positions()
    freq = frames[0].info.get("frequency_cm1")
    return (
        eq,
        vec,
        {
            "mode": frames[0].info.get("mode", mode_index),
            "frequency_cm1": freq,
            "imaginary": bool(freq is not None and float(freq) < 0),
        },
    )


def displace(
    source: str, mode_index: int | None, amplitude: float, both: bool, output_dir: str | None
) -> dict[str, Any]:
    path = existing_file(source, "mode source")
    if amplitude <= 0:
        raise AseError(
            "invalid_input",
            "amplitude is the largest atomic displacement in Å and must be positive.",
        )
    atoms, vec, info = _mode_from_source(path, mode_index)
    peak = float(np.linalg.norm(vec, axis=1).max())
    if peak == 0:
        raise AseError("invalid_input", "That mode has no displacement.")
    unit = vec / peak
    folder = structures_dir(output_dir)
    mode = info.get("mode")
    out = []
    for sign in (1.0, -1.0) if both else (1.0,):
        displaced = atoms.copy()
        displaced.set_positions(atoms.get_positions() + sign * amplitude * unit)
        tag = "plus" if sign > 0 else "minus"
        target = unique_file(folder, f"mode{mode}-{tag}{amplitude:g}A", ".xyz")
        rec = write_xyz(
            target,
            displaced,
            f"displaced {sign * amplitude:+g} Å along mode "
            f"{mode} ({info.get('frequency_cm1')} cm-1)",
        )
        out.append({"direction": tag, "structure": rec})
    return {
        "source": str(path),
        **info,
        "amplitude_A": amplitude,
        "amplitude_meaning": "largest single-atom displacement",
        "structures": out,
    }


def frame(trajectory: str, index: int, output_dir: str | None) -> dict[str, Any]:
    path = existing_file(trajectory, "trajectory")
    n = count_frames(path)
    if not -n <= index < n:
        raise AseError(
            "invalid_input", f"{path.name} has {n} frames: use 0 to {n - 1}, or -1 for the last."
        )
    atoms = read_structure(path, index)
    k = index % n
    out = unique_file(structures_dir(output_dir), f"{path.stem}-frame{k}", ".xyz")
    energy = stored_energy(atoms)
    rec = write_xyz(out, atoms, f"frame {k} of {path.name}")
    return {
        "trajectory": str(path),
        "n_frames": n,
        "frame": k,
        "energy_eV": energy,
        "fmax_eV_per_A": atoms.info.get("fmax_eV_per_A"),
        "structure": rec,
    }
