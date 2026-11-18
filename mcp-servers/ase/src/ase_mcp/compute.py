"""One ASE calculation, run inside the worker process: energy, opt, vib, thermo or ir.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/ase_core.py
(``run_ase_core``). Differences: structures are files, outputs go to one run folder in
one JSON format (``ase-mcp/run@1``) with energies in eV and kJ/mol, the optimization is
written as an extended-XYZ trajectory plus a per-step CSV, every vibrational mode is
written as a short animation, constraints are supported, and translations/rotations are
projected out of the Hessian (:mod:`.vibrations`).
"""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np

from . import calculators, geometry, vibrations
from .errors import AseError
from .records import file_record, read_structure, structure_record, write_xyz
from .schemas import RunInput

SCHEMA = "ase-mcp/run@1"
EV = vibrations.EV_TO_KJ_MOL


def energy_record(value: float) -> dict[str, float]:
    return {"eV": round(float(value), 8), "kJ_mol": round(float(value) * EV, 4)}


class Progress:
    """Writes ``progress.json`` in the run folder for the server to relay."""

    def __init__(self, run_dir: Path) -> None:
        self.path = run_dir / "progress.json"
        self._last = 0.0

    def __call__(
        self, phase: str, done: int, total: int | None, message: str, force: bool = False
    ) -> None:
        now = time.monotonic()
        if not force and now - self._last < 0.25:
            return
        self._last = now
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(
            json.dumps({"phase": phase, "done": done, "total": total, "message": message})
        )
        os.replace(tmp, self.path)


def _constraints(atoms: Any, specs: list[Any], notes: list[str]) -> tuple[list[Any], list[dict]]:
    from ase.constraints import FixAtoms, FixInternals

    out: list[Any] = []
    recorded: list[dict] = []
    bonds, angles, dihedrals = [], [], []
    for c in specs:
        if c.type == "fix_atoms":
            idx = geometry.check_indices(atoms, c.indices, len(c.indices), "fix_atoms")
            out.append(FixAtoms(indices=idx))
            recorded.append({"type": "fix_atoms", "indices": idx})
            continue
        if c.type == "bond":
            if c.value_A is not None:
                geometry.set_distance(atoms, c.indices, c.value_A)
            m = geometry.measure(atoms, c.indices)
            bonds.append([m["value"], m["indices"]])
            recorded.append({"type": "bond", "indices": m["indices"], "value_A": m["value"]})
        elif c.type == "angle":
            if c.value_deg is not None:
                geometry.set_angle(atoms, c.indices, c.value_deg)
            m = geometry.measure(atoms, c.indices)
            angles.append([m["value"], m["indices"]])
            recorded.append({"type": "angle", "indices": m["indices"], "value_deg": m["value"]})
        elif c.type == "dihedral":
            if c.value_deg is not None:
                geometry.set_dihedral(atoms, c.indices, c.value_deg)
            m = geometry.measure(atoms, c.indices)
            dihedrals.append([m["value"], m["indices"]])
            recorded.append({"type": "dihedral", "indices": m["indices"], "value_deg": m["value"]})
    if bonds or angles or dihedrals:
        out.append(
            FixInternals(
                bonds=bonds or None, angles_deg=angles or None, dihedrals_deg=dihedrals or None
            )
        )
    if any(c.value_A is not None or c.value_deg is not None for c in specs):
        notes.append(
            "Constrained internal coordinates with a value were set to it (moving "
            "the smaller fragment) before the optimization started."
        )
    return out, recorded


def _optimize(atoms: Any, job: RunInput, run_dir: Path, progress: Progress) -> dict[str, Any]:
    from ase.io import write
    from ase.optimize import BFGS, FIRE, LBFGS, GPMin, MDMin

    optimizer_cls = {"bfgs": BFGS, "lbfgs": LBFGS, "fire": FIRE, "gpmin": GPMin, "mdmin": MDMin}[
        job.optimizer
    ]
    traj_path = run_dir / "opt_trajectory.xyz"
    csv_path = run_dir / "opt_steps.csv"
    rows: list[tuple[int, float, float]] = []
    traj_path.unlink(missing_ok=True)
    csv = csv_path.open("w", encoding="utf-8")
    csv.write("step,energy_eV,fmax_eV_per_A\n")

    def observe() -> None:
        energy = float(atoms.get_potential_energy())
        forces = atoms.get_forces()
        fmax = float(np.sqrt((forces**2).sum(axis=1).max())) if len(atoms) else 0.0
        step = len(rows)
        rows.append((step, energy, fmax))
        csv.write(f"{step},{energy:.8f},{fmax:.6f}\n")
        csv.flush()
        frame = atoms.copy()
        frame.calc = None
        frame.constraints = []
        frame.info = {"step": step, "energy_eV": round(energy, 8), "fmax_eV_per_A": round(fmax, 6)}
        write(str(traj_path), frame, format="extxyz", append=True)
        progress(
            "optimizing",
            step,
            job.steps,
            f"optimization step {step}: E = {energy:.5f} eV, fmax = {fmax:.4f} eV/Å",
        )

    try:
        opt = optimizer_cls(atoms, logfile=str(run_dir / "opt.log"))
        opt.attach(observe, interval=1)
        converged = bool(opt.run(fmax=job.fmax, steps=job.steps))
        nsteps = int(opt.nsteps)
    finally:
        csv.close()
    return {
        "optimizer": job.optimizer,
        "fmax_target_eV_per_A": job.fmax,
        "max_steps": job.steps,
        "steps": nsteps,
        "converged": converged,
        "final_fmax_eV_per_A": round(rows[-1][2], 6) if rows else None,
        "energy_change": energy_record(rows[-1][1] - rows[0][1]) if rows else None,
        "trajectory": file_record(traj_path, "trajectory", format="extxyz", n_frames=len(rows)),
        "steps_csv": file_record(csv_path, "csv", columns=["step", "energy_eV", "fmax_eV_per_A"]),
    }


def require_dipole(atoms: Any, kind: str) -> None:
    """Fail before any work when an IR run's calculator gives no dipole moment."""
    from ase.calculators.calculator import PropertyNotImplementedError

    try:
        dip = atoms.get_dipole_moment()
        ok = dip is not None and np.asarray(dip).shape == (3,)
    except (PropertyNotImplementedError, NotImplementedError):
        ok = False
    if not ok:
        raise AseError(
            "dipole_unavailable",
            f"The {kind} calculator gives no dipole moment, so IR intensities cannot be "
            "computed. Use xtb, nwchem or orca for driver 'ir', or driver 'vib' for "
            "frequencies only.",
        )


def _vibrations(
    atoms: Any,
    job: RunInput,
    run_dir: Path,
    potential_energy: float,
    progress: Progress,
    notes: list[str],
) -> dict[str, Any]:
    want_ir = job.driver == "ir"
    if atoms.constraints:
        notes.append(
            "Constraints were used for the optimization only; the frequencies are "
            "from the full, unconstrained Hessian."
        )
        atoms.set_constraint()

    def report(done: int, total: int) -> None:
        progress("vibrations", done, total, f"finite differences {done}/{total}")

    fd = vibrations.hessian(atoms, dipoles=want_ir, progress=report)
    nm = vibrations.normal_modes(atoms, fd["hessian"])
    freqs = nm["frequencies_cm1"]
    imaginary = [i for i, lam in enumerate(nm["eigenvalues"]) if lam < 0]

    intensities = None
    if want_ir:
        intensities = vibrations.ir_intensities(atoms, nm["modes_mw"], fd["dipole_derivatives"])

    from ase.io import write

    modes_dir = run_dir / "modes"
    modes_dir.mkdir(exist_ok=True)
    rows = []
    for i, f in enumerate(freqs):
        anim = modes_dir / f"mode_{i:03d}.xyz"
        frames = vibrations.mode_animation(atoms, nm["modes_cart"][i])
        for k, frame in enumerate(frames):
            frame.info = {"mode": i, "frame": k, "frequency_cm1": round(float(f), 2)}
        write(str(anim), frames, format="extxyz")
        label = f"{abs(f):.2f}i" if i in imaginary else f"{f:.2f}"
        row: dict[str, Any] = {
            "mode": i,
            "frequency_cm1": round(float(f), 4),
            "imaginary": i in imaginary,
            "label": label,
            "energy_meV": round(float(abs(nm["energies"][i])) * 1000, 4),
            "animation": str(anim),
        }
        if intensities is not None:
            row["ir_intensity_D2_A2_amu"] = round(float(intensities[i]), 6)
            row["ir_intensity_km_mol"] = round(
                float(intensities[i]) * vibrations.KM_MOL_PER_D2_A2_AMU, 4
            )
        rows.append(row)

    freq_csv = run_dir / "frequencies.csv"
    with freq_csv.open("w", encoding="utf-8") as fh:
        cols = ["mode", "frequency_cm1", "imaginary", "label", "energy_meV"]
        if intensities is not None:
            cols += ["ir_intensity_km_mol"]
        fh.write(",".join(cols) + "\n")
        for r in rows:
            fh.write(
                ",".join(str(r[c]).lower() if isinstance(r[c], bool) else str(r[c]) for c in cols)
                + "\n"
            )

    modes_file = run_dir / "normal_modes.npz"
    np.savez(
        modes_file,
        numbers=atoms.get_atomic_numbers(),
        positions=atoms.get_positions(),
        modes=nm["modes_cart"],
        frequencies_cm1=freqs,
    )

    small_imag = [freqs[i] for i in imaginary if abs(freqs[i]) < 50]
    if small_imag:
        notes.append(
            f"{len(small_imag)} small imaginary frequencies (< 50 cm⁻¹) are often "
            "numerical noise from a loose optimization; a tighter fmax removes them."
        )

    vib = {
        "method": f"finite differences (±{fd['delta']} Å), translations and rotations "
        "projected out of the mass-weighted Hessian",
        "geometry": nm["geometry"],
        "n_modes": len(rows),
        "n_imaginary": len(imaginary),
        "imaginary_cm1": [round(float(-freqs[i]), 2) for i in imaginary],
        "zpe": energy_record(
            0.5 * sum(float(e.real) for e, lam in zip(nm["energies"], nm["eigenvalues"]) if lam > 0)
        ),
        "modes": rows,
        "frequencies_csv": file_record(freq_csv, "csv"),
        "modes_file": file_record(
            modes_file, "npz", arrays=["numbers", "positions", "modes", "frequencies_cm1"]
        ),
        "mode_animations": file_record(
            modes_dir, "folder", description="mode_NNN.xyz: 20 frames, one period"
        ),
    }
    out: dict[str, Any] = {"vibrations": vib}

    if job.driver == "thermo":
        progress("thermochemistry", 1, 1, "thermochemistry", force=True)
        out["thermochemistry"] = vibrations.thermochemistry(
            atoms,
            nm["energies"],
            nm["eigenvalues"],
            geometry=nm["geometry"],
            potential_energy=potential_energy,
            multiplicity=job.multiplicity,
            temperature=job.temperature,
            pressure=job.pressure,
        )
    if want_ir and intensities is not None:
        km = intensities * vibrations.KM_MOL_PER_D2_A2_AMU
        peaks_csv = run_dir / "ir_peaks.csv"
        with peaks_csv.open("w", encoding="utf-8") as fh:
            fh.write("mode,frequency_cm1,imaginary,intensity_km_mol,intensity_D2_A2_amu\n")
            for r, i_km, i_d in zip(rows, km, intensities):
                fh.write(
                    f"{r['mode']},{r['frequency_cm1']},{str(r['imaginary']).lower()},"
                    f"{i_km:.6f},{i_d:.8f}\n"
                )
        grid, spec = vibrations.broadened_spectrum(freqs, km)
        spec_csv = run_dir / "ir_spectrum.csv"
        with spec_csv.open("w", encoding="utf-8") as fh:
            fh.write("frequency_cm1,absorbance\n")
            for x, y in zip(grid, spec):
                fh.write(f"{x:.1f},{y:.8g}\n")
        real = [(float(f), float(i)) for f, i in zip(freqs, km) if f > 0]
        strongest = sorted(real, key=lambda p: -p[1])[:5]
        out["ir"] = {
            "peaks_csv": file_record(peaks_csv, "csv"),
            "spectrum_csv": file_record(spec_csv, "csv", columns=["frequency_cm1", "absorbance"]),
            "broadening": "Lorentzian, FWHM 20 cm⁻¹, 400–4000 cm⁻¹; imaginary modes left out",
            "intensity_unit": "km/mol",
            "strongest": [
                {"frequency_cm1": round(f, 2), "intensity_km_mol": round(i, 3)}
                for f, i in strongest
            ],
        }
    return out


def run(job: RunInput) -> dict[str, Any]:
    """Run one calculation and write its files; returns the result record."""
    import ase

    t0 = time.time()
    run_dir = Path(job.run_dir)
    progress = Progress(run_dir)
    notes: list[str] = []
    warnings_out: list[str] = []
    resolved = job.calculator.resolved()

    atoms = read_structure(job.structure)
    input_record = structure_record(Path(job.structure), atoms)
    if len(atoms) == 0:
        raise AseError("invalid_input", "The structure has no atoms.")
    if any(atoms.pbc):
        notes.append(
            "The structure is periodic; thermochemistry and symmetry treat it as a free molecule."
        )

    progress("setup", 0, None, f"loading {resolved['type']} calculator", force=True)
    workdir = run_dir / "calc"
    if resolved["type"] in ("nwchem", "orca"):
        workdir.mkdir(exist_ok=True)
    try:
        calc, info, calc_notes, versions = calculators.build(
            resolved, job.charge, job.multiplicity, workdir
        )
    except AseError:
        raise
    except ImportError as exc:
        raise AseError(
            "calculator_unavailable",
            f"The {resolved['type']} calculator could not be imported: {exc}.",
        ) from exc
    notes.extend(calc_notes)
    atoms.info.update(info)
    atoms.calc = calc

    if job.driver == "ir" and len(atoms) > 1:
        require_dipole(atoms, resolved["type"])
    constraints, constraint_records = _constraints(atoms, job.constraints, notes)
    if constraints:
        atoms.set_constraint(constraints)
        if job.driver == "energy":
            notes.append("Constraints do not change a single-point energy.")

    result: dict[str, Any] = {
        "schema": SCHEMA,
        "run_name": job.run_name,
        "run_dir": str(run_dir),
        "driver": job.driver,
        "status": "completed",
        "input_structure": input_record,
        "calculator": {
            **resolved,
            "models_spin": calculators.models_spin(resolved),
            "models_charge": calculators.models_charge(resolved),
            "versions": versions,
        },
        "charge": job.charge,
        "multiplicity": job.multiplicity,
        "constraints": constraint_records,
    }

    optimized = False
    if job.driver != "energy" and (job.driver == "opt" or job.optimize) and len(atoms) > 1:
        result["optimization"] = _optimize(atoms, job, run_dir, progress)
        optimized = True
        if not result["optimization"]["converged"]:
            result["status"] = "not_converged"
            warnings_out.append(
                f"The optimization did not reach fmax {job.fmax} eV/Å in {job.steps} steps "
                f"(last fmax {result['optimization']['final_fmax_eV_per_A']} eV/Å). Restart "
                "from the final structure or raise steps."
            )
    elif job.driver in ("vib", "thermo", "ir") and not job.optimize:
        notes.append(
            "The structure was used as given (optimize false): frequencies are only "
            "meaningful at a stationary point."
        )

    progress("energy", 0, None, "energy", force=True)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces(apply_constraint=False)
    fmax = float(np.sqrt((forces**2).sum(axis=1).max()))
    result["energy"] = energy_record(energy)
    result["fmax_eV_per_A"] = round(fmax, 6)
    from ase.calculators.calculator import PropertyNotImplementedError

    try:
        dip = atoms.get_dipole_moment()
        result["dipole"] = {
            "e_A": [round(float(x), 6) for x in dip],
            "debye": round(float(np.linalg.norm(dip)) * vibrations.DEBYE_PER_E_A, 4),
        }
    except (PropertyNotImplementedError, NotImplementedError):
        result["dipole"] = None  # the calculator gives no dipole: recorded as absent

    if optimized:
        final_path = run_dir / "final.xyz"
        atoms.info["energy_eV"] = round(energy, 8)
        result["final_structure"] = write_xyz(
            final_path, atoms, f"{job.run_name} final, E = {energy:.6f} eV"
        )
        if constraint_records:
            result["final_constraint_values"] = [
                geometry.measure(atoms, c["indices"])
                for c in constraint_records
                if c["type"] != "fix_atoms"
            ]

    if job.driver in ("vib", "thermo", "ir") and len(atoms) > 1:
        result.update(_vibrations(atoms, job, run_dir, energy, progress, notes))
        thermo = result.get("thermochemistry")
        if thermo:
            warnings_out.extend(thermo.pop("warnings"))
    elif job.driver == "thermo":  # one atom: translational and electronic terms only
        result["thermochemistry"] = vibrations.thermochemistry(
            atoms,
            np.array([]),
            np.array([]),
            geometry="monatomic",
            potential_energy=energy,
            multiplicity=job.multiplicity,
            temperature=job.temperature,
            pressure=job.pressure,
        )
        warnings_out.extend(result["thermochemistry"].pop("warnings"))
    elif job.driver in ("vib", "ir"):
        notes.append("A single atom has no vibrations.")

    result["warnings"] = warnings_out
    result["notes"] = notes
    result["wall_time_s"] = round(time.time() - t0, 2)
    result["environment"] = {
        "python": platform.python_version(),
        "ase": ase.__version__,
        "numpy": np.__version__,
    }
    results_file = run_dir / "result.json"
    result["results_file"] = str(results_file)
    results_file.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
    progress("done", 1, 1, "done", force=True)
    return result
