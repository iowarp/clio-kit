"""Harmonic vibrations, IR intensities and ideal-gas thermochemistry.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/ase_core.py
(``_calculate_thermochemistry``, ``get_symmetry_number``, ``is_linear_molecule``). The
Hessian is built here by central finite differences of the calculator's forces (and,
for IR, dipoles), like ``ase.vibrations.Vibrations``/``Infrared``; translations and
rotations are then projected out of the mass-weighted Hessian, so the 3N−6 (3N−5 when
linear) vibrations are reported by index, imaginary ones included, instead of
skipping the first six modes by position (which drops the imaginary mode of a
transition state). Thermochemistry uses ``ase.thermochemistry.IdealGasThermo`` with
the rotational symmetry number from pymatgen's ``PointGroupAnalyzer``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

EV_TO_KJ_MOL = 96.48533212331002
DEBYE_PER_E_A = 4.803204712570263
# (D/Å)²/amu → km/mol, the factor ASE's Infrared uses.
KM_MOL_PER_D2_A2_AMU = 42.255
DELTA = 0.01  # Å, finite-difference step (ASE default)


def hessian(
    atoms: Any,
    *,
    dipoles: bool = False,
    delta: float = DELTA,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Finite-difference Hessian (eV/Å²) and optional dipole derivatives (e)."""
    n = len(atoms)
    pos0 = atoms.get_positions().copy()
    hess = np.zeros((3 * n, 3 * n))
    dmu = np.zeros((3 * n, 3)) if dipoles else None
    total = 6 * n
    done = 0
    for a in range(n):
        for k in range(3):
            row = 3 * a + k
            forces = []
            mus = []
            for sign in (1.0, -1.0):
                pos = pos0.copy()
                pos[a, k] += sign * delta
                atoms.set_positions(pos, apply_constraint=False)
                forces.append(atoms.get_forces(apply_constraint=False).ravel())
                if dipoles:
                    mus.append(np.asarray(atoms.get_dipole_moment(), dtype=float))
                done += 1
                if progress:
                    progress(done, total)
            hess[row] = -(forces[0] - forces[1]) / (2 * delta)
            if dipoles and dmu is not None:
                dmu[row] = (mus[0] - mus[1]) / (2 * delta)
    atoms.set_positions(pos0, apply_constraint=False)
    hess = 0.5 * (hess + hess.T)
    return {"hessian": hess, "dipole_derivatives": dmu, "delta": delta}


def _external_basis(atoms: Any) -> np.ndarray:
    """Orthonormal translations + rotations in mass-weighted coordinates (3N × r)."""
    masses = atoms.get_masses()
    pos = atoms.get_positions()
    com = (masses[:, None] * pos).sum(0) / masses.sum()
    r = pos - com
    sqrtm = np.sqrt(masses)
    vectors = []
    for k in range(3):
        t = np.zeros_like(pos)
        t[:, k] = 1.0
        vectors.append((t * sqrtm[:, None]).ravel())
    for k in range(3):
        e = np.zeros(3)
        e[k] = 1.0
        rot = np.cross(e, r)
        vectors.append((rot * sqrtm[:, None]).ravel())
    b = np.array(vectors).T
    u, s, _ = np.linalg.svd(b, full_matrices=False)
    rank = int((s > 1e-6 * s.max()).sum())
    return u[:, :rank]


def normal_modes(atoms: Any, hess: np.ndarray) -> dict[str, Any]:
    """Vibrational frequencies and Cartesian modes with translations/rotations removed."""
    from ase import units

    n = len(atoms)
    masses = atoms.get_masses()
    sqrtm = np.repeat(np.sqrt(masses), 3)
    hmw = hess / np.outer(sqrtm, sqrtm)
    ext = _external_basis(atoms)
    n_external = ext.shape[1]
    # Basis of the complement of the external space.
    q, _ = np.linalg.qr(np.hstack([ext, np.eye(3 * n)]))
    internal = q[:, n_external : 3 * n]
    h_int = internal.T @ hmw @ internal
    eigval, eigvec = np.linalg.eigh(0.5 * (h_int + h_int.T))
    modes_mw = internal @ eigvec  # 3N × nvib, unit norm
    scale = units._hbar * 1e10 / np.sqrt(units._e * units._amu)
    energies = scale * np.sqrt(eigval.astype(complex))  # eV, imaginary when λ < 0
    freq_signed = np.where(eigval < 0, -np.abs(energies), np.abs(energies)) / units.invcm
    cart = modes_mw / sqrtm[:, None]
    cart /= np.linalg.norm(cart, axis=0)
    return {
        "eigenvalues": eigval,
        "energies": energies,
        "frequencies_cm1": freq_signed,  # negative = imaginary
        "modes_mw": modes_mw,
        "modes_cart": cart.T.reshape(-1, n, 3),
        "n_external": n_external,
        "geometry": "monatomic" if n == 1 else ("linear" if n_external == 5 else "nonlinear"),
    }


def ir_intensities(atoms: Any, modes_mw: np.ndarray, dmu: np.ndarray) -> np.ndarray:
    """IR intensities in (D/Å)²/amu for each mode (columns of ``modes_mw``)."""
    masses = atoms.get_masses()
    sqrtm = np.repeat(np.sqrt(masses), 3)
    dmu_dq = (modes_mw / sqrtm[:, None]).T @ dmu  # e / sqrt(amu)
    return (dmu_dq**2).sum(axis=1) * DEBYE_PER_E_A**2


def broadened_spectrum(
    freqs: np.ndarray,
    intensities: np.ndarray,
    *,
    fwhm: float = 20.0,
    start: float = 400.0,
    end: float = 4000.0,
    step: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Lorentzian-broadened spectrum of the real modes (cm⁻¹, km/mol per cm⁻¹)."""
    grid = np.arange(start, end + step, step)
    gamma = fwhm / 2.0
    spec = np.zeros_like(grid)
    for f, i in zip(freqs, intensities):
        if f <= 0:
            continue
        spec += i * (gamma / np.pi) / ((grid - f) ** 2 + gamma**2)
    return grid, spec


def symmetry(atoms: Any) -> dict[str, Any]:
    """Point group and rotational symmetry number (pymatgen PointGroupAnalyzer)."""
    if len(atoms) == 1:
        return {"point_group": "Kh", "symmetry_number": 1}
    from pymatgen.core import Molecule
    from pymatgen.symmetry.analyzer import PointGroupAnalyzer

    mol = Molecule(atoms.get_chemical_symbols(), atoms.get_positions(), charge_spin_check=False)
    pga = PointGroupAnalyzer(mol)
    return {
        "point_group": str(pga.sch_symbol),
        "symmetry_number": int(pga.get_rotational_symmetry_number()),
    }


def thermochemistry(
    atoms: Any,
    energies: np.ndarray,
    eigenvalues: np.ndarray,
    *,
    geometry: str,
    potential_energy: float,
    multiplicity: int,
    temperature: float,
    pressure: float,
) -> dict[str, Any]:
    """Ideal-gas H, S, G (and ZPE) at ``temperature`` and ``pressure``."""
    import warnings

    import ase
    from ase.thermochemistry import IdealGasThermo

    sym = symmetry(atoms)
    real = [float(e.real) for e, lam in zip(energies, eigenvalues) if lam > 0]
    n_imag = int(sum(1 for lam in eigenvalues if lam < 0))
    warnings_out: list[str] = []
    if n_imag:
        warnings_out.append(
            f"{n_imag} imaginary mode(s) were left out of the thermochemistry (as for a "
            "transition state); for a minimum, imaginary modes mean the structure is not "
            "fully optimized."
        )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        thermo = IdealGasThermo(
            vib_energies=real,
            geometry=geometry,  # type: ignore[arg-type]
            potentialenergy=potential_energy,
            atoms=atoms,
            symmetrynumber=sym["symmetry_number"],
            spin=(multiplicity - 1) / 2.0,
            vib_selection="all",
        )
        zpe = float(thermo.get_ZPE_correction())
        h = float(thermo.get_enthalpy(temperature, verbose=False))
        s = float(thermo.get_entropy(temperature, pressure, verbose=False))
    warnings_out.extend(str(w.message) for w in caught)
    g = h - temperature * s

    def both(value: float) -> dict[str, float]:
        return {"eV": round(value, 8), "kJ_mol": round(value * EV_TO_KJ_MOL, 4)}

    return {
        "temperature_K": temperature,
        "pressure_Pa": pressure,
        "geometry": geometry,
        "point_group": sym["point_group"],
        "symmetry_number": sym["symmetry_number"],
        "multiplicity": multiplicity,
        "spin_S": (multiplicity - 1) / 2.0,
        "n_modes_used": len(real),
        "n_imaginary_excluded": n_imag,
        "zpe": both(zpe),
        "enthalpy": both(h),
        "entropy": {"eV_per_K": round(s, 10), "J_per_mol_K": round(s * EV_TO_KJ_MOL * 1000, 4)},
        "minus_TS": both(-temperature * s),
        "gibbs_free_energy": both(g),
        "enthalpy_correction": both(h - potential_energy),
        "gibbs_correction": both(g - potential_energy),
        "method": "ase.thermochemistry.IdealGasThermo (harmonic oscillator, rigid rotor, "
        "ideal gas)",
        "ase_version": ase.__version__,
        "warnings": warnings_out,
    }


def mode_animation(
    atoms: Any, mode: np.ndarray, *, frames: int = 20, amplitude: float = 0.25
) -> list[Any]:
    """One period of a normal mode as ``frames`` structures; ``amplitude`` is the largest
    atomic displacement in Å."""
    peak = float(np.linalg.norm(mode, axis=1).max()) or 1.0
    vec = mode / peak * amplitude
    out = []
    pos0 = atoms.get_positions()
    for k in range(frames):
        img = atoms.copy()
        img.calc = None
        img.constraints = []
        img.set_positions(pos0 + np.sin(2 * np.pi * k / frames) * vec)
        out.append(img)
    return out
