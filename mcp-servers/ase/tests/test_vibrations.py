"""Hessian projection, frequencies, symmetry numbers, thermochemistry and IR, in-process
with EMT (fast; EMT is not a molecular method, so only the machinery is checked)."""

from __future__ import annotations

import numpy as np
import pytest
from ase.build import molecule
from ase.calculators.emt import EMT
from ase.optimize import BFGS

from ase_mcp import vibrations


def relaxed(name: str):
    atoms = molecule(name)
    atoms.calc = EMT()
    BFGS(atoms, logfile=None).run(fmax=1e-4, steps=500)
    return atoms


@pytest.mark.parametrize(
    "name, n_vib, geometry",
    [("H2O", 3, "nonlinear"), ("CO2", 4, "linear"), ("NH3", 6, "nonlinear")],
)
def test_projection_leaves_3n_minus_6_or_5(name: str, n_vib: int, geometry: str) -> None:
    atoms = relaxed(name)
    fd = vibrations.hessian(atoms)
    nm = vibrations.normal_modes(atoms, fd["hessian"])
    assert len(nm["frequencies_cm1"]) == n_vib
    assert nm["geometry"] == geometry
    assert nm["modes_cart"].shape == (n_vib, len(atoms), 3)
    # Modes are orthonormal in mass-weighted coordinates.
    gram = nm["modes_mw"].T @ nm["modes_mw"]
    assert np.allclose(gram, np.eye(n_vib), atol=1e-8)


def test_saddle_keeps_its_imaginary_mode() -> None:
    """A synthetic Hessian with one negative internal curvature: the imaginary mode
    survives the projection (the positional 'skip first six' rule would drop it)."""
    atoms = molecule("H2O")
    masses = atoms.get_masses()
    sqrtm = np.repeat(np.sqrt(masses), 3)
    ext = vibrations._external_basis(atoms)
    q, _ = np.linalg.qr(np.hstack([ext, np.eye(9)]))
    internal = q[:, ext.shape[1] : 9]
    lam = np.array([-0.05, 0.3, 0.6])  # eV/Å²/amu
    hmw = internal @ np.diag(lam) @ internal.T
    hess = hmw * np.outer(sqrtm, sqrtm)
    nm = vibrations.normal_modes(atoms, hess)
    f = nm["frequencies_cm1"]
    assert (f < 0).sum() == 1 and f[0] < 0


@pytest.mark.parametrize(
    "name, sigma, group",
    [("H2O", 2, "C2v"), ("NH3", 3, "C3v"), ("CH4", 12, "Td"), ("CO2", 2, "D*h")],
)
def test_symmetry_numbers(name: str, sigma: int, group: str) -> None:
    sym = vibrations.symmetry(molecule(name))
    assert sym["symmetry_number"] == sigma
    assert sym["point_group"] == group


def test_thermochemistry_units_and_spin() -> None:
    atoms = relaxed("CO2")
    nm = vibrations.normal_modes(atoms, vibrations.hessian(atoms)["hessian"])
    e = float(atoms.get_potential_energy())
    singlet = vibrations.thermochemistry(
        atoms,
        nm["energies"],
        nm["eigenvalues"],
        geometry=nm["geometry"],
        potential_energy=e,
        multiplicity=1,
        temperature=298.15,
        pressure=101325,
    )
    triplet = vibrations.thermochemistry(
        atoms,
        nm["energies"],
        nm["eigenvalues"],
        geometry=nm["geometry"],
        potential_energy=e,
        multiplicity=3,
        temperature=298.15,
        pressure=101325,
    )
    assert singlet["geometry"] == "linear" and singlet["symmetry_number"] == 2
    h = singlet["enthalpy"]
    assert h["kJ_mol"] == pytest.approx(h["eV"] * vibrations.EV_TO_KJ_MOL, rel=1e-6)
    g = singlet["gibbs_free_energy"]["eV"]
    assert g == pytest.approx(h["eV"] - 298.15 * singlet["entropy"]["eV_per_K"], abs=1e-6)
    # Electronic entropy R ln 3 for a triplet.
    ds = triplet["entropy"]["J_per_mol_K"] - singlet["entropy"]["J_per_mol_K"]
    assert ds == pytest.approx(8.314462618 * np.log(3), rel=1e-3)


def test_ir_intensities_and_spectrum() -> None:
    atoms = molecule("H2O")
    n = len(atoms)
    hess = np.eye(3 * n) * 30.0
    dmu = np.zeros((3 * n, 3))
    dmu[0, 0] = 1.0  # only x of the O atom changes the dipole
    nm = vibrations.normal_modes(atoms, hess)
    inten = vibrations.ir_intensities(atoms, nm["modes_mw"], dmu)
    assert inten.shape == (3,) and (inten >= 0).all()
    grid, spec = vibrations.broadened_spectrum(np.array([1000.0, -50.0]), np.array([10.0, 99.0]))
    assert grid[0] == 400 and grid[-1] == 4000
    assert grid[np.argmax(spec)] == pytest.approx(1000.0, abs=2)


def test_mode_animation_one_period() -> None:
    atoms = molecule("H2O")
    mode = np.zeros((3, 3))
    mode[1, 0] = 1.0
    frames = vibrations.mode_animation(atoms, mode, frames=20, amplitude=0.25)
    assert len(frames) == 20
    shift = frames[5].positions[1, 0] - atoms.positions[1, 0]
    assert shift == pytest.approx(0.25)
    assert np.allclose(frames[0].positions, atoms.positions)
