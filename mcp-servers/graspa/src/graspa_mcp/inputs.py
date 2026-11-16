"""gRASPA input folders for a single-component GCMC adsorption run.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), ``tools/graspa_core.py``
(``_calculate_cell_size``, the template substitution in ``run_graspa_core``) and the
force-field files in ``tools/files/template_graspa_sycl/`` (copied unchanged into
``templates/``). Changed: one ``simulation.input`` that both gRASPA (CUDA/HIP, which
requires ``NumberOfTrialPositions`` and ``UseChargesFromCIFFile``) and gRASPA-SYCL
read, since each ignores the keys it does not know; framework charges are an explicit
choice; the production phase has its own cycle count.
"""

from __future__ import annotations

import math
import re
import shutil
from dataclasses import asdict, dataclass
from importlib.resources import files
from pathlib import Path
from typing import Any

import numpy as np

from .errors import GraspaError

ADSORBATES = ("CO2", "H2O", "N2")
FORCE_FIELD_FILES = ("force_field.def", "force_field_mixing_rules.def", "pseudo_atoms.def")

TEMPLATE = """\
NumberOfInitializationCycles {init_cycles}
NumberOfEquilibrationCycles  0
NumberOfProductionCycles     {prod_cycles}

UseMaxStep  yes
MaxStepPerCycle 1

UseChargesFromCIFFile {cif_charges}

RestartFile no
RandomSeed  {seed}

NumberOfTrialPositions 10
NumberOfTrialOrientations 10
Widom_Trials 10
Widom_Orientation 10

NumberOfBlocks 5
AdsorbateAllocateSpace 30240
NumberOfSimulations 1
SingleSimulation yes

DifferentFrameworks yes
InputFileType cif
FrameworkName {framework}
UnitCells 0 {cells}

ChargeMethod Ewald
Temperature {temperature}
Pressure    {pressure}

OverlapCriteria 1e5
CutOffVDW {cutoff}
CutOffCoulomb {cutoff}
EwaldPrecision 1e-6

Component 0 MoleculeName             {adsorbate}
            IdealGasRosenbluthWeight 1.0
            FugacityCoefficient      1.0
            TranslationProbability   1.0
            RotationProbability      1.0
            ReinsertionProbability   1.0
            SwapProbability          2.0
            CreateNumberOfMolecules  0
"""


@dataclass
class GcmcSpec:
    """One adsorption point; every field is recorded in result.json."""

    adsorbate: str
    temperature_K: float = 298.15
    pressure_Pa: float = 101325.0
    init_cycles: int = 10000
    prod_cycles: int = 10000
    cutoff_A: float = 12.8
    framework_charges: str = "none"
    seed: int = 0

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def read_framework(path: Path) -> Any:
    from ase.io import read

    if path.suffix.lower() != ".cif":
        raise GraspaError(
            "invalid_input",
            f"{path.name}: gRASPA reads frameworks from CIF files; give a .cif.",
        )
    try:
        return read(str(path))
    except Exception as exc:  # ASE raises many types for unreadable files
        raise GraspaError(
            "unreadable_structure", f"ASE could not read {path.name}: {exc}.", path=str(path)
        ) from exc


def cell_replication(atoms: Any, cutoff: float) -> list[int]:
    """Unit cells along a, b, c so that each perpendicular width is ≥ 2 × cutoff."""
    cell = np.asarray(atoms.cell)
    volume = abs(float(np.linalg.det(cell)))
    if not np.isfinite(cell).all() or not math.isfinite(volume) or volume <= 1e-6:
        raise GraspaError("invalid_input", "The CIF has no finite, non-degenerate unit cell.")
    widths = [
        volume / float(np.linalg.norm(np.cross(cell[(i + 1) % 3], cell[(i + 2) % 3])))
        for i in range(3)
    ]
    return [math.ceil(2 * cutoff / w) for w in widths]


def framework_mass_g_mol(atoms: Any) -> float:
    return float(sum(atoms.get_masses()))


def cif_has_charges(path: Path) -> bool:
    return "_atom_site_charge" in path.read_text(errors="replace")


def check_spec(spec: GcmcSpec) -> None:
    if spec.adsorbate not in ADSORBATES:
        raise GraspaError(
            "invalid_input",
            f"adsorbate {spec.adsorbate!r}: one of {', '.join(ADSORBATES)} (single component).",
        )
    if spec.temperature_K <= 0 or spec.pressure_Pa < 0:
        raise GraspaError("invalid_input", "temperature > 0 K and pressure ≥ 0 Pa.")
    if spec.init_cycles < 0 or spec.prod_cycles <= 0:
        raise GraspaError("invalid_input", "production cycles > 0, initialization ≥ 0.")
    if not 6.0 <= spec.cutoff_A <= 20.0:
        raise GraspaError("invalid_input", "cutoff: between 6 and 20 Å.")
    if spec.framework_charges not in ("none", "cif"):
        raise GraspaError("invalid_input", "framework_charges: 'none' or 'cif'.")


def write_run(source: Path, atoms: Any, spec: GcmcSpec, run_dir: Path) -> dict[str, Any]:
    """Copy the CIF, the force field and the molecule, and write simulation.input."""
    check_spec(spec)
    charges_in_cif = cif_has_charges(source)
    if spec.framework_charges == "cif" and not charges_in_cif:
        raise GraspaError(
            "invalid_input",
            f"framework_charges 'cif' needs _atom_site_charge in {source.name}, and it has "
            "none. Assign charges first (for example with an EQeq or DDEC tool) or use "
            "'none'.",
        )
    framework = re.sub(r"[^A-Za-z0-9_-]", "_", source.stem)[:60] or "framework"
    shutil.copyfile(source, run_dir / f"{framework}.cif")
    cells = cell_replication(atoms, spec.cutoff_A)
    templates = files("graspa_mcp").joinpath("templates")
    for name in (*FORCE_FIELD_FILES, f"{spec.adsorbate}.def"):
        (run_dir / name).write_bytes(templates.joinpath(name).read_bytes())
    (run_dir / "simulation.input").write_text(
        TEMPLATE.format(
            init_cycles=spec.init_cycles,
            prod_cycles=spec.prod_cycles,
            cif_charges="yes" if spec.framework_charges == "cif" else "no",
            seed=spec.seed,
            framework=framework,
            cells=" ".join(map(str, cells)),
            temperature=f"{spec.temperature_K:g}",
            pressure=f"{spec.pressure_Pa:g}",
            cutoff=f"{spec.cutoff_A:g}",
            adsorbate=spec.adsorbate,
        )
    )
    notes = []
    if spec.framework_charges == "none":
        notes.append(
            "Framework atoms carry no charges (pseudo_atoms.def), so there is no "
            "framework–adsorbate electrostatics"
            + (
                "; the CIF has _atom_site_charge, use framework_charges 'cif' to use them."
                if charges_in_cif
                else "."
            )
        )
    return {
        "framework": framework,
        "formula": atoms.get_chemical_formula(mode="hill"),
        "n_atoms": len(atoms),
        "unit_cells": cells,
        "framework_mass_g_mol": framework_mass_g_mol(atoms),
        "cif_has_charges": charges_in_cif,
        "notes": notes,
    }
