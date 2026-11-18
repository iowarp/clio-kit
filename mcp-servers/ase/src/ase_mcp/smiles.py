"""SMILES to a 3D structure with RDKit: ETKDG embedding, hydrogens, force-field relax.

Adapted from ChemGraph (argonne-lcf/ChemGraph, Apache-2.0), tools/cheminformatics_core.py
(``smiles_to_3d``): the embedding now uses ETKDGv3 and relaxes with MMFF94 when RDKit
has MMFF parameters for every atom, else UFF, and reports which; it writes XYZ and SDF.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .errors import AseError
from .records import slug, structure_record, structures_dir, unique_file


def build(smiles: str, name: str | None, seed: int, output_dir: str | None) -> dict[str, Any]:
    from importlib import import_module

    from ase import Atoms
    from rdkit import Chem
    from rdkit.Chem import rdMolDescriptors

    # AllChem re-exports its functions with star imports that type checkers cannot see.
    AllChem: Any = import_module("rdkit.Chem.AllChem")
    RDLogger: Any = import_module("rdkit.RDLogger")

    RDLogger.DisableLog("rdApp.*")
    text = (smiles or "").strip()
    if not text:
        raise AseError("invalid_input", "Give a SMILES string, e.g. 'CCO' for ethanol.")
    mol = Chem.MolFromSmiles(text)
    if mol is None:
        raise AseError(
            "invalid_smiles",
            f"RDKit could not parse the SMILES {text!r}. Check brackets, ring closures and "
            "valences; the pubchem server can give the SMILES of a named compound.",
            smiles=text,
        )
    canonical = Chem.MolToSmiles(mol)
    notes: list[str] = []
    unassigned = [
        idx
        for idx, label in Chem.FindMolChiralCenters(
            mol, includeUnassigned=True, useLegacyImplementation=False
        )
        if label == "?"
    ]
    if unassigned:
        notes.append(
            f"Stereocentres without a given configuration (atoms {unassigned}): RDKit "
            "embedded one arbitrary stereoisomer. Give @/@@ in the SMILES to choose."
        )

    molh = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = int(seed)
    status = AllChem.EmbedMolecule(molh, params)
    embedding = "ETKDGv3"
    if status != 0:
        params.useRandomCoords = True
        status = AllChem.EmbedMolecule(molh, params)
        embedding = "ETKDGv3 from random coordinates"
        notes.append(
            "The standard ETKDG embedding failed; the structure was embedded from random "
            "starting coordinates instead (fallback)."
        )
    if status != 0:
        raise AseError(
            "embedding_failed",
            f"RDKit could not generate 3D coordinates for {canonical!r} (also not from "
            "random coordinates). Try another seed, or start from a structure file.",
            smiles=canonical,
        )

    if AllChem.MMFFHasAllMoleculeParams(molh):
        force_field = "MMFF94"
        result = AllChem.MMFFOptimizeMolecule(molh, maxIters=2000)
    else:
        force_field = "UFF"
        notes.append("MMFF94 has no parameters for every atom here, so UFF relaxed it.")
        result = AllChem.UFFOptimizeMolecule(molh, maxIters=2000)
    if result == 1:
        notes.append(f"The {force_field} relaxation stopped at 2000 iterations unconverged.")
    elif result == -1:
        notes.append(f"{force_field} could not be set up; the embedded geometry is unrelaxed.")

    conf = molh.GetConformer()
    atoms = Atoms(
        numbers=[a.GetAtomicNum() for a in molh.GetAtoms()],
        positions=[list(conf.GetAtomPosition(i)) for i in range(molh.GetNumAtoms())],
    )
    formula = rdMolDescriptors.CalcMolFormula(molh)
    charge = Chem.GetFormalCharge(molh)
    radicals = sum(a.GetNumRadicalElectrons() for a in molh.GetAtoms())
    stem = slug(name) if name else slug(formula)
    folder = structures_dir(output_dir)
    xyz_path = unique_file(folder, stem, ".xyz")
    sdf_path = xyz_path.with_suffix(".sdf")

    from ase.io import write

    atoms.info = {"smiles": canonical, "charge": charge}
    write(str(xyz_path), atoms, format="extxyz")
    molh.SetProp("_Name", name or canonical)
    molh.SetProp("SMILES", canonical)
    with Chem.SDWriter(str(sdf_path)) as writer:
        writer.write(molh)

    atoms.info = {}
    return {
        "smiles": text,
        "canonical_smiles": canonical,
        "name": name,
        "formula": formula,
        "formal_charge": charge,
        "radical_electrons": radicals,
        "multiplicity_from_smiles": radicals + 1,
        "seed": int(seed),
        "embedding": embedding,
        "force_field": force_field,
        "structure": structure_record(xyz_path, atoms),
        "sdf": structure_record(Path(sdf_path), atoms),
        "notes": notes,
    }
