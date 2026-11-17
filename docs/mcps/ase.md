---
title: Ase MCP
description: "Molecular calculations with ASE: SMILES to 3D, energies, optimizations, vibrations, thermochemistry and IR spectra."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Ase"
  icon="🧪"
  category="Utilities"
  description="Molecular calculations with ASE: SMILES to 3D, energies, optimizations, vibrations, thermochemistry and IR spectra."
  version="1.0.0"
  actions={["smiles_to_structure", "inspect_structure", "set_geometry", "run_ase", "displace_along_mode", "trajectory_frame", "calculators"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "chemistry", "ASE", "xTB", "MACE", "UMA", "NWChem", "ORCA", "thermochemistry", "vibrations"]}
  license="BSD-3-Clause"
  tools={[{"name": "smiles_to_structure", "description": "Build a 3D structure from a SMILES string with RDKit: ETKDG embedding, hydrogens added, MMFF94 (else UFF) relaxation. Writes XYZ and SDF and returns the structure record, the canonical SMILES, formal charge and radical-electron count.", "function_name": "smiles_to_structure"}, {"name": "inspect_structure", "description": "Report a structure file's formula, composition, bonds (covalent radii) with lengths, fragments and size, and measure distances (2 indices), angles (3) or dihedrals (4) for the index lists in measure. Indices are 0-based in file order. For a trajectory, frame picks the frame (default last).", "function_name": "inspect_structure"}, {"name": "set_geometry", "description": "Write a new structure with a distance (\u00c5), angle (degrees) and/or dihedral (degrees) set to a value, each as {indices, value}. The smaller of the two fragments the bond splits the molecule into moves rigidly; bonds in rings are refused. Reports the values before and after.", "function_name": "set_geometry"}, {"name": "run_ase", "description": "Run a calculation on a structure file. driver: energy (single point), opt (optimize), vib (optimize, then harmonic frequencies), thermo (vib + ideal-gas H, S, G at temperature/pressure), ir (vib + IR intensities; needs a dipole: xtb, nwchem, orca). calculator: {type: emt|xtb|mace|uma|nwchem|orca, model?, method?, basis?, ...settings}, e.g. {type: 'xtb'} (GFN2-xTB), {type: 'mace'} (MACE-OFF23 small), {type: 'nwchem', method: 'B3LYP', basis: '6-31G*'}. charge and multiplicity go to the calculator and the thermochemistry. constraints hold atoms or internal coordinates during the optimization: {type: 'fix_atoms', indices}, {type: 'bond', indices: [a, b], value_A?}, {type: 'angle', indices, value_deg?}, {type: 'dihedral', indices: [a, b, c, d], value_deg?}; frequencies always use the unconstrained Hessian. optimize false runs vib/thermo/ir on the structure as given. Writes result.json, final.xyz, the trajectory (extended XYZ) and a per-step CSV, frequencies.csv, one animation per mode, IR CSVs. The first xtb/mace/uma calculation builds that calculator's environment (minutes).", "function_name": "run_ase"}, {"name": "displace_along_mode", "description": "Write a structure displaced along a normal mode, e.g. to leave a saddle point along its imaginary mode before re-optimizing. source: a run_ase result JSON (with vibrations), its normal_modes.npz, or a mode_NNN.xyz animation; mode_index as in frequencies.csv; amplitude is the largest atomic displacement in \u00c5. both=true also writes the opposite direction.", "function_name": "displace_along_mode"}, {"name": "trajectory_frame", "description": "Write one frame of a trajectory (e.g. opt_trajectory.xyz or an ASE .traj) as a structure file, to restart from it. frame is 0-based; -1 is the last.", "function_name": "trajectory_frame"}, {"name": "calculators", "description": "Which calculators are ready here, which need setup and what each missing one needs (NWChem/ORCA executables, a Hugging Face token for UMA, uv for the xTB/MACE/UMA environments), and which calculator environments are already prepared. prepare: calculators whose environment to build now (xtb, mace, uma), so the first calculation does not wait for it.", "function_name": "calculators"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `smiles_to_structure`, `inspect_structure`, `set_geometry`, `run_ase`, `displace_along_mode`, `trajectory_frame`, `calculators`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
