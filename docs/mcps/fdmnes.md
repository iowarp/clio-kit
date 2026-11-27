---
title: Fdmnes MCP
description: "X-ray absorption near-edge spectra (XANES) of crystals and molecules with FDMNES: run, normalize and compare spectra."
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Fdmnes"
  icon="🔬"
  category="Utilities"
  description="X-ray absorption near-edge spectra (XANES) of crystals and molecules with FDMNES: run, normalize and compare spectra."
  version="1.0.0"
  actions={["run_xanes", "read_spectrum", "compare_spectra", "fetch_materials_project", "fdmnes_status"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["MCP", "chemistry", "XANES", "XAS", "FDMNES", "spectroscopy"]}
  license="BSD-3-Clause"
  tools={[{"name": "run_xanes", "description": "Compute a XANES spectrum with FDMNES for a structure file: a crystal (CIF, POSCAR; periodic) or a molecule/cluster (XYZ, SDF, PDB). absorber: an element ('Fe'; every Fe site, FDMNES groups the equivalent ones) or a list of 0-based site indices (one spectrum each). edge: K, L1, L2, L3, L23, M45 \u2026. method 'green' (multiple scattering, muffin-tin; fast, the usual first choice) or 'fdm' (finite differences, full potential; slow). radius: cluster radius in \u00c5 (6 is typical; convergence needs checking). scf: self-consistent potential (default on). energy_range: Emin, step, [E, step, \u2026] Emax in eV from the Fermi level. Writes the FDMNES input and logs, spectrum CSVs (energy_eV, photon_energy_eV, mu, mu_norm) and result.json, and reports E0, white line and peaks. Minutes for a small crystal with green; much longer for fdm or large radii.", "function_name": "run_xanes"}, {"name": "read_spectrum", "description": "Read a XANES spectrum (an FDMNES *_conv.txt, or a measured spectrum as two columns energy, absorption in text or CSV), shift its energies by energy_shift eV, normalize it (pre-edge line subtracted, divided by the edge step at E0, the steepest rise) and write a CSV with energy_eV, mu and mu_norm. Reports E0, white line and peaks; when the range is too short to normalize it says why.", "function_name": "read_spectrum"}, {"name": "compare_spectra", "description": "Compare two or more XANES spectra (FDMNES runs, spectrum CSVs from run_xanes, or measured two-column files): each is normalized; align true shifts each so its E0 matches the first (needed between a calculation, relative to the Fermi level, and a measurement, in photon energy); all are put on the first one's energy grid over the shared range. Writes one CSV (energy_eV and a column per spectrum) and reports each one's E0, shift, R-factor and correlation against the first.", "function_name": "compare_spectra"}, {"name": "fetch_materials_project", "description": "Crystal structures from the Materials Project by formula ('Fe3O4') or material id ('mp-19306'), written as CIF files (with magnetic moments when the material has them) under output_dir/chemistry/fdmnes/structures/. A formula returns every matching material within energy_above_hull_max eV/atom of the convex hull, up to max_results, ordered by energy above the hull, each with its space group, magnetic ordering and CIF path; none is chosen for you. Needs a Materials Project API key (MP_API_KEY; mp_key_missing otherwise).", "function_name": "fetch_materials_project"}, {"name": "fdmnes_status", "description": "Whether the FDMNES executable is available here, where it was found (FDMNES_COMMAND, FDMNES_EXE or PATH), and how to install it when it is not.", "function_name": "fdmnes_status"}]}
>

{/* clio-kit:usage:start */}

### Usage

Available tools: `run_xanes`, `read_spectrum`, `compare_spectra`, `fdmnes_status`.

Inspect the tool input schema in your MCP client before calling it. Use a small known input, check the result and any reported errors, then expand to the intended workload.

{/* clio-kit:usage:end */}

</MCPDetail>
