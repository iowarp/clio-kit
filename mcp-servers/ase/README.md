# ASE MCP

Molecular calculations with the [Atomic Simulation Environment](https://ase-lib.org):
build a 3D structure from SMILES, look at it and change its internal coordinates, then
compute an energy, an optimized geometry, harmonic vibrations, ideal-gas
thermochemistry or an IR spectrum with EMT, xTB, MACE, UMA, NWChem or ORCA.

Structures and results travel **as files**. Every tool takes a structure file (XYZ,
extended XYZ, SDF/MOL, PDB, CIF, ASE `.traj`, POSCAR) and returns a `structure` record
`{path, format, formula, n_atoms, sha256}` for each structure it writes; calculations
also return `results_file`. Outputs go to `chemistry/ase/<run-name>/` (calculations) or
`chemistry/ase/structures/` (structures) under `output_dir`, else `$CLIO_KIT_ARTIFACTS`
(CLIO sets it to the workspace), else the current folder. Nothing is overwritten: a
taken name gets `-2`, `-3` …

## Tools

| Tool | What it does |
| --- | --- |
| `smiles_to_structure(smiles, name?, seed?, output_dir?)` | RDKit: ETKDGv3 embedding, hydrogens added, MMFF94 relaxation (UFF when MMFF lacks parameters; reported). Writes XYZ + SDF; returns the canonical SMILES, formal charge and radical-electron count. Unassigned stereocentres are named in the notes. |
| `inspect_structure(structure, measure?, frame?)` | Formula, composition, bonds (covalent radii × 1.15) with lengths, fragments, extent, and distances/angles/dihedrals for the index lists in `measure` (0-based). |
| `set_geometry(structure, dihedral?, distance?, angle?, name?)` | Sets internal coordinates `{indices, value}` by moving the smaller fragment rigidly; refuses bonds in rings (`ring_bond`); reports before/after. |
| `run_ase(structure, calculator, driver, charge, multiplicity, optimizer, fmax, steps, optimize, temperature, pressure, constraints?, run_name?, output_dir?)` | The calculation (below). |
| `displace_along_mode(source, mode_index, amplitude=0.3, both?)` | A structure displaced along a normal mode (from `result.json`, `normal_modes.npz` or a `mode_NNN.xyz` animation); `amplitude` is the largest atomic displacement in Å. |
| `trajectory_frame(trajectory, frame=-1)` | One frame of a trajectory as a structure, to restart from it. |
| `calculators(prepare?)` | Which calculators are ready, which need setup and what each needs; which environments are prepared. `prepare: ["xtb", "mace"]` builds environments ahead of time. |

### run_ase

- `driver`: `energy`, `opt`, `vib` (optimize, then frequencies), `thermo` (vib + ideal-gas
  H, S, G at `temperature` K and `pressure` Pa), `ir` (vib + IR intensities; needs a
  dipole: xtb, nwchem, orca). `optimize: false` runs vib/thermo/ir on the structure as
  given (an optimized transition state, say).
- `calculator`: `{type, model?, method?, basis?, ...settings}` (or just the type);
  unknown settings are refused, not ignored, and every default is recorded in the result.

  | type | defaults | settings |
  | --- | --- | --- |
  | `emt` | ASE EMT | `asap_cutoff` |
  | `xtb` | `method` GFN2-xTB (GFN1-xTB, IPEA1-xTB) | `accuracy`, `electronic_temperature`, `max_iterations`, `solvation` (`"alpb:water"`, `"gbsa:water"`, `"cpcm:water"`) |
  | `mace` | `model` mace-off-small (`mace-off-medium/large`, `mace-mp-small/medium/medium-mpa-0`, `mace-omol-extra-large`, or a `.model` path) | `dtype` (float64), `dispersion` (MACE-MP only) |
  | `uma` | `model` uma-s-1p1, task omol | `task`, `inference_settings` |
  | `nwchem` | `method` (functional) B3LYP, `basis` 6-31G\*, theory dft | `theory`, `nprocs`, `blocks` (extra NWChem blocks) |
  | `orca` | `method` B3LYP, `basis` def2-SVP | `nprocs`, `blocks` |

- `charge` and `multiplicity` (2S+1) go to every calculator that models them (xTB
  `charge`/`multiplicity`; NWChem `charge` and `mult` (DFT) or `nopen` (HF-based); ORCA
  `*xyz charge mult`; UMA and MACE-OMOL `atoms.info`) **and** to the thermochemistry
  (electronic spin S = (multiplicity − 1)/2). EMT and MACE-OFF/MP model neither: the
  result says so in `notes` when charge ≠ 0 or multiplicity ≠ 1, and
  `calculator.models_spin` records it.
- `constraints` hold coordinates during the optimization: `{type: fix_atoms, indices}`,
  `{type: bond, indices: [a, b], value_A?}`, `{type: angle, indices: [a, b, c],
  value_deg?}`, `{type: dihedral, indices: [a, b, c, d], value_deg?}` (ASE `FixAtoms` /
  `FixInternals`). A given value is set first (moving the smaller fragment). Frequencies
  always come from the full, unconstrained Hessian, so a constrained optimization at an
  eclipsed dihedral followed by vibrations gives the torsional saddle point and its
  imaginary mode in one call.

Files in the run folder:

| File | Content |
| --- | --- |
| `result.json` | One format (`schema: ase-mcp/run@1`): input and final structure records, the calculator with every setting and package versions, charge, multiplicity, constraints, energy in eV and kJ/mol, optimization (converged, steps, final fmax), vibrations (each mode: signed frequency, imaginary flag, animation path), thermochemistry (ZPE, H, S, −TS, G and the corrections, in eV and kJ/mol; point group, symmetry number, spin), IR, dipole, warnings, notes, timing. |
| `final.xyz` | The optimized structure (extended XYZ with its energy). |
| `opt_trajectory.xyz` | Every optimization step as extended XYZ, `energy_eV` and `fmax_eV_per_A` per frame. |
| `opt_steps.csv` | `step,energy_eV,fmax_eV_per_A`: a chart linked to the trajectory frames. |
| `frequencies.csv` | `mode,frequency_cm1,imaginary,label,energy_meV[,ir_intensity_km_mol]`; imaginary modes are negative, with `label` like `293.28i`. |
| `modes/mode_NNN.xyz` | Each normal mode as 20 frames (one period, largest atomic displacement 0.25 Å) for a viewer to play. |
| `normal_modes.npz` | Equilibrium geometry, Cartesian modes and frequencies (for `displace_along_mode`). |
| `ir_peaks.csv`, `ir_spectrum.csv` | Per-mode intensities (km/mol and (D/Å)²/amu) and a Lorentzian-broadened spectrum (FWHM 20 cm⁻¹, 400–4000 cm⁻¹). |
| `input.json`, `worker.log`, `opt.log` | The exact job, the worker's output, the optimizer log. A failed run keeps `error.json`. |

How vibrations are computed: central finite differences of the forces (±0.01 Å, and of
the dipole for IR), like ASE's `Vibrations`/`Infrared`; translations and rotations are
projected out of the mass-weighted Hessian, leaving exactly 3N−6 (3N−5 linear) modes,
imaginary ones included. (ChemGraph dropped the first six modes by position, which
loses the imaginary mode of a saddle point.) Thermochemistry uses
`ase.thermochemistry.IdealGasThermo` with the point group and rotational symmetry
number from pymatgen's `PointGroupAnalyzer` and linear/nonlinear from the rank of the
rotations; imaginary modes are left out with a warning.

## Calculators and their environments

| Calculator | Where it runs | Needs |
| --- | --- | --- |
| EMT | server environment | nothing (not a molecular method: metals, and H/C/N/O for testing only) |
| xTB (tblite) | worker environment, extra `xtb` | built on first use (downloads ~30 MB) |
| MACE (mace-torch, CPU PyTorch) | worker environment, extra `mace` | built on first use (~1.3 GB); model weights download on first use into `~/.cache/mace` (MACE-OFF23 small 7 MB) |
| UMA (fairchem-core, CPU PyTorch) | worker environment, extra `uma` | built on first use (~2.3 GB); `HF_TOKEN` of a Hugging Face account that accepted the [UMA licence](https://huggingface.co/facebook/UMA) (`hf_token_missing`, also when Hugging Face rejects the token) |
| NWChem | server environment + external program | `ASE_NWCHEM_COMMAND` (full command, e.g. `mpirun -np 4 nwchem PREFIX.nwi > PREFIX.nwo`), `NWCHEM_COMMAND` (the executable) or `nwchem` on PATH (`binary_missing`). The nwchem plugin's install skill installs it with Spack. |
| ORCA | server environment + external program | `ORCA_COMMAND` or `ASE_ORCA_COMMAND` (full path of `orca`) or `orca` on PATH (`binary_missing`). ORCA is downloaded from the ORCA forum after registration. |

**Every calculation runs in a worker process** (`python -m ase_mcp.worker`), EMT
included, so the server stays responsive and a crash is contained. Progress (each
optimization step with its energy and fmax, each finite-difference displacement, an
environment build) is sent as MCP progress notifications.

**Why extras and worker environments.** The clio-kit launcher installs the server with
`uv sync --frozen --no-dev` and no extras. xTB, MACE and UMA are heavy, and MACE (e3nn
0.4.4) and UMA (e3nn ≥ 0.5) cannot share an environment (declared in `[tool.uv]
conflicts`). The wheel ships the server's own `pyproject.toml` and `uv.lock`
(`ase_mcp/_project`, via hatch `force-include`); on first use of xtb/mace/uma the
server runs `uv sync --frozen --no-dev --no-install-project --extra <x> --python
<server python>` with `UV_PROJECT_ENVIRONMENT=<user cache>/clio-kit/ase-mcp/envs/<x>-<uv.lock
hash>/venv` (`ASE_MCP_ENV_DIR` overrides the cache folder) under a file lock, so
concurrent first uses build once, and writes `ready.json`. The worker imports this
package from a copy of its source (`<cache>/code/<hash>`) through `PYTHONPATH`, so the
server's own site-packages never mix in. A new lock gets a new environment; old ones
can be deleted. Torch comes from the PyTorch CPU index (not on macOS, where PyPI's
wheels are CPU builds). uv is found as `UV` or on PATH (`uv_missing` otherwise). The
first build is reported in words and as progress. Measured here: xtb 4.5 s and mace
9.7 s from an empty uv cache on a fast network, under a second with a warm cache; the
environments take 0.6 GB (xtb), 1.3 GB (mace), 2.3 GB (uma) on disk, mostly hard links
into the uv cache (the base dependencies, RDKit and pymatgen-core, are in each).

**MACE default: MACE-OFF23 small.** MACE-OFF is trained on organic molecules (SPICE,
ωB97M-D3BJ; H, C, N, O, F, P, S, Cl, Br, I), which is what this server is for; MACE-MP
is trained on inorganic crystals (PBE) and is weaker for molecular conformations and
vibrations. "Small" is fast on a CPU and good enough for screening; choose
`mace-off-medium`/`large` for production. MACE-OFF weights are under the Academic
Software License (non-commercial); MACE-MP is MIT. MACE-OFF/MP model neither charge
nor spin and give no dipole (no IR).

## Errors

Failures are MCP tool errors whose text ends with `[reason]`: `not_found`,
`invalid_input`, `invalid_smiles`, `embedding_failed`, `unreadable_structure`,
`ring_bond`, `binary_missing`, `hf_token_missing`, `uv_missing`, `env_build_failed`,
`dipole_unavailable`, `calculator_unavailable`, `calculation_failed`. The message says
what is missing and how to provide it, and for a calculation names the run folder.

## Run

```sh
clio-kit mcp-server ase      # via the clio-kit launcher
ase-mcp                      # direct entry point
```

## Test

```sh
uv run --frozen pytest                    # offline: schemas, geometry, vibrations, EMT runs, gating
uv run --frozen pytest --run-slow         # also builds the xtb and mace environments and runs
                                          # water, O2 and the ethane rotation barrier with them
```

Real runs here (Linux x86-64, 4 CPUs, CPU only): GFN2-xTB water 1539/3643/3651 cm⁻¹;
ethane eclipsed (dihedral constrained) has one imaginary mode, 293i cm⁻¹ (xTB) and
287i cm⁻¹ (MACE-OFF23 small); displaced along it and re-optimized it relaxes to
staggered (±60°) with a barrier of 10.8 (xTB) and 10.4 (MACE) kJ/mol (experiment
about 12); NWChem B3LYP/6-31G\* water 1710/3723/3847 cm⁻¹. The NWChem test runs only
when nwchem is found; the UMA test only checks the typed error for a rejected token.

## NOTICE

Parts of this server are adapted from **ChemGraph** (https://github.com/argonne-lcf/ChemGraph),
by the ChemGraph authors (Argonne National Laboratory), licensed under the Apache License 2.0:
`tools/ase_core.py` (calculator loading, drivers, thermochemistry, symmetry number,
linear detection, result format), `schemas/ase_input.py` and
`schemas/calculators/{emt,tblite,mace,fairchem,nwchem,orca}_calc.py` (calculator
settings), and `tools/cheminformatics_core.py` (SMILES to 3D). The adapted modules say
so in their docstrings and were changed: files instead of AtomsData, one result format,
worker processes, a projected Hessian, constraints and new tools. The
LangChain/LangGraph wrappers, AtomsData tools, Parsl/Globus/EnsembleLauncher, report
tools and ChemGraph's runtime context were not ported. The Apache License 2.0 is at
https://www.apache.org/licenses/LICENSE-2.0.
