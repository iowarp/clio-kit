# FDMNES MCP

X-ray absorption near-edge spectra (XANES) of crystals and molecules with
[FDMNES](https://fdmnes.neel.cnrs.fr/) (Institut Néel, CNRS): run a calculation on a
structure file, normalize a computed or measured spectrum, and compare spectra.

Structures travel **as files**: a CIF or POSCAR (periodic, written as `Crystal` with
10-digit fractions so FDMNES can find the symmetry) or an XYZ, SDF/MOL or PDB molecule
or cluster (written as `Molecule` in Cartesian Å). Runs go to
`chemistry/fdmnes/<run-name>/` under `output_dir`, else `$CLIO_KIT_ARTIFACTS` (CLIO sets
it to the workspace), else the current folder. A taken name gets `-2`, `-3` …

## Tools

| Tool | What it does |
| --- | --- |
| `run_xanes(structure, absorber, edge="K", method="green", radius=6, scf=true, quadrupole=false, magnetism=false, energy_range?, extra_keywords?, timeout_s?, run_name?, output_dir?)` | Writes the FDMNES input, runs FDMNES (SCF cycles are sent as progress), and writes `spectrum.csv` (one per site when `absorber` is a list of sites) and `result.json`. Reports the threshold and edge energy, inequivalent absorbing sites, point group, cluster size, SCF cycles, and for each spectrum E0, white line, edge step and peaks. |
| `read_spectrum(path, energy_shift=0, output_dir?)` | Reads an FDMNES `*_conv.txt` or a measured two-column file (text or CSV), shifts and normalizes it, writes `spectra/<name>.normalized.csv`. |
| `compare_spectra(paths, labels?, align=false, output_dir?)` | Normalizes each spectrum, optionally shifts each so its E0 matches the first, interpolates all onto the first one's grid over the shared range, writes `spectra/comparison.csv`, and reports each one's E0, shift, R-factor (Σ(y−y₀)²/Σy₀²) and correlation against the first. |
| `fetch_materials_project(query, energy_above_hull_max=0.05, max_results=5, output_dir?)` | Crystal structures from the Materials Project by formula or `mp-` id, written as CIF (with magnetic moments) under `chemistry/fdmnes/structures/`. A formula returns every material within the hull window, ordered by energy above the hull, with space group and magnetic ordering; none is chosen. Needs `MP_API_KEY`. |
| `fdmnes_status()` | Whether FDMNES is found, where, and how to install it. |

- `absorber` is always given: an element (`"Fe"`, every Fe site; FDMNES computes the
  inequivalent ones and sums them) or 0-based site indices (`[0, 4]`, one spectrum each,
  FDMNES's `Absorber`). Nothing is chosen for the caller.
- `method`: `green` is multiple scattering with a muffin-tin potential (fast, the usual
  first choice); `fdm` is the finite-difference method with a full potential (slow,
  better for low-symmetry sites and pre-edges).
- `energy_range` is FDMNES's `Range` (eV from the Fermi level): default
  `[-30, 1, -10, 0.2, 20, 0.5, 80]`.
- `extra_keywords` are further FDMNES input blocks, written verbatim (for example
  `"Hubbard\n 5."` or `"Spinorbit"`); `Filout` and `End` are refused.

### Spectrum CSV

`energy_eV` (from the Fermi level, FDMNES's convention), `mu` (the convoluted
cross-section FDMNES writes, Mbarn), `photon_energy_eV` (`energy_eV` + the tabulated edge
energy FDMNES reports; for lining up with a measurement start from `compare_spectra`
with `align`, since the absolute position is not predictive), and `mu_norm` when the
range allows normalization.

Normalization: E0 is the steepest rise; a line fitted to the points at least 10 eV
below E0 is subtracted, and the result divided by the edge step at E0 (a line fitted
to the points at least 20 eV above E0, minus the pre-edge line). When either window
has fewer than 3 points the spectrum is not normalized and the result says why
(`normalization_skipped`). The result notes when FDMNES reports an unconverged SCF.

## Needs

The FDMNES executable, found as `FDMNES_COMMAND`, `FDMNES_EXE` (ChemGraph's variable),
or `fdmnes` / `fdmnes_linux` on PATH (`binary_missing` otherwise). This server never
downloads FDMNES. The `fdmnes` plugin's install skill builds it with Spack from the
Fortran source (`spack install fdmnes`, recipe in clio-kit's Spack repository): serial,
Gaussian solver, LAPACK.

`fetch_materials_project` needs a Materials Project API key in `MP_API_KEY` (free after
signing in at next-gen.materialsproject.org). It is a small client of the REST API
(`api.materialsproject.org/materials/summary/`, `X-API-KEY`), not mp-api or pymatgen.
Only the mocked tests ran here: no key was available, so no live fetch was made.

Errors end with `[reason]`: `not_found`, `invalid_input`, `unreadable_structure`,
`binary_missing`, `fdmnes_failed` (with the last lines of FDMNES's output; the run
folder and `result.json` with `status: failed` are kept), `timeout`,
`unreadable_spectrum`, `mp_key_missing`, `mp_key_rejected`, `mp_unavailable`.

## Tests

```sh
uv run --frozen pytest -m "not integration"     # offline, with a stand-in executable
FDMNES_COMMAND=/path/to/fdmnes uv run --frozen pytest -m integration   # real FDMNES
```

The stand-in writes a Cu K-edge spectrum recorded from a real FDMNES run
(`tests/fixtures/cu_conv_fdmnes.txt`). Measured here with FDMNES (revision 5 November
2026) built from source with gfortran 13 and LAPACK (serial, Gaussian solver): Cu fcc
K edge, Green, radius 4 Å, SCF: 1 inequivalent site, point group m3m, 16 s on one CPU.

## NOTICE

Part of the Chemistry in CLIO work, which ports ChemGraph (argonne-lcf/ChemGraph,
Apache-2.0) to clio-kit. `inputs.py` is adapted from ChemGraph's
`tools/xanes_core.py` `write_fdmnes_input`, and the normalization in `spectrum.py` from
its `get_normalized_xanes`; both say so and what changed (the absorber is never chosen
for the caller, molecules in Cartesian Å, 10-digit fractions, E0 from the steepest rise,
no silent normalization). ChemGraph's Materials Project fetch (mp-api) is replaced by a small REST client
(`materials.py`); the Parsl wrappers and plotting were not ported. The Apache License 2.0 is at https://www.apache.org/licenses/LICENSE-2.0.
FDMNES is by Yves Joly and collaborators (Institut Néel, CNRS) and is not distributed
here.
