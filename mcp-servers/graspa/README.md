# gRASPA MCP

Gas adsorption in porous crystals (MOFs, zeolites) by grand-canonical Monte Carlo
(GCMC) with [gRASPA](https://github.com/snurr-group/gRASPA) (Snurr group, MIT
licence): the uptake of CO2, H2O or N2 in a framework CIF at a temperature and pressure,
and a ranking of frameworks by uptake or working capacity.

gRASPA needs a GPU. Where it runs here, `run_graspa` runs it; elsewhere,
`prepare_graspa` writes the run folder to submit to a GPU node (for example with the
slurm server) and `read_graspa` reads it back. Runs go to `chemistry/graspa/<run-name>/`
under `output_dir`, else `$CLIO_KIT_ARTIFACTS` (CLIO sets it to the workspace), else the
current folder; a taken name gets `-2`, `-3` …

## Tools

| Tool | What it does |
| --- | --- |
| `run_graspa(structure, adsorbate, temperature=298.15, pressure=101325, init_cycles=10000, prod_cycles=10000, framework_charges="none", cutoff=12.8, seed=0, timeout_s?, run_name?, output_dir?)` | Prepares the folder, runs gRASPA (cycles and molecule counts are sent as progress), reads the loading, writes `result.json`. |
| `prepare_graspa(...)` | The same folder without running it: `<framework>.cif`, `simulation.input`, `force_field.def`, `force_field_mixing_rules.def`, `pseudo_atoms.def`, `<adsorbate>.def`, `run.json`. Run gRASPA in it with stdout saved to `graspa_stdout.txt`. |
| `read_graspa(run_dir)` | Reads a finished folder: uptake in mol/kg with its block-average error bar, molecules in the box and per unit cell, mg/g, g/L and the Henry coefficient when printed. |
| `rank_frameworks(results, adsorption, desorption?, top_fraction=1)` | Ranks frameworks from `result.json` files (or folders holding them) by uptake at `adsorption` `{temperature_K, pressure_Pa}`, or by working capacity (adsorption minus desorption uptake). Repeats are averaged; a framework without a completed run at every requested condition is excluded with the reason. Writes `rankings/ranking.csv`. |
| `graspa_status()` | Whether gRASPA is found here and how to get it. |

- The cell is replicated so each perpendicular width is at least twice the cutoff
  (`UnitCells 0 a b c`).
- Force field: the files ChemGraph ships for gRASPA (`templates/`): Lennard-Jones
  parameters for 112 element types (UFF-like) with rigid TraPPE-style CO2 and N2 and a
  rigid water model. `framework_charges: "none"` leaves the framework uncharged (the
  result notes it, and notes when the CIF has `_atom_site_charge` that could be used);
  `"cif"` uses the CIF's charges (`UseChargesFromCIFFile yes`) and is refused when the CIF
  has none.
- One `simulation.input` serves both gRASPA builds: the CUDA/HIP gRASPA requires
  `NumberOfTrialPositions` and `UseChargesFromCIFFile`, the SYCL port reads
  `Widom_Trials`; each ignores the keys it does not know.

### Output formats read

`result.json` names the format in `loading.flavour`:

- `graspa` (snurr-group/gRASPA, CUDA or HIP): the `BLOCK AVERAGES (LOADING: mol/kg)`
  table, taking the block of the adsorbate **by name** (the framework is component 0
  and has its own row); the error bar is the block-average error (`-nan` is recorded as
  none).
- `graspa-sycl` (the SYCL port ChemGraph runs on Aurora): one `Overall: Average:`
  molecule count, converted to mol/kg with the framework mass × unit cells. No error bar.

## Needs

`run_graspa` needs the gRASPA executable: `GRASPA_COMMAND`, `CHEMGRAPH_GRASPA_EXECUTABLE`
(ChemGraph's variable), or `nvc_main.x`, `hip_main.x`, `graspa` or `sycl.out` on PATH
(`binary_missing` otherwise, pointing to `prepare_graspa`). This server never downloads
or builds it; the graspa plugin's install skill builds it with Spack on a GPU node
(`spack install graspa`). `prepare_graspa`, `read_graspa` and `rank_frameworks` need
nothing.

Errors end with `[reason]`: `not_found`, `invalid_input`, `unreadable_structure`,
`binary_missing`, `graspa_failed` (with the last output lines; the folder and a
`result.json` with `status: failed` are kept), `timeout`, `unreadable_output`.

## Tests

```sh
uv run --frozen pytest
```

No GPU was available where this was written, so gRASPA itself has not run against this
server. The tests use stand-in executables that replay both output formats: an excerpt
of the CO2-in-MFI example output shipped with gRASPA
(`tests/fixtures/graspa_cuda_co2_mfi_stdout.txt`, uptake 0.44065 ± 0.00884 mol/kg) and
the SYCL lines ChemGraph's own hermetic tests use.

## NOTICE

Part of the Chemistry in CLIO work, which ports ChemGraph (argonne-lcf/ChemGraph,
Apache-2.0) to clio-kit. Adapted from ChemGraph: `inputs.py` (cell replication and the
input template, from `tools/graspa_core.py`), the SYCL reader in `parse.py`
(`_read_graspa_sycl_output`) and the ranking in `compute.py` (`rank_records`,
`tools/graspa_analysis.py`); `templates/*.def` are ChemGraph's
`tools/files/template_graspa_sycl/` files, unchanged. Each module says what changed. The
mock runner, LangChain/Parsl/Globus wrappers and ensemble launcher were not ported. The
Apache License 2.0 is at https://www.apache.org/licenses/LICENSE-2.0. The test fixture
excerpt is from gRASPA's `Examples/CO2-MFI` (MIT, © Zhao Li, Snurr Group). gRASPA itself
is not distributed here.
