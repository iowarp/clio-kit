# Ocean MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

Ocean analysis on observations and model output: CTD casts, Argo and glider
profiles, TEOS-10, water masses, mixed layer depth and stratification,
sections, transports and overturning, tides and spectra, sea-level trends,
bloom phenology, mesoscale eddies, SWOT swaths, Lagrangian particles, marine
heatwaves, mixed-layer heat budgets, cruise plans, model drift,
model-observation colocation and regional MOM6 pre-flight checks.

The agent chooses thresholds, baselines, regions and methods; every tool takes
them as arguments and returns them with the result. Gridded inputs
(reanalyses, altimetry, model output) are files on disk, usually fetched first
with [copernicus-mcp](https://github.com/CliDyn/copernicus-mcp). The server
fetches only Argo (argopy), IOOS gliders (ERDDAP) and SWOT swaths
(earthaccess).

## Results and provenance

Every tool returns a structured result: `status`, a one-sentence `summary`,
the numbers, and `outputs` (each `{path, role, media_type, provenance, rows}`).
Tables are Parquet (profile tables have one row per sample with `profile`,
`time`, `latitude`, `longitude`, `pressure` and the variables); grids and
sections are NetCDF. Each output has a `<file>.provenance.json` sidecar
(`clio.provenance/1`): sources with DOI or reference, subset and data licence,
the method and its parameters, and the versions of every package used. Cite it.

Errors are `ToolError`s whose message starts with a reason code:
`bad_input`, `no_data`, `credential_missing`, `backend_unavailable`,
`out_of_range`, `timeout`, `worker_failed` or `internal_error`.
`mom6_preflight` returns `status: "partial"` with `outcome: "checks_failed"`
when a check fails.

## Tools

| Tool | What it does |
| --- | --- |
| `ctd_read` | Sea-Bird `.cnv`, CCHDO WHP-exchange (`_ct1.csv`, zips), CF NetCDF or CSV casts to a profile table |
| `teos10` | gsw: SA, CT, pt0, sigma0, depth; optionally N² per profile |
| `water_masses` | Water masses by θ-S-pressure against a stated definition (built-in `atlantic_emery_meincke_1986`, or your own) |
| `argo_fetch` | Argo core or BGC profiles through argopy (box with dates, or floats), standard-mode QC |
| `glider_fetch` | Search the IOOS Glider DAC, or fetch a deployment split into transects with suspect samples flagged |
| `profile_metrics` | MLD by a stated criterion, mean N² over a band, anomaly profiles on standard levels and their vertical type |
| `section` | A section from profiles or from a gridded field along a path; the path is stored with it |
| `transport` | Volume and heat transport normal to a path (approximate on a regular grid, and says so) |
| `overturning` | Overturning streamfunction in depth or density space on the model grid; maximum at a latitude |
| `harmonics` | UTide harmonic analysis of heights or currents; comparison with another constituent table |
| `spectra` | Welch spectra, or rotary spectra of u + iv, with the inertial frequency |
| `sea_level_decompose` | Gauge and altimetry trends over their common period and their difference (local part, incl. VLM) |
| `particles` | Parcels 3 RK4 particles forward or backward from a station; trajectories and end-point density |
| `eddy_track` | py-eddy-tracker identification and tracking on daily sea level |
| `track_crossings` | Tracks reaching one side of a contour of a field (e.g. an isobath) |
| `swath_fetch` | SWOT L2 LR granules: search (no login) or download (Earthdata Login) |
| `swath_compare` | SWOT swath against a gridded field at each pass time |
| `mhw_point` | Hobday (2016) heatwaves at a point: climatology, threshold, events, metrics, categories |
| `mhw_track` | marEx detection and tracking of marine heatwaves over a region |
| `mhw_events` | Hobday metrics and categories for each tracked event |
| `bloom_phenology` | Bloom start per year (median + 5 %, or steepest rise) |
| `mixed_layer_budget` | Tendency against surface flux / (rho cp h), residual for advection and mixing |
| `cruise_plan` | Stations across a moving feature within ship time, or re-time moved stations |
| `drift` | Domain means per level and an equilibration test with a stated tolerance |
| `colocate` | Model at each observation; bias and RMSE by depth band and region |
| `mom6_preflight` | Open boundaries, dates, land mask (isolated cells, one-cell channels), minimum depth, layout and mask table |

## Copyleft workers

marEx (AGPL-3.0-or-later) and py-eddy-tracker (GPL-3.0) are not imported by
this server, which is BSD-3-Clause. Each runs in its own locked environment
under `src/ocean_mcp/workers/<name>/`, started as a subprocess that exchanges
files; the worker scripts carry the upstream's licence. The first
`mhw_track` or `eddy_track` call builds the environment with `uv sync
--frozen` (a few minutes); `OCEAN_MCP_WORKER_ENVS` sets where environments
live (default: the system temporary directory). py-eddy-tracker 3.6.1 pins
`numpy<1.23`, so its worker runs on Python 3.10.

## Credentials

Downloads of SWOT granules need NASA Earthdata Login
(`EARTHDATA_USERNAME` and `EARTHDATA_PASSWORD`, or `EARTHDATA_TOKEN`), set by
the connector. Searching needs none. Argo and the Glider DAC need none.

## Configuration

- `OCEAN_MCP_OUTPUT_DIR`: where outputs go (default `./ocean-outputs`).
- `OCEAN_MCP_WORKER_ENVS`: where worker environments are built.
- `OCEAN_MCP_UV`: path to `uv` if it is not on `PATH`.

## Tests

`uv run pytest -m "not worker"` runs the science and tool tests on synthetic
data with known answers (TEOS-10 check values, planted mixed layers,
uniform-flow transports, known tidal constituents, a planted heatwave, a
planted one-cell channel). `uv run pytest -m worker` builds the two worker
environments and runs eddy and heatwave tracking end to end.
