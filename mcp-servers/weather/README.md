# Weather MCP

Install the launcher using the [CLIO Kit setup guide](../../setup.md) before
using the commands below. See [agent integrations](../../README.md#agent-integrations)
for MCP and skill configuration.

Weather and climate analysis on gridded fields and series: ensemble
exceedance probabilities, threshold events such as heat waves,
cross-sections and soundings with CAPE, CIN and lifted index, objects tracked
with TempestExtremes (atmospheric rivers, blocks), forecast scores, station
verification, cyclone tracks and their errors, ENSO and NAO indices and
composites, run evaluation against ERA5, CMIP6 regional change, and AI
forecasts and downscaling with Earth2Studio.

The agent chooses thresholds, baselines, regions and methods; every tool takes
them as arguments and returns them with the result. ERA5 point series,
deterministic open-data forecasts, Polytope and CDS requests are left to the
[ECMWF weather skills](https://github.com/ecmwf/ecmwf-weather-skills) and
[copernicus-mcp](https://github.com/CliDyn/copernicus-mcp), used as their
maintainers ship them. This server fetches only what has no keyless
maintained route: ERA5 fields from the analysis-ready ARCO-ERA5 copy on
Google Cloud, ECMWF ensemble open data, IBTrACS best tracks, NOAA GHCN-hourly station
observations and CMIP6 from the Pangeo catalog.

## Results and provenance

Every tool returns `status`, a one-sentence `summary`, the numbers, and
`outputs` (`{path, role, media_type, provenance, rows}`): tables as Parquet,
grids and sections as NetCDF. Each output has a `<file>.provenance.json`
sidecar (`clio.provenance/1`) with sources, data licences (ERA5, ECMWF open
data, CMIP6, IBTrACS, GHCNh), method, model checkpoint and its licence, and
package versions. Errors start with a reason code (`bad_input`, `no_data`,
`too_large`, `backend_unavailable`, `worker_failed`, `timeout`,
`internal_error`).

## Tools

| Tool | What it does |
| --- | --- |
| `era5_fetch` | ERA5 hourly subset from ARCO-ERA5 (no key), with a size limit |
| `ensemble_fetch` | ECMWF IFS or AIFS ensemble open data at a point or box |
| `exceedance` | Ensemble probabilities of exceeding thresholds per window and over the period |
| `threshold_events` | Calendar-day percentile threshold and events (heat waves) from a stated baseline |
| `cross_section` | Section along a path with theta, theta-e and along/across-path wind (MetPy) |
| `soundings` | Profiles at points with SB/ML/MU CAPE and CIN, lifted index, LCL and the parcel path |
| `detect_blobs` | TempestExtremes DetectBlobs + StitchBlobs (optional magnitude, e.g. IVT) |
| `stitch_blobs` | Re-stitch tagged blobs with another lifetime or overlap |
| `blob_stats` | Per-object area, centroid, fields under it, landfall on a region and accumulation |
| `climate_index` | Niño 3.4 / ONI with phases, or the EOF-based NAO with its pattern |
| `composite` | Seasonal anomaly composites per phase with Welch t-test p-values |
| `evaluate_run` | Bias map, zonal-mean bias and regional seasonal cycles against a reference |
| `cmip_change` | CMIP6 regional change per model, multi-model mean and spread (Pangeo) |
| `score` | RMSE, bias, ACC and fair CRPS by lead time; error map at a lead |
| `station_fetch` | NOAA GHCN-hourly observations for ICAO, WMO or GHCNh station ids |
| `station_verify` | Forecast at stations against observations by lead time |
| `ibtracs_fetch` | IBTrACS v04r01 best tracks |
| `track_score` | Position, pressure and wind errors of forecast tracks by model and lead |
| `ai_forecast` | FourCastNet or DLWP with Earth2Studio from ARCO ERA5, IFS or GFS |
| `tc_track` | Cyclone tracks with TempestExtremes DetectNodes/StitchNodes (CPU) or Earth2Studio (CUDA) |
| `downscale` | CorrDiffTaiwan downscaling on its trained domain |

## Prerequisites

- TempestExtremes binaries for `detect_blobs`, `stitch_blobs`, `blob_stats`
  and `tc_track`: build from
  [source](https://github.com/ClimateGlobalChange/tempestextremes) with CMake
  and NetCDF and set `TEMPESTEXTREMES_BIN` to its `bin` directory.
- Earth2Studio runs in its own environment under
  `src/weather_mcp/workers/earth2/`, built on first use (several GB, CPU
  torch from the PyTorch CPU index). Without a CUDA GPU, `ai_forecast` and
  `downscale` run only with `allow_cpu`; Earth2Studio's cyclone trackers need
  CUDA, so `tc_track` defaults to TempestExtremes.

## Configuration

- `WEATHER_MCP_OUTPUT_DIR`: where outputs go (default `./weather-outputs`).
- `WEATHER_MCP_WORKER_ENVS`: where the Earth2Studio environment is built.
- `TEMPESTEXTREMES_BIN`: TempestExtremes `bin` directory.

## Tests

`uv run pytest` runs the tools on synthetic fields with known answers
(member-counted exceedance, a planted heat wave, soundings against MetPy
directly, uniform-bias scores, fair CRPS, a one-degree track offset, a
planted ENSO event and NAO dipole, a planted moving IVT blob and cyclone).
The TempestExtremes tests skip unless `TEMPESTEXTREMES_BIN` is set.
