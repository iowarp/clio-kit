# Changelog

All notable user-facing changes to CLIO Kit are documented here, newest first.

## 2.12.1 - 2026-11-11

Launcher fixes, and ndp 2.2.8 works against the public NDP catalog. Every
other MCP server keeps its 2.12.0 version.

- **Self-healing server environments**: a Python server environment left
  half-built by an interrupted or denied first build (no `pyvenv.cfg` or
  interpreter) is set aside and rebuilt on the next launch instead of failing
  every later start with "not a valid Python environment". A missing
  environment is created in a temporary sibling directory and renamed into
  place, and concurrent launches of the same server wait on a per-environment
  build lock (#403).
- **`clio-kit mcp-server --prepare <server>`** builds or syncs a server's
  environment without starting it, prints one `clio-kit.mcp-prepare.v1` JSON
  result line, and exits with a typed code: 1 unknown server, 10 build failed,
  11 access denied, 12 path too long, 13 network, 14 uv missing. It is safe to
  repeat (#402).
- **ndp 2.2.8 defaults to the public NDP catalog**: with no configuration the
  server now queries the NDP CKAN catalog at
  `https://nationaldataplatform.org/catalog` (override with `NDP_CATALOG_URL`)
  instead of a retired endpoint that no longer answers. Setting `NDP_API_URL`
  still selects a self-hosted NDP Endpoint API, which is required for the
  `local` and `pre_ckan` catalogs. Connection errors name the URL and the
  variable that controls it, and the `ndp://catalogs` resource reports the
  active backend (#401).

## 2.12.0 - 2026-11-08

Five servers gain tool behavior found by running them through a coding agent
(Clio Coder) on planted scientific datasets. Every other server takes a patch
release for the joserfc 1.7.7 and pyasn1 0.6.5 lock updates (#398).

- **adios 2.3.0**: new `variable_stats` gives per-step min, max, mean and std
  with the index of each extreme in one call. `read_variable_at_step` reads
  `start`/`count` slices and refuses reads above `max_elements` (default
  10000) with a hint instead of returning 55k values. `list_bp5` defaults to
  the working directory, single string attributes come back as strings, and
  the instructions name the real tools.
- **hdf5 2.3.0**: `hdf5_aggregate_stats` takes `axis` for per-index profiles,
  `visit` returns shape, dtype, chunks, compression and attributes per node
  from an optional group path, `get_chunks` reports compression, and
  `read_partial_dataset` accepts list or bracketed `start`/`count`.
  `open_file` and `close_file` are annotated read-only.
- **pandas 2.4.0**: new read-only `aggregate` returns grouped results inline
  without writing a file, with `column:month` style date buckets, duplicate
  removal, a query filter and several operations per column. `profile_data`
  reports IQR outlier counts and the rows furthest from the median, and each
  `validate_data` violation lists up to 10 offending rows. Tools that write
  side files are annotated destructive.
- **parquet 2.3.0**: `aggregate_column` takes `group_by` and comma-separated
  operations and reads only the needed columns; `summarize` reports per-column
  null counts. Local-file tools declare `openWorldHint: false`.
- **slurm 3.1.0** (contract `clio-kit-slurm-user-v3.1`; v3 stays available):
  the default partition is flagged and node counts are reported, existing
  jobs report the scheduler's reason instead of "Job not found", an invalid
  partition error lists the valid ones, and array jobs no longer write banner
  lines into user stdout.
- **web 2.1.5**: `fetch`, `fetch_events` and `search` declare `openWorldHint`.
- Patch releases for the lock updates: arxiv 2.2.7, chronolog 2.0.5,
  compression 2.2.7, darshan 2.2.7, geo 2.3.3, jarvis 3.7.6, lmod 3.0.3,
  ndp 2.2.7, node-hardware 2.3.2, parallel-sort 2.3.2, paraview 2.2.7,
  plot 2.2.7, scientific-catalog 1.1.6, seismology 2.3.3, spack 2.3.3,
  terrain 2.2.7.

The clio-agentic-search CI jobs and documentation removed in #389 are restored,
and `clio-converter/` and `handoffs/` are no longer tracked.

## 2.11.1 - 2026-10-28

Update audited runtime dependencies for the launcher verification environment
and maintained MCP servers, including patched FastMCP and OpenTelemetry API.
Each affected server receives a new immutable patch version. Tool schemas and
the selective-installation layout remain the ones published in 2.11.0.

## 2.11.0

CLIO Kit now provides a meta-marketplace with 22 MCP servers, six workflow
bundles, 60 portable Agent Skills (20 workflow skills, the dataset-report
skill and 39 adapted Clio Coder skills), and two Claude Code agent definitions.
External authors can validate and submit plugin entries; indexed marketplace
collections compile into a versioned catalogue with recorded provenance.
Node/TypeScript and Go projects can use the locked local-project launcher.
The shipped scientific servers remain Python projects.

The CLI can install portable skill folders into an agent's discovery directory.
Native dependency bundles and agent manifests currently target Claude Code;
other clients configure skills and stdio MCP separately. See `setup.md` and the
website's marketplace documentation for tested paths and backend prerequisites.

Scientific fixes make Pandas interpolation, directional fills and categorical
mode fills honor the requested method and report actual fill counts. HDF5
labels sampled statistics with coverage and omits misleading combined totals.
Parallel-sort recognizes bracketed log levels consistently. Slurm diagnostics
use stderr so job submission cannot corrupt the MCP stdout stream.

This release incorporates main's JARVIS 1.8.1 dependency and Web task-based
fetch support, with ordinary-call compatibility for agents without the tasks
extension. All 22 server locks update vulnerable HTTP dependencies, and
fresh registry versions point to the 2.11.0 wheel. The compatible JARVIS user
schema remains v3.7.2; its component release is 3.7.4.

All 22 servers now use stable FastMCP 4.0.3 with MCP Python SDK 2.2.0.
The verification client negotiates the 2026-07-28 protocol and supports legacy
connections. HDF5 exports accept an explicit `export_format` so clients can
select CSV, JSON or NumPy without a mid-call prompt; modern connections default
to JSON. Legacy elicitation remains available when the argument is omitted.

Website builds now use a committed npm lock, updated Docusaurus dependencies
and a pre-build image-format check, documented in `website/README.md`.

### Removed and renamed

The standalone `clio-kit search` command is no longer part of this package.
The `geojson` server merged into `geo`, and `sac` and `seismic` merged into
`seismology`; tool names are unchanged, and launching a retired name prints
where it went. The launcher now lists 22 servers.

## 2.10.5

### Python 3.13: pandas and plot servers install again

The pandas and plot MCP servers declared stale numpy-2.0-transition
ceilings (`numpy<2`) that are unsatisfiable on Python 3.13-only hosts --
numpy 1.x ships no 3.13 wheels, so those servers could never install there
(found live on a university cluster). The ceilings are gone and every
related floor is re-anchored to the first numpy-2-compatible releases
(pandas >= 2.2.2, matplotlib >= 3.8.4, scipy >= 1.13, PyTables >= 3.10.1),
and -- the part that actually reaches your machine -- the BUNDLED lockfiles
both servers install from were regenerated, so `uv run --frozen` now
resolves real wheels on Python 3.10 through 3.13. Python 3.13 joined the CI
matrix so this class cannot ship again. The pandas server also no longer
drags pytest in as a runtime dependency.
## 2.10.3

### Interceptors: instrument an application with Darshan through JARVIS

A JARVIS pipeline can now bind an interceptor package (such as the Darshan
I/O profiler) to a target application using jarvis-cd's own directional
`interceptors` mechanism: the target names its interceptor, recognition is
by package type rather than name or position, and the interceptor's
environment injection follows jarvis-cd's real lifecycle. This also fixes a
real ordering defect — the kit previously called `configure()` eagerly,
while jarvis-cd's lifecycle only calls `modify_env()`, so interceptor
environments never reached the launched processes.

### JARVIS user contract advances to v3.7.2

The interceptor target-binding surface is a contract change, so the JARVIS
user contract advances from v3.7.1 to v3.7.2. Relays that verify contract
identity by exact digest must learn the v3.7.2 digest **before** this kit is
deployed to a registered cluster, or the JARVIS registration will be
refused.

## 2.10.2

### JARVIS execution outputs are now fetchable artifacts

When a JARVIS pipeline execution finishes, the files it produced directly in
its execution directory (stdout/stderr logs, dumps, checkpoints, and other
generated output) are now automatically declared as artifacts on that
execution. Previously these files existed on disk but were invisible to
`jarvis_get_execution` unless a package explicitly registered them; now they
show up in the same bounded, paged artifact listing as everything else, typed
by role (`log`, `frame`, `output`, ...) and capped so a runaway output
directory can't blow out a response. Combined with the existing bounded
inline content read (`content_max_bytes`), this means a failed run's own
`stdout.log`/`stderr.log`/simulation log is directly readable through
`jarvis_get_execution` without a separate file-access tool.

### JARVIS user contract advances to v3.7.1

The wire contract for the JARVIS user tool surface (`clio-kit-jarvis-user`)
advances from v3.7 to v3.7.1, an additive patch revision: the artifact `role`
enum gains `frame` (for scientific output files like `.h5`, `.dcd`, `.vtk`,
trajectory/checkpoint frames) alongside the execution-output declarations
above. No tool is renamed, removed, or gains a required field. Every prior
contract revision (down to `clio-kit-jarvis-user-v3`) remains loadable by
exact identifier — nothing already deployed is invalidated by this bump.

This release also closes a gap where a kit build could be asked to load the
`clio-kit-jarvis-user-v3.7` contract family and not recognize it. The kit now
fully recognizes and ships v3.7 (historical) and v3.7.1 (current).

## 2.10.1 and earlier

See [GitHub Releases](https://github.com/iowarp/clio-kit/releases) for notes
on prior versions.
