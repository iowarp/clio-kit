---
title: Dependency upgrade, November 2026
description: Outdated packages, upgrades, holds and verification for the launcher, servers, clio-agentic-search, the pdf server and the website.
---

# Dependency upgrade, November 2026

Branch `chore/dependency-upgrade-2026-11`, from `main` at 2333f5f (after the
v2.13.0 release commit 27531f7). Latest versions come from PyPI and npm on
2026-11-21. Every package was re-locked with its own tool: `uv lock` for each
uv project and `npm update` for the npm locks.

## Method

1. **Patch/minor first.** Every locked package moved to its newest release
   within its current major. Projects were grouped by fastmcp lane.
2. **Majors one at a time.** Each major bump got its own commit. ruff counts
   0.x minor bumps as majors, and mypy, pyarrow, google-genai, isort,
   fastapi, uvicorn, duckdb, sentence-transformers and prometheus-client
   each had one.
3. **Transitive majors last.** These are the majors the declared ranges
   allow, in one commit per package group.

Python servers keep their `.python-version` pins. No `requires-python` or
`.python-version` changed, so many locks now fork by interpreter. Python 3.10
keeps the last release that supports it: fastmcp 4.1.8, starlette 1.7.0,
numpy 2.2.6, pandas 2.3.3 and pyarrow 25.0.1. Python 3.11 and later take
fastmcp 4.2.4, starlette 1.10.0 and pyarrow 26.0.1. numpy 2.5.5 needs 3.12
or later.

## Shared packages across the 38 Python servers

Counts are servers per locked version set. `a / b` means the lock forks by
Python version.

| Package | Before | After | Latest | Changes that matter |
|---|---|---|---|---|
| fastmcp | 4.1.4 (20); 4.2.0 (9); mixed 4.1.8/4.2.x (4); 4.2.0a1 (1) | 4.1.8 / 4.2.4 (22); 4.2.4 (12) | 4.2.4 | 4.1.7-4.1.8 harden OAuth redirects and require patched joserfc/Authlib. 4.2 requires Python 3.11 or later, renames the skills wire model to `MCPSkill` and deprecates the 4.1 skills API (removed in 4.3). 4.2.1 rejects JWTs with a future `nbf`. 4.2.4 fixes an OAuth-proxy double-redeem race (GHSA-78c8-3jhx-5mg7). |
| mcp | 2.3.0 (36); 1.27.2 (1) | 2.3.0 (36); 1.27.2 (1) | 2.3.0 | 2.3 requires `httpx2>=2.10` and rejects invalid `x-mcp-header` annotations at registration. `ctx.meta`/`ctx.params` can be `None`. copernicus stays on 1.27.2 because upstream `copernicus-mcp` 0.7.2 pins it. The launcher moved from 2.2.0 to 2.3.0. |
| pydantic | 2.12.5 (12); 2.13.4 (9); 2.14.3 (17) | 2.14.4 (38) | 2.14.4 | 2.14 drops Python 3.9 and makes `model_copy()` deep-copy fields that aren't updated. 2.14.2 treats a key matching only an unused alias as extra, so `extra='forbid'` can newly fail. pydantic-core moved from 2.41/2.46 to 2.50.5. |
| httpx | 0.28.1 (13) | 0.28.1 (13) | 0.28.1 | Already the latest stable. 1.0 exists only as dev pre-releases. mcp and google-genai 3 now use `httpx2`. |
| numpy | 2.2.6 to 2.5.4 in 9 combinations | 2.2.6 / 2.4.6 / 2.5.5 (12); 2.4.6 / 2.5.5 (7); 2.5.5 (3); 2.4.6 (1) | 2.5.5 | 2.5 requires Python 3.12 or later. It removes `numpy.distutils`, `row_stack` and `recfromcsv`/`recfromtxt`, makes `linalg.eig` always return complex, and changes `Generator.binomial`/`multinomial` streams. ase stays on 2.4.6 because its ML extras' ranges block 2.5. |
| xarray | 2026.9.0 (3); 2025.12.0 (1) | unchanged | 2026.9.0 | ocean holds 2025.12.0 on purpose (see holds). 2026.x requires zarr 3, removes zarr v2 writing, turns bottleneck off by default and makes `identical()` compare indexes. |
| starlette | 1.3.1 (21); 1.7.0 / 1.10.0 (5); 1.10.0 (12) | 1.7.0 / 1.10.0 (23); 1.10.0 (15) | 1.10.0 | 1.7 requires anyio 4 and returns 400 for invalid multipart. 1.8 drops Python 3.10. 1.9 changes `TestClient.stream()` and `request.url` path encoding. 1.10 makes `QueryParams` equality order-sensitive. |
| pyarrow | 21 / 23 / 25 / 26 | 21.0.0 (copernicus); 25.0.1 / 26.0.1 (parquet, pandas); 26.0.1 (2) | 26.0.1 | 25 stops adding path partition columns when reading a single Parquet file. 26 drops Python 3.10 and bounds schema nesting depth on read. |

## Majors taken

| Package | Where | Notes |
|---|---|---|
| ruff 0.15/0.16 -> 0.17.6 | launcher, clio-agentic-search, 21 servers | 0.16 grew the implicit default rule set from 59 to 413 rules, so projects without a `select` jumped to 10-307 findings. Those 17 servers and the launcher now pin `select = ["E4", "E7", "E9", "F"]`, the pre-0.16 defaults. Ruff reads the nearest `[tool.ruff]` table, so the six servers that had none (copernicus, geoprocess, geoservices, grids, ocean, weather) now carry an explicit table. That keeps them on ruff's own defaults instead of inheriting the launcher's selection. 0.16 also formats Markdown code blocks, which needed a one-line fix in parallel-sort's `USAGE_GUIDE.md`. |
| mypy 1.19/1.20 -> 2.4.1 | adios, arxiv, compression, hdf5, jarvis, ndp, plot, clio-agentic-search | 2.0 turns on `--local-partial-types` and `--strict-bytes` by default, and 2.4 makes the native parser the default. clio-agentic-search raised its cap to `<3` and sets `native_parser = false` (see fastapi). |
| pyarrow 23 -> 25/26 | parquet; pandas (25 -> 26 on Python 3.11+) | See the shared table. |
| google-genai 1.75 -> 2.33 / 3.6.1 | chronolog | 3.0 defaults to the `v1` API and the `httpx2` client. The package is declared but never imported under `chronolog/src`. |
| isort 8 -> 9 | slurm (dev group) | Not configured or run by CI. |
| fastapi 0.139 -> 0.165 | jarvis, clio-agentic-search | 0.155 rejects unknown `Query`/`Path` kwargs, 0.151 separates explicit null from an omitted value, and 0.143 makes OpenTelemetry opt-in. fastapi now depends on `opentelemetry-api`, which trips a mypy 2.4 native-parser false positive in clio-agentic-search. |
| uvicorn 0.41 -> 0.54 | clio-agentic-search | 0.50 exits with code 3 on startup failure and no longer imports the app in the parent process. |
| duckdb 1.5 -> 2.0.1 | clio-agentic-search (cap raised to `<3`) | 2.0 makes the PEG parser the default and moves ICU into an extension. Files written by 2.0 cannot be opened by duckdb 1.x. |
| sentence-transformers 5.7 -> 6.3 | clio-agentic-search `semantic` extra | 6.x requires transformers 5 and returns float32 `CrossEncoder` scores. The embedder loads and encodes on 6.3. |
| prometheus-client 0.24 -> 0.26 | clio-agentic-search `telemetry` extra | |
| Transitive | most fastmcp servers | attrs 26, cachetools 8, cyclopts 5, websockets 17, rich 15, rich-rst 2, python-dateutil 3/4, affine 3 (geo, terrain), url-normalize 4 (weather), primp 2 (web); in clio-agentic-search, huggingface-hub 2, protobuf 7, rich 15 and setuptools 84. |

## Held back

| Package | Where | Why |
|---|---|---|
| xarray 2025.12.0 (2026.9.0 available) | ocean | `pyproject.toml` pins `xarray<2026` because Parcels 3 writes zarr v2, and xarray 2026.x dropped zarr v2 writing. |
| parcels 3.1.4 (4.0.3 available) | ocean | 4.0 is a rewrite in pure Python on xarray, dask and zarr. It removes APIs and is slower above about 100,000 particles, so it needs a code migration, not a re-lock. |
| readability-lxml 0.8.4.1 (0.9 available) | web | Resolving 0.9 downgrades chardet from 7.7.0 to 5.2.0 and cssselect from 1.6.0 to 1.3.0, and extraction output changes because hidden and `<noscript>` content is stripped. |
| pytest-asyncio 1.5.0 (2.0.0a1 available) | web | 2.0 is only an alpha. web resolves pre-releases, so it was excluded explicitly. |
| torch 2.10.0 (2.15.0 available) | clio-agentic-search `semantic` extra | 2.15 moves the Linux wheels from CUDA 12 to CUDA 13 (cuda-bindings 13, newer driver floor). That GPU-compatibility choice belongs to maintainers. |
| tensorflow / tensorflow-cpu 2.20.0 (2.21.0 available) | decimer | Exact pin plus override for DECIMER. The live model tests skip here, so 2.21 cannot be verified. |
| mcp 1.27.2, pyarrow 21.0.0 | copernicus | Pinned by upstream `copernicus-mcp` 0.7.2. `uv lock -P mcp -P pyarrow` resolves no change. |
| fastmcp 4.2, numpy 2.5, pandas 3, starlette 1.8+, pyarrow 26 on Python 3.10 | servers whose `requires-python` includes 3.10 | Those releases drop 3.10, and CI still tests 3.10. Raising the floor is a separate decision. |

## npm packages

| Package | Before | After | Latest | Notes |
|---|---|---|---|---|
| website: @docusaurus/* | 3.10.2 | 3.10.2 | 3.10.2 | 4.0 exists only as canaries. `npm update` changed 211 transitive packages. |
| website: react, react-dom | 19.3.1 | 19.3.1 | 19.3.1 | Docusaurus has supported React 19 since 3.7. |
| website: @mdx-js/react, clsx, prism-react-renderer | 3.1.1, 2.1.1, 2.4.1 | unchanged | same | |
| pdf: @modelcontextprotocol/server-pdf | 2.0.4 | 2.0.4 | 2.0.4 | Five transitive packages updated. |

## Verification

The tests ran as root in a Linux container. Each server ran in its own locked
environment on its pinned Python, following CI: `uv sync --frozen
--all-extras --dev` (`--dev` only for ase, which declares conflicting
extras), `ruff check`, `ruff format --check`, `mypy src --ignore-missing-imports`
and `pytest tests`. Each failing server was also run on unchanged `main`.

- **Launcher root.** `pytest tests` gives 747 passed. ruff check and format,
  mypy (42 files) and `check_file_size.py` all pass. On `main`, 2 tests failed
  because the geoservices 1.1.0 generated artifacts were never regenerated.
  This branch regenerates them.
- **`verify_marketplace_install.py --all-servers`.** Builds, installs, every
  workflow step, the TypeScript and Go fixtures, and 37 server connections
  pass. The run then stops at `connect-home-assistant`, because that server
  exits without Home Assistant credentials. `main` stops at the same point.
  A first cold run also timed out the Go fixture while the go1.25 toolchain
  downloaded; it passed on the re-run and on `main`.
- **Servers.** 31 of 38 pass everything. grids failed ruff on `main` and now
  passes. The remaining failures are identical on `main`:
  - geoservices, ocean and weather mypy (pyproj and xarray typing);
  - one root-only permission test each in jarvis, spack and parallel-sort;
  - ndp `test_module_imports` (`server.BaseModel`).
- **Python 3.10 and 3.11 spot checks.** hdf5 (237 passed) and parquet (294
  passed) on both.
- **clio-agentic-search.** `uv sync --locked --dev`, ruff, strict mypy and
  pytest (106 passed). With `--all-extras`, 104 pass. The 2 failures assume
  sentence-transformers is absent, and mypy flags a redundant cast in
  `PrometheusMetrics.export`. Both are the same before the bumps.
- **pdf server.** `npm run lint` and `npm test` (3 passed).
- **Website.** `npm --prefix website run build` fails on `main` and on this
  branch for 2 content reasons. The `../../CHANGELOG.md` link in
  `docs/releases/v2.13.0.md` doesn't resolve, which is v2.13.0 release content
  and was left alone. `docs/mcps/copernicus.md` asks for catalogue entry
  `Copernicus-Kit`, which doesn't exist. With broken Markdown links set to
  warn, the upgraded dependencies compile, and only the copernicus page fails,
  as on `main`.

## Adopted from the new versions

These changes are low-risk and clearly useful, so each went in as its own
commit. Every affected server's suite was run in its own locked
environment.

| Commit | What and why | Verified |
|---|---|---|
| `test: read MCP SDK v2 snake_case tool fields` | fastmcp 4.2 raises `FastMCPDeprecationWarning` on the camelCase aliases that MCP SDK v2 renamed. Tests now read `read_only_hint`, `destructive_hint`, `open_world_hint` and `input_schema`, which exist on both fastmcp lanes (4.1.8 and 4.2.4). | arxiv, darshan, hdf5, paraview, slurm, spack and web pass with no deprecation warnings. |
| `fix: use pandas day_of_year/day_of_week ahead of pandas 4` | pandas 3.1 deprecates `dayofyear` and `dayofweek` on `DatetimeIndex` and `Timestamp`, and pandas 4 removes them. ocean's marine-heatwave climatology and weather's climatology and verification computed day-of-year in server code. | ocean 64, weather 42 and pandas 370 tests pass, with no `Pandas4Warning`. |
| `test(clio-agentic-search): use httpx2 for the test clients` | Starlette 1.x deprecates `TestClient` on `httpx` and asks for `httpx2`, which mcp 2.3 and huggingface-hub 2 already use. | 106 tests pass; strict mypy is clean. |
| `test(clio-agentic-search): set pytest-asyncio's fixture loop scope` | pytest-asyncio 1.5 warns until `asyncio_default_fixture_loop_scope` is set. It is now set to `"function"`, the existing default. | 106 tests pass with no warnings. |
| `fix(ruff): stop config-less servers inheriting the launcher's lint selection` | Fixes a regression from the ruff 0.17 commit on this branch (see Majors taken). | ruff check and format pass for the six servers. ocean had picked up an E712 error by inheriting the root rules. |
| `style(grids): clear ruff findings in the browser stand-in script` | Removes four unused `noqa` directives and formats one test helper. | grids' ruff check and format pass for the first time; its 6 tests pass. |

Nothing was adopted from fastmcp's new `Skill` API, because no server uses
the deprecated 4.1 skills API. The NumPy 2.5 shape and dtype deprecation
warnings come from inside ase and xarray, not from server code.

## Decisions for the user

Each item lists the value, the risk and a recommendation. None of these is
implemented on this branch.

### Platform and pins

1. **Drop Python 3.10 from the servers that still declare it.**
   - Value: Python 3.10 reached end of life in October 2026. Dropping it ends
     the per-interpreter lock forks, so every server gets the same fastmcp
     4.2.4, starlette 1.10, numpy 2.5, pandas 3.1 and pyarrow 26. It also
     unlocks the APIs below that need 3.11 or 3.12 or later.
   - Risk: users on 3.10 lose new releases. `requires-python`, the CI matrix
     and `tests/test_server_python_pin.py` all change together.
   - Recommendation: do it in the release after 2.13, raising
     `requires-python` to `>=3.11` and dropping 3.10 from the
     `quality_control.yml` matrix. The `.python-version` pins (3.12/3.13)
     don't need to change.
2. **ocean: Parcels 4 with xarray 2026.x.**
   - Value: Parcels 4 is pure Python on xarray, dask and zarr 3 and writes
     Parquet. xarray 2026 brings zarr v3, `StringDType`, Arrow PyCapsule
     export and `DataTree` `apply_ufunc`.
   - Risk: a rewrite with API removals (`open_raw_zarr`, mesh and `fieldset`
     arguments). It is slower above about 100,000 particles and is described
     upstream as an early release. ocean's particle-tracking tools and the
     zarr v2 output they write would both change.
   - Recommendation: keep `parcels==3.1.4` and `xarray<2026` for now. Follow
     the v3-to-v4 migration guide on a dedicated branch, with a regression
     fixture comparing trajectories, once Parcels 4.1 lands.
3. **clio-agentic-search: torch 2.15 (CUDA 13).**
   - Value: current torch, unified with Python 3.15 and later.
   - Risk: the Linux wheels move from CUDA 12 to CUDA 13 (cuda-bindings 13),
     which raises the NVIDIA driver floor. GPU users on older drivers would
     break.
   - Recommendation: stay on 2.10 (CUDA 12) until the deployment hosts'
     drivers are confirmed. A CPU-only index, as ase uses, would take the
     CUDA question out entirely for CPU deployments.
4. **decimer: tensorflow 2.21.**
   - Value: a current TensorFlow with security and bug fixes.
   - Risk: DECIMER pins `tensorflow==2.20.0` through an override. Its model
     loading can't be tested offline here, because `test_decimer_live`
     skips.
   - Recommendation: bump the pin and the override together in a branch
     whose CI can run the live test with the DECIMER models cached.
5. **web: readability-lxml 0.9.**
   - Value: lxml 6 support and cleaner extraction, because hidden and
     `<noscript>` content is stripped.
   - Risk: the resolve downgrades chardet from 7.7 to 5.2 and cssselect from
     1.6 to 1.3, and the extracted text changes.
   - Recommendation: hold until readability-lxml relaxes those
     requirements, or accept them after checking web's extraction fixtures.
6. **web: pytest-asyncio 2.0.**
   - Value and risk: unknown; only `2.0.0a1` exists.
   - Recommendation: wait for the final release.

### New features worth adopting (design-level)

7. **clio-agentic-search storage on duckdb 2.0.**
   - Value: the `con.appender()` context manager (`append_row`/`append_rows`)
     for bulk loads, async Parquet and CSV I/O, and the `VARIANT` type for
     metadata. Embeddings currently go in as `vector_json` text; storing
     them as `FLOAT[]` would allow `array_cosine_similarity` in SQL.
   - Risk: today's writes are `INSERT OR REPLACE` upserts, and the appender
     only appends. Adopting it needs a staging table and a `MERGE`, or a
     delete-then-append. A vector-column change is a schema migration, and
     files written by 2.0 can't be opened by duckdb 1.x.
   - Recommendation: benchmark the appender with a staging table in
     `DuckDBStore`, then decide. Do the `FLOAT[]` migration with a version
     marker in the store.
8. **clio-agentic-search embeddings on sentence-transformers 6.x.**
   - Value: `encode_query()`/`encode_document()` apply each model's
     asymmetric prompts, which improves retrieval quality for models that
     define them. `model.compile()` speeds up `encode()`. `MultiVectorEncoder`
     adds ColBERT-style late interaction.
   - Risk: switching to query/document encoding changes the vectors, so
     existing indexes would need re-embedding. `compile()` adds warm-up time.
   - Recommendation: adopt `encode_query`/`encode_document` together with a
     stored embedder version that forces a re-index. Measure `compile()` on
     the benchmark suite first.
9. **MCP 2026-07-28 protocol features (fastmcp 4.2, mcp 2.3).**
   - Value:
     - Output schemas and `structuredContent` on every tool; several servers
       already return `ToolResult`.
     - Elicitation, with multi-select and URL mode, for confirmation gates
       such as copernicus's size confirmation and jarvis's or slurm's
       destructive operations.
     - `Annotated[T, Depends(fn)]` injection.
     - `ErrorHandlingMiddleware` for consistent error mapping.
     - Background-task TTL.
     - `SkillsDirectoryProvider` to serve this repository's portable skills
       over MCP.
   - Risk: behaviour that clients can see. Elicitation depends on client
     support, and output schemas turn mismatched payloads into validation
     errors (see the jarvis note below). The tasks extension (SEP-2663) is
     not confirmed in mcp 2.3.
   - Recommendation: start with output schemas on the servers with stable
     contracts (jarvis, slurm, spack already pin contracts), then
     elicitation for the existing confirm-twice flows.
10. **Data-format servers on pyarrow 26.**
    - Value:
      - `RecordBatchFileReader.count_rows()`.
      - CSV `WriteOptions(eol=, null_string=)`.
      - `ParseOptions(pad_short_rows=, ignore_extra_columns=)` for messy
        CSVs (pandas, parquet).
      - Parquet bloom-filter write options.
      - Up to 50% faster Parquet column reads, which comes for free.
    - Risk: 26 is only on Python 3.11 and later while 3.10 is supported (see
      item 1).
    - Recommendation: adopt after item 1.
11. **pydantic 2.14.**
    - Value: the `MISSING` sentinel separates "not provided" from `None` in
      tool arguments. Validation and serialization speedups and
      constant-time `SecretStr` comparison come for free.
    - Risk: a model-by-model API change. The 2.14.2 alias-as-extra rule can
      reject payloads under `extra='forbid'`.
    - Recommendation: use `MISSING` in new tools. Audit `extra='forbid'`
      models that use aliases.
12. **Starlette 1.x in the clio-agentic-search HTTP API.**
    - Value: `max_body_size=` on the app and routes bounds request bodies,
      which hardens the jobs API. There is also an experimental
      `OpenTelemetryMiddleware` and `GZipMiddleware(exclude_content_types=)`.
    - Risk: choosing a size limit could reject legitimate large ingest
      requests.
    - Recommendation: set `max_body_size` to the largest supported ingest
      payload.
13. **ruff 0.16/0.17 rules.**
    - Value: newly stable checks include `PLR1708`, `RUF036`, `RUF068`,
      `FURB192`, `LOG004`, `PLE0304`, `ISC004` and `RUF071`. ruff's new
      413-rule default set would catch much more than E4/E7/E9/F.
    - Risk: the launcher and the 17 servers pinned to the old defaults have
      10-307 findings each.
    - Recommendation: adopt the new defaults one server at a time, fixing
      findings per commit, starting with the servers already at 0 findings
      under the new defaults (ndp, spack, scientific-catalog, web).
14. **NumPy 2.5.**
    - Value: `np.sort(..., descending=True)`, a `searchsorted` up to 20x
      faster, and faster `axis=None` reductions.
    - Risk: needs Python 3.12 or later, so it's only usable where servers
      require 3.12.
    - Recommendation: revisit after item 1.

### Existing issues found while verifying (not caused by the upgrade)

15. **Unenforced pytest marks.**
    - avogadro and ocean mark tests with `pytest.mark.timeout`, but
      `pytest-timeout` isn't installed, so those timeouts never apply.
    - parquet uses an unregistered `slow` mark.
    - Recommendation: add `pytest-timeout` to those dev groups or drop the
      marks, and register `slow`.
16. **jarvis tool results don't match their output models.**
    - Two contract-pin tests trigger pydantic serializer warnings:
      `JarvisDescribePackageSearchResult` and
      `JarvisProgressSnapshotDocument` are returned as plain dicts. There
      were 6 such warnings on `main` with fastmcp 4.1.4 and 2 now.
    - Recommendation: return the models, or relax the annotations, before
      output schemas tighten.
17. **Failures that are also on `main`.**
    - mypy errors in geoservices (pyproj `transform_bounds`), ocean and
      weather (xarray typing).
    - Root-only permission tests in jarvis, spack and parallel-sort.
    - ndp `test_module_imports`.
    - The two website-build content errors.
    - Recommendation: fix these in their own small PRs.

## Per-project direct dependencies

Rows list direct dependencies that were outdated before the upgrade or
changed. "Latest" is the newest release on PyPI. Packages missing from a
table were already current.

### launcher (root)

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| click | 8.4.2 | 8.5.1 | 8.5.1 |  |  |
| mcp | 2.2.0 | 2.3.0 | 2.3.0 |  |  |
| mypy | 2.2.0 | 2.4.1 | 2.4.1 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pyyaml | 6.0.3 | 6.0.4 | 6.0.4 |  |  |
| ruff | 0.15.21 | 0.17.6 | 0.17.6 | yes |  |
| tomli | 2.4.1 | 2.5.0 | 2.5.0 |  |  |
| types-pyyaml | 6.0.12.20260906 | 6.0.12.20261113 | 6.0.12.20261113 |  |  |

### clio-agentic-search

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| build | 1.4.0 | 1.6.1 | 1.6.1 |  |  |
| duckdb | 1.4.4 | 2.0.1 | 2.0.1 | yes |  |
| fastapi | 0.139.0 | 0.165.0 | 0.165.0 | yes |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| numpy | 2.4.2 | 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pre-commit | 4.5.1 | 4.7.0 | 4.7.0 |  |  |
| prometheus-client | 0.24.1 | 0.26.0 | 0.26.0 | yes |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-benchmark | 5.2.3 | 5.3.1 | 5.3.1 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| sentence-transformers | 5.2.3 | 6.3.0 | 6.3.0 | yes |  |
| tenacity | 9.1.4 | 9.2.1 | 9.2.1 |  |  |
| uvicorn | 0.41.0 | 0.54.0 | 0.54.0 | yes |  |

### mcp-servers/adios

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| numpy | 2.2.6 / 2.4.2 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/arxiv

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pip-audit | 2.10.0 | 2.10.1 | 2.10.1 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/ase

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fairchem-core | 2.25.0 | 2.26.0 | 2.26.0 |  |  |
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| filelock | 4.2.10 | 4.2.12 | 4.2.12 |  |  |
| numpy | 2.4.6 | 2.4.6 | 2.5.5 |  | no newer version resolves with the MACE/UMA extras' ranges (uv lock -P numpy: no change) |
| platformdirs | 4.14.5 | 4.14.8 | 4.14.8 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |
| torch | 2.13.0+cpu / 2.13.0 | 2.13.0+cpu / 2.13.0 | 2.15.0 |  | resolved from the pytorch-cpu index for the MACE/UMA extras; uv lock -P torch: no change |

### mcp-servers/avogadro

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/chronolog

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| google-genai | 1.64.0 | 2.33.0 / 3.6.1 | 3.6.1 | yes | 1.75 -> 2.33.0 (3.10) / 3.6.1 (>=3.11): 3.0 defaults to the v1 API and httpx2; not imported by chronolog/src |
| h5py | 3.15.1 | 3.16.0 | 3.16.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/compression

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pip-audit | 2.10.0 | 2.10.1 | 2.10.1 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/copernicus

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/darshan

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| h5py | 3.15.1 | 3.16.0 | 3.16.0 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| matplotlib | 3.10.8 | 3.10.9 / 3.11.2 | 3.11.2 |  |  |
| numpy | 2.2.6 / 2.4.2 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pandas | 2.3.3 / 3.0.1 | 2.3.3 / 3.1.1 | 3.1.1 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic | 2.12.5 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| scipy | 1.15.3 / 1.17.0 | 1.15.3 / 1.17.1 / 1.18.1 | 1.18.1 |  |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/decimer

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| platformdirs | 4.14.5 | 4.14.8 | 4.14.8 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |
| tensorflow | 2.20.0 | 2.20.0 | 2.21.0 |  | held at the exact pin 2.20.0 (tensorflow==2.20.0 override for DECIMER); 2.21 needs a deliberate pin change, and the live model tests skip here, so it cannot be verified |
| tensorflow-cpu | 2.20.0 | 2.20.0 | 2.21.0 |  | held at the exact pin 2.20.0 (tensorflow==2.20.0 override for DECIMER); 2.21 needs a deliberate pin change, and the live model tests skip here, so it cannot be verified |

### mcp-servers/earthdata

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/fdmnes

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/geo

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| contextily | 1.7.0 | 1.7.1 | 1.7.1 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| geopandas | 1.1.3 | 1.1.4 / 1.2.0 | 1.2.0 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| matplotlib | 3.10.9 | 3.10.9 / 3.11.2 | 3.11.2 |  |  |
| mypy | 2.1.0 | 2.4.1 | 2.4.1 |  |  |
| numpy | 2.2.6 / 2.4.6 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| platformdirs | 4.10.0 | 4.12.4 / 4.14.8 | 4.14.8 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.0.3 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.16 | 0.17.6 | 0.17.6 | yes |  |
| shapely | 2.1.2 | 2.1.2 / 2.2.0 | 2.2.0 |  |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/geoprocess

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.8 | 1.9.10 | 1.9.10 |  |  |
| fastmcp | 4.1.8 / 4.2.1 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/geoservices

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.8 | 1.9.10 | 1.9.10 |  |  |
| fastmcp | 4.1.8 / 4.2.0 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/graspa

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/grids

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.8 | 1.9.10 | 1.9.10 |  |  |
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/hdf5

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| h5py | 3.15.1 | 3.16.0 | 3.16.0 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| numpy | 2.2.6 / 2.4.2 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pip-audit | 2.10.0 | 2.10.1 | 2.10.1 |  |  |
| pydantic | 2.12.5 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/home-assistant

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/jarvis

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastapi | 0.139.0 | 0.165.0 | 0.165.0 | yes | 0.165.0; same notes as clio-agentic-search |
| fastmcp | 4.1.4 | 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| pandas | 3.0.1 | 3.1.1 | 3.1.1 |  |  |
| pydantic | 2.12.5 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.34 | 0.0.34 |  |  |
| pywin32 | 311 | 312 | 312 | yes |  |
| pyyaml | 6.0.3 | 6.0.4 | 6.0.4 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.10.0 | 1.10.0 |  |  |
| types-requests | 2.32.4.20260107 | 2.33.0.20261030 | 2.33.0.20261030 |  |  |
| typing-extensions | 4.15.0 | 4.16.0 | 4.16.0 |  |  |

### mcp-servers/lmod

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/ndp

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pip-audit | 2.10.0 | 2.10.1 | 2.10.1 |  |  |
| pydantic | 2.12.5 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| pytest-httpx | 0.36.0 | 0.36.2 | 0.36.2 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/node-hardware

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| mypy | 2.2.0 | 2.4.1 | 2.4.1 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| ruff | 0.15.21 | 0.17.6 | 0.17.6 | yes |  |
| types-psutil | 7.2.2.20260518 | 7.2.2.20261111 | 7.2.2.20261111 |  |  |

### mcp-servers/ocean

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.8 | 1.9.10 | 1.9.10 |  |  |
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| parcels | 3.1.4 | 3.1.4 | 4.0.3 | yes | held: 4.0 is a rewrite (pure Python on xarray/zarr, API removals, slower above ~100k particles); needs a code migration, not a re-lock |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |
| xarray | 2025.12.0 | 2025.12.0 | 2026.9.0 | yes | held: pyproject pins xarray&lt;2026 because Parcels 3 writes zarr v2 and xarray 2026.x dropped zarr v2 writing |

### mcp-servers/pandas

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| numpy | 2.2.6 / 2.4.6 / 2.5.2 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pandas | 2.3.3 / 3.0.5 | 2.3.3 / 3.1.1 | 3.1.1 |  |  |
| pyarrow | 25.0.0 | 25.0.1 / 26.0.1 | 26.0.1 | yes | 25.0.1 on 3.10, 26.0.1 on >=3.11 |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| ruff | 0.15.21 | 0.17.6 | 0.17.6 | yes |  |
| scipy | 1.15.3 / 1.17.1 / 1.18.0 | 1.15.3 / 1.17.1 / 1.18.1 | 1.18.1 |  |  |

### mcp-servers/parallel-sort

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| pandas | 2.3.3 / 3.0.1 | 2.3.3 / 3.1.1 | 3.1.1 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |
| types-aiofiles | 25.1.0.20251011 | 25.1.0.20261107 | 25.1.0.20261107 |  |  |

### mcp-servers/paraview

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.8 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| platformdirs | 4.12.4 / 4.14.5 | 4.12.4 / 4.14.8 | 4.14.8 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/parquet

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pyarrow | 23.0.1 | 25.0.1 / 26.0.1 | 26.0.1 | yes | 23 -> 25.0.1 (3.10) / 26.0.1 (>=3.11): single-file reads no longer add path partition columns (25); nesting depth bounded (26) |
| pydantic | 2.12.5 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/plot

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| matplotlib | 3.10.9 / 3.11.1 | 3.10.9 / 3.11.2 | 3.11.2 |  |  |
| mypy | 1.19.1 | 2.4.1 | 2.4.1 | yes |  |
| numpy | 2.2.6 / 2.4.6 / 2.5.2 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pandas | 2.3.3 / 3.0.5 | 2.3.3 / 3.1.1 | 3.1.1 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic | 2.12.5 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.3.0 | 1.5.0 | 1.5.0 |  |  |
| pytest-cov | 7.0.0 | 7.1.0 | 7.1.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.2 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/pubchem

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/scientific-catalog

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.4 | 4.2.4 | 4.2.4 |  |  |
| mypy | 2.3.0 | 2.4.1 | 2.4.1 |  |  |
| pydantic | 2.13.4 | 2.14.4 | 2.14.4 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| ruff | 0.15.21 | 0.17.6 | 0.17.6 | yes |  |

### mcp-servers/seismology

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| matplotlib | 3.10.9 | 3.10.9 / 3.11.2 | 3.11.2 |  |  |
| mypy | 2.1.0 | 2.4.1 | 2.4.1 |  |  |
| numpy | 2.2.6 / 2.4.6 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.0.3 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| ruff | 0.15.16 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/slicer

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.8 / 4.2.0 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| platformdirs | 4.12.4 / 4.14.5 | 4.12.4 / 4.14.8 | 4.14.8 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/slurm

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| isort | 8.0.1 | 9.0.2 | 9.0.2 | yes | 9.0 removes deprecated options and changes some sort output; not configured or run by CI |
| mypy | 2.2.0 | 2.4.1 | 2.4.1 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| ruff | 0.15.21 | 0.17.6 | 0.17.6 | yes |  |

### mcp-servers/spack

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| mypy | 2.2.0 | 2.4.1 | 2.4.1 |  |  |
| pydantic | 2.13.4 | 2.14.4 | 2.14.4 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| ruff | 0.15.21 | 0.17.6 | 0.17.6 | yes |  |

### mcp-servers/spice

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/terrain

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.1 | 1.9.10 | 1.9.10 |  |  |
| cryptography | 50.0.0 | 50.0.2 | 50.0.2 |  |  |
| fastmcp | 4.1.4 | 4.1.8 / 4.2.4 | 4.2.4 |  |  |
| idna | 3.18 | 3.20 | 3.20 |  |  |
| mypy | 2.1.0 | 2.4.1 | 2.4.1 |  |  |
| numpy | 2.2.6 / 2.4.6 | 2.2.6 / 2.4.6 / 2.5.5 | 2.5.5 |  |  |
| pip | 26.2.1 | 26.3 | 26.3 |  |  |
| pydantic-settings | 2.14.2 | 2.15.0 / 2.16.3 | 2.16.3 |  |  |
| pygments | 2.20.0 | 2.21.0 | 2.21.0 |  |  |
| pytest | 9.0.3 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  |  |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| python-multipart | 0.0.32 | 0.0.32 / 0.0.34 | 0.0.34 |  |  |
| rasterio | 1.4.4 / 1.5.0 | 1.4.4 / 1.5.2 | 1.5.2 |  |  |
| ruff | 0.15.16 | 0.17.6 | 0.17.6 | yes |  |
| starlette | 1.3.1 | 1.7.0 / 1.10.0 | 1.10.0 |  |  |

### mcp-servers/weather

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| authlib | 1.9.8 | 1.9.10 | 1.9.10 |  |  |
| fastmcp | 4.2.0 | 4.2.4 | 4.2.4 |  |  |
| ruff | 0.17.5 | 0.17.6 | 0.17.6 |  |  |

### mcp-servers/web

| Package | Before | After | Latest | Major | Notes |
|---|---|---|---|---|---|
| ddgs | 9.14.4 | 9.16.1 | 9.16.1 |  |  |
| fastmcp | 4.2.0a1 | 4.2.4 | 4.2.4 |  | 4.2.0a1 pre-release -> 4.2.4 final |
| mypy | 2.3.0 | 2.4.1 | 2.4.1 |  |  |
| platformdirs | 4.11.0 | 4.14.8 | 4.14.8 |  |  |
| pydantic | 2.13.4 | 2.14.4 | 2.14.4 |  |  |
| pydantic-settings | 2.14.2 | 2.16.3 | 2.16.3 |  |  |
| pytest | 9.1.1 | 9.2.0 | 9.2.0 |  |  |
| pytest-asyncio | 1.4.0 | 1.5.0 | 1.5.0 |  | 2.0.0a1 is a pre-release (web resolves pre-releases); held at 1.5.0 |
| python-dotenv | 1.2.2 | 1.2.4 | 1.2.4 |  |  |
| readability-lxml | 0.8.4.1 | 0.8.4.1 | 0.9 | yes | held: 0.9 forces chardet 7.7.0 -> 5.2.0 and cssselect 1.6.0 -> 1.3.0 downgrades; extraction output also changes (hidden/noscript stripping) |
| ruff | 0.16.0 | 0.17.6 | 0.17.6 | yes |  |
| trafilatura | 2.1.0 | 2.3.1 | 2.3.1 |  |  |

