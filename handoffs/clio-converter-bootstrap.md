# Session prompt: organize clio-converter into a minimal runnable project

Paste everything below the line into a new coding-agent session started at the clio-kit repo root.

---

You own `clio-converter/` in the clio-kit monorepo (`/home/akougkas/iowarp/clio-kit`). It is a new, **low-priority** project that will eventually be a service agents call to convert and normalize scientific content (units and quantities, formulas, and later scientific file and data formats). Your job in this session is narrow: **turn the existing legacy material into a very minimal, basic, runnable package.** Do not build the big vision yet.

## Where things stand (verified at hand-off)

- Read `clio-converter/README.md` and `clio-converter/legacy/agentic-search/MANIFEST.md` first. The legacy files were moved verbatim from `clio-agentic-search` (with `git mv`, so `git log --follow` shows their history). Their imports still say `clio_agentic_search.*` and nothing runs yet.
- Check `git status` first. The separation that created this folder may still be uncommitted on `main`.
- `clio-agentic-search` no longer contains any unit, measurement or formula code and must not import converter. Do not edit `clio-agentic-search/` in this session.
- Converter is not in the root wheel/sdist (`pyproject.toml` shared-data and the sdist include list), the `clio-kit` launcher, or CI (`.github/workflows/quality_control.yml`). Keep it out of the root release in this session, though a standalone CI job is fine (see step 6).
- Code worth carrying forward (all under `legacy/agentic-search/src/`):
  - `indexing/scientific.py`: unit canonicalization table (15 units), `Measurement`, `canonicalize_measurement`, `extract_measurements`, `encode/decode_measurements`, `normalize_formula`, `extract_formula_signatures`, `measurements_close`. Its `build_structure_aware_chunk_plan` and `_build_*_chunks` helpers are document chunking; search kept its own copy, so leave them out of the converter package. If you want the table-header unit trick (`Pressure (kPa): 150` → a `150 kPa` measurement), keep it as a small "extract quantities from a table cell plus header" function.
  - `retrieval/scientific.py`: `NumericRangeOperator`, `UnitMatchOperator`, `ScientificQueryOperators`, `score_scientific_metadata`. Reframe these as predicates over extracted quantities and formulas, not as retrieval scoring.
  - `evals/scientific.py`: `numeric_exactness` and `unit_consistency` (`precision_at_k` is a retrieval metric that stays in search; don't copy it).
  - `evals/quality_gate.py` and `tests/scientific/*`: these depend on search connectors. Keep them in `legacy/` as reference for a future integration test; do not port them now.
  - `tests/unit/test_formula_normalization.py` (14 pure tests) ports directly. `tests/unit/test_indexed_scientific.py` depends on DuckDB tables in search, so port only what tests pure functions.
  - `extracted_search_integration.txt`: how search used to call this code (CLI syntax `--numeric-range min:max:unit`, `--unit-match [value:]unit`, `--formula`; the API request models). Use it to shape the converter's own CLI and API inputs.

## Minimal target (definition of done)

1. `clio-converter/pyproject.toml` (hatchling, `src/` layout, package `clio_converter`, Python ≥ 3.11, same ruff/mypy-strict style as `clio-agentic-search/pyproject.toml`), plus `uv.lock`. Keep runtime dependencies at zero or near zero. FastAPI goes behind an optional `api` extra only if you add the HTTP service.
2. `src/clio_converter/` with small modules, for example `units.py` (canonicalization and conversion between compatible units), `quantities.py` (extract measurements from text and from table cells with header units), `formulas.py` (normalize and extract signatures), `predicates.py` (range, unit and formula matching over quantities and signatures), and `metrics.py` (`numeric_exactness`, `unit_consistency`). Fix imports and keep behaviour identical to legacy unless a legacy test proves a bug, and note any such fix.
3. A tiny CLI entry point `clio-convert` with, at minimum:
   - `clio-convert quantity "36 km/h" --to m/s` → `10 m/s` (plain text output, with JSON available via `--json`)
   - `clio-convert extract <file-or->` → JSON list of quantities and formula signatures found in the text
   - `clio-convert formula "P V = n R T"` → normalized signature
   Optionally add a FastAPI `POST /convert` and `POST /extract` behind the `api` extra. A runnable CLI is enough for "minimal runnable".
4. Tests under `clio-converter/tests/`: port `test_formula_normalization.py`, add unit and quantity tests (round-trip conversions, unknown unit → clear error, incompatible units → clear error), and add CLI smoke tests. Everything passes with `uv run pytest`, `uv run ruff check .`, `uv run ruff format --check .` and `uv run mypy src/`.
5. Update `clio-converter/README.md` with real usage. Mark which legacy files were ported and which remain reference-only in `MANIFEST.md`. Do not delete legacy files you haven't ported.
6. Optional, only if quick: a `converter-*` CI job in `quality_control.yml` that mirrors the agentic-search jobs and triggers on `^clio-converter/` changes. If you add it, check `tests/test_release_workflow.py` and the infrastructure change-detection regex for anything that has to stay consistent, and run the root tests (`uv run pytest tests/`, 158 passing at hand-off).

## Out of scope for this session (write them down as a roadmap section in the README instead)

A broad unit system (consider adopting `pint` rather than growing the hand-written table), currencies, uncertainty and significant figures, scientific file formats (HDF5, NetCDF, FITS, CSV schema and units), a service contract that search could call, packaging into the root `clio-kit` wheel and launcher (`clio-kit convert ...`), and an MCP server.

## Constraints

- Keep it small: the goal is a clean, runnable base, not features.
- Record consequential choices (package layout, CLI shape, whether to add the API now) as decisions with rejected alternatives.
- Report exactly which checks you ran and their results, and anything left unported.
