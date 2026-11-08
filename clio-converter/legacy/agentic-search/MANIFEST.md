# Legacy material from clio-agentic-search

These files were moved out of `clio-agentic-search` at commit `8fc2e7ef` (parent of the separation change). Files moved whole used `git mv`, so `git log --follow <path>` shows their full history. They are **verbatim**: imports still say `clio_agentic_search.*` and nothing here runs as-is.

## Modules (`src/`)

| File here | Original path | Contents |
|---|---|---|
| `src/indexing/scientific.py` | `clio_agentic_search/indexing/scientific.py` | `_UNIT_CANONICALIZATION` (15 units: m, kg, s, Pa, m/s families), `Measurement`, `canonicalize_measurement`, `extract_measurements` (regex), `encode/decode_measurements` (`unit\|value\|raw_unit\|raw_value;...`), `normalize_formula` (side-sorting, commutative factor sorting, `^{2}`/`**2` → `^2`), `extract_formula_signatures` (`$$`, `$`, plain-text equations), `measurements_close`. **Also** the original structure-aware chunker (`build_structure_aware_chunk_plan`), which adds `scientific.measurements` / `scientific.formulas` chunk metadata and appends header units to numeric table cells. Search now uses its own copy without those parts (`clio_agentic_search/indexing/structure.py`). |
| `src/retrieval/scientific.py` | `clio_agentic_search/retrieval/scientific.py` | `NumericRangeOperator`, `UnitMatchOperator`, `ScientificQueryOperators`, `score_scientific_metadata` (range +1.2, unit +1.0, formula +1.4; any operator miss → 0). |
| `src/evals/scientific.py` | `clio_agentic_search/evals/scientific.py` | `precision_at_k` (also kept in search as `evals/metrics.py`), `numeric_exactness`, `unit_consistency`. |
| `src/evals/quality_gate.py` | `clio_agentic_search/evals/quality_gate.py` | 6-check scientific gate over a 3-file reactor corpus (precision@3 ≥ 0.8, numeric_exactness ≥ 0.9, unit_consistency = 1.0, two negative controls). It passed 6/6 on the last run before the move. It needs search's `FilesystemConnector` + `RetrievalCoordinator` with the scientific branch, which no longer exists. |

## Tests (`tests/`)

| File here | Tests | Notes |
|---|---|---|
| `tests/unit/test_formula_normalization.py` | 14 | Pure functions only; easiest to make runnable first. |
| `tests/unit/test_indexed_scientific.py` | 5 | Need the DuckDB `scientific_measurements` / `scientific_formulas` tables. |
| `tests/scientific/test_hardening.py` | 13 | Noisy multi-file corpus, quality gate, negative controls, determinism, mixed units. |
| `tests/scientific/test_phase3_scientific_retrieval.py` | 3 | Structure + measurement extraction, operators, reproducible metrics. |
| `tests/scientific/test_realistic_retrieval_scenarios.py` | 3 | Mixed corpora, formula targeting, S3 object-store cell citations. |

All 38 passed in the pre-separation search suite (138 total).

## `extracted_search_integration.txt`

These are verbatim, commented-out excerpts of code removed from modules that stayed in search, each labelled with its original file and line range at `8fc2e7ef`:

- `search_scientific` on the filesystem and S3 object-store connectors (DuckDB pre-filter on range/formula, then metadata scoring)
- DuckDB schema, indexes, write path, range/formula queries and delete cascades for `scientific_measurements` and `scientific_formulas`
- the `StorageAdapter` protocol methods and the `ScientificSearchCapable` capability
- the coordinator's scientific branch and hard-filter step (when operators were active, only chunks matched by the scientific branch survived)
- the API request models (`ScientificOperatorsRequest`, …) and the CLI flags/parser (`--numeric-range min:max:unit`, `--unit-match [value:]unit`, `--formula`)
- the two CLI tests that were removed from `tests/unit/test_cli.py`
