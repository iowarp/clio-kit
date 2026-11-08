# Session prompt: make clio-agentic-search the best agent search environment we can offer

Paste everything below the line into a new coding-agent session started at the clio-kit repo root.

---

You own `clio-agentic-search/` in the clio-kit monorepo (`/home/akougkas/iowarp/clio-kit`). Your goal is to make it the best search environment we can offer to AI agents: an agent asks a question about a corpus and gets back grounded, ranked, citable evidence fast, in a shape it can reason over cheaply.

The codebase was written about a year ago with much weaker models. Be skeptical of everything in it, including names, README claims, test thresholds and "SLOs". You have full freedom to redesign the algorithms, data model, interfaces, input breadth and output contracts. Preserving today's internals is not a goal. Preserving what is genuinely good (see "Keep") is.

## Where things stand (verified at hand-off)

- Scientific units, measurements, formula matching, scientific operators, scientific metrics and the scientific quality gate were **moved out** to `clio-converter/` (see `clio-converter/README.md`). Search must not import converter. Do not re-add unit or formula semantics to search; if search later needs them, design a service or plugin contract and record that as a decision.
- Removed inputs are rejected loudly: `POST /query` has `extra="forbid"` (an unknown field such as `scientific_operators` → 422), and the CLI no longer has `--numeric-range/--unit-match/--formula`. Old DuckDB files have the `scientific_*` tables dropped on connect.
- Check `git status` first. The separation may still be uncommitted on `main` (the branch was 1 ahead and 7 behind upstream at hand-off).
- Baseline: `cd clio-agentic-search && uv sync --locked --dev`, then `uv run ruff check .`, `uv run ruff format --check .`, `.venv/bin/python -m mypy src/ --ignore-missing-imports`, and `uv run pytest --ignore=tests/benchmarks` (106 passed, 0 failed). From the repo root, `uv run pytest tests/` gives 158 passed. CI runs exactly the same commands (`.github/workflows/quality_control.yml`, agentic-search jobs, Python 3.11–3.13) and rejects skipped tests (`scripts/assert_no_skipped_tests.py`).
- Environment traps: the `.venv/bin/*` shebangs point at an old path (`/home/akougkas/projects/...`), so `uv run mypy` fails to spawn. Use `.venv/bin/python -m mypy` or recreate the venv. `uv sync --all-extras` installs `sentence-transformers`, which makes two tests fail (`test_mvp_gaps::...test_no_results_prints_diagnostic`, `test_production_hardening::...test_falls_back_to_hash...`) because they assume the hash embedder. That is a test-isolation bug to fix, not a regression.
- `scripts/check_file_size.py` enforces an 800-line cap per file. `storage/duckdb_store.py` and the connectors are the largest files.

## What the current engine actually does (read the code to confirm; do not trust the README)

Pipeline: `retrieval/coordinator.py::RetrievalCoordinator._query_single_connector`. It finds which search capabilities a connector supports with `isinstance` checks on `Protocol` classes, then runs the lexical, vector and graph branches sequentially, merges by chunk id (max per score type), applies exact-match metadata filters, reranks and keeps `top_k`.

Known weaknesses, each observed in code:
1. **Lexical is not BM25**, although the README used to say so. Score = `overlap_count / len(query_tokens)` (`connectors/filesystem/connector.py::search_lexical`). There is no IDF, no term frequency and no length normalisation. The tokenizer is `[a-zA-Z0-9_]+`, lowercased, with no stemming and no phrase handling.
2. **The default "vector" search is a 16-dimensional token-hash bag** (`indexing/text_features.py::embed_text`, `HashEmbedder`, model id `hash16-v1`). The real model (`all-MiniLM-L6-v2`) is optional, and with the hash embedder, vector candidates are pre-filtered to the top lexical matches (`search_vector`), so the vector branch mostly re-scores lexical hits rather than adding new recall.
3. **Fusion sums incomparable scores**: `combined_score = lexical + vector + metadata` (`retrieval/capabilities.py`). The "reranker" just sorts by that sum (`retrieval/rerank.py`). There is no rank fusion and no cross-encoder.
4. **Multi-namespace results sort raw scores across different backends** (`query_namespaces`), so connectors whose scales differ are not comparable.
5. **Every query reindexes first** (`api/app.py` and `cli/main.py` call `connector.index(...)` before retrieval). Pagination (`offset/limit`) is applied after the `top_k` cut, so `total_count ≤ top_k`.
6. **Citations are thin**: the snippet is the first 160 characters of the chunk, with no matched span, no highlight and no reason the chunk matched. The score is a rounded opaque sum. Chunk offsets exist in storage but are not returned.
7. **Chunking**: structure-aware (sections, captions, tables, per-cell chunks with `#table=N&row=R&column=C`, equations; `indexing/structure.py`), with 400-character fixed windows only when no structure is found. There is no overlap, no parent/child hierarchy, and section chunks can be arbitrarily large. Only text files are read; there is no PDF, notebook, code-aware or data-file ingestion.
8. **Nothing is agentic**: there is no query planning, decomposition, multi-hop retrieval, iteration, feedback or answer-sufficiency signal, and no LLM anywhere in `src/`. Agents can only use it through HTTP or CLI. It is not an MCP server, although every other CLIO Kit tool is.
9. **Evaluation is weak**: the only quality checks were the scientific gate (now moved) on a 3-file corpus. Search has `evals/metrics.py::precision_at_k` and nothing else: no labelled benchmark, recall, nDCG or MRR. `tests/benchmarks/test_throughput.py` sets p95 < 200 ms at 10k chunks and ≥ 50 chunks/s indexing, but the 10k check runs only with `CLIO_ENFORCE_LARGE_SLO=1`.

Keep (these are good): namespace registry and connector abstraction; structure-aware chunking with cell-level citation URIs; the per-stage trace returned with each response (agents and developers both benefit); DuckDB storage with incremental file-index state; the job queue with cancellation; deterministic ordering (tests assert identical results across repeated queries and reindexing).

## Objectives

1. **Measure first.** Before changing ranking, build an evaluation harness that can prove improvement:
   - at least one public IR set relevant to research corpora (for example a BEIR subset such as SciFact, NFCorpus or SCIDOCS, small enough for CI or cached outside it) plus a hand-labelled CLIO corpus (repo docs, MCP server READMEs, a few scientific markdown files with tables);
   - metrics: Recall@k, nDCG@10, MRR, citation correctness (does the returned span support the query), p50/p95 latency, index throughput, and **agent cost**: tokens of payload needed to reach the answer.
   - Record the baseline numbers of today's engine before you replace anything, and keep a regression gate in CI with thresholds you justify.
2. **Retrieval quality**: real BM25 (DuckDB FTS or your own postings with IDF and length normalisation), real dense embeddings by default when available with a clearly reported fallback, rank fusion (for example RRF) instead of score sums, optional cross-encoder reranking, deduplication and diversity (e.g. MMR) across overlapping section, cell and table chunks.
3. **Agent-grade inputs**: richer query language (phrases, field and metadata predicates beyond exact equality, path/type/date filters, namespace selection), and ingestion breadth (PDF, Markdown, notebooks, source code with symbol-aware chunks, structured data such as CSV and Parquet headers). Chunking should be hierarchical (parent section ↔ child passage) with overlap.
4. **Agent-grade structured output**: per result, return the stable citation URI, the exact matched span with offsets, its parent context on request, which branches matched (lexical, vector, graph, metadata) with their ranks, a calibrated or at least monotone score, and a deduplication group. Per response, return coverage and "no good evidence" signals, plus a budget-aware mode (`max_tokens`) that packs the best evidence into the agent's token budget. Version the response schema.
5. **Agentic retrieval**: an optional loop for query decomposition, multi-hop follow-ups using retrieved entities, and query reformulation when evidence is weak, with every step visible in the trace. Keep the pure retrieval path deterministic and LLM-free. Make the agentic layer pluggable and testable with a fake model.
6. **Agent access**: expose search as an MCP server (tools such as `search`, `fetch_context`, `list_namespaces`, `index_status`) consistent with the other CLIO Kit servers, in addition to HTTP and CLI. Stop reindexing on every query: index explicitly or with a watcher, and report index staleness.
7. **Performance**: hold or beat the p95 < 200 ms target at 10k chunks on the new pipeline and report numbers at 100k chunks. Remove per-candidate storage round-trips (metadata filtering currently fetches metadata chunk by chunk).

## How to work

- Start by reading the code paths named above and writing a short design note in `clio-agentic-search/docs/` that lists: the current behaviour you confirmed, the target architecture, the response schema, the eval plan, and the phased order. Record each consequential choice (fusion method, embedding default, schema versioning, MCP surface) as a decision with rejected alternatives.
- Work in phases that each leave the suite green and add tests: eval harness and baseline → lexical and fusion → embeddings and rerank → output schema → ingestion breadth → MCP and agentic loop → performance.
- Do not quote improvements without eval numbers from the harness. Report the before and after tables. Name anything you could not measure and why.
- Breaking changes to the API and CLI are allowed (the package is marked Experimental), but update `clio-agentic-search/README.md`, `clio-kit-website/docs/agentic-search.md`, the root `README.md` Agentic Search section, the `search` command help in `src/clio_kit/__init__.py`, and `scripts/generate_docs.py` plus `clio-kit-website/src/data/mcpData.js` (the root test `tests/test_generate_docs_contract.py` checks the showcase entry).
- Keep `uv.lock` consistent (`uv lock --check` runs in CI) when you add dependencies. Put heavy model dependencies behind extras, and keep the core install and CI fast.
- Do not touch `clio-converter/`, except to read it for context.
