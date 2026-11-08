---
title: Agentic Search
description: "Hybrid retrieval engine with lexical, vector, graph, and metadata search over namespaced document corpora. Part of CLIO Kit (IoWarp Platform)."
---

# Agentic Search

Hybrid retrieval engine for AI agents. Indexes documents into namespace-specific backends and supports lexical, vector, graph, and metadata retrieval in one pipeline, with structure-aware chunking and table-cell citations.

**Type**: Standalone FastAPI service (not an MCP server)

## Features

- **Multi-namespace registry** with runtime/auth config bundles
- **Connectors**: local filesystem, S3-compatible object store, Qdrant vector store, Neo4j graph, Redis KV — all backed by DuckDB
- **Structure-aware chunking**: sections, captions, tables, table cells, equations, with cell-level citation URIs
- **Background indexing** with async job queue, cancellation tokens, serialized per-namespace execution
- **Retry wrappers** with exponential backoff
- **Telemetry**: OpenTelemetry tracing, Prometheus metrics at `/metrics`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Liveness probe |
| GET | `/version` | Package version |
| GET | `/documents` | List indexed documents and chunk counts |
| POST | `/query` | Run retrieval, return citations + trace events |
| POST | `/jobs/index` | Submit async index job |
| GET | `/jobs/{job_id}` | Fetch job status/result |
| DELETE | `/jobs/{job_id}` | Request cancellation |
| GET | `/metrics` | Prometheus text exposition |

## Quick Start

```bash
cd clio-agentic-search
uv sync --all-extras --dev

# Index a namespace
uv run clio index --namespace local_fs

# Query
uv run clio query --namespace local_fs --q "reactor pressure calibration"

# Run API server
uv run uvicorn clio_agentic_search.api.app:app --reload
```

## CLI Commands

- `clio query` — Run retrieval queries
- `clio index` — Index documents into a namespace
- `clio list` — List indexed documents
- `clio seed` — Seed sample data
- `clio serve` — Start the API server

## Quality Checks

```bash
uv run ruff check .
uv run ruff format --check .
uv run mypy src/
uv run pytest --ignore=tests/benchmarks
```
