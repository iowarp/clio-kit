# clio-converter

> **Status: Pre-alpha, not runnable yet, not packaged or released.** This folder holds the scientific-format work split out of `clio-agentic-search`. Nothing in the CLIO Kit wheel, launcher, or CI depends on it yet.

Part of [**CLIO Kit**](../README.md).

## Purpose

A service that agents (and `clio-agentic-search`) can call to convert and normalize scientific content:

- **Units and quantities**: parse a measurement in text (`250 kPa`, `36 km/h`), canonicalize it to SI, and compare values across units.
- **Formulas**: extract equations (LaTeX `$...$`, `$$...$$`, plain text like `E = mc^2`) and normalize them into comparable signatures.
- **Scientific predicates**: numeric-range, unit-match, and formula-equivalence checks over extracted quantities.
- **Format conversion** (planned, much broader): scientific file and data formats, such as tables and units in CSV/HDF5/NetCDF/FITS metadata.

## What is here now

Everything currently lives in [`legacy/agentic-search/`](legacy/agentic-search/MANIFEST.md): the original modules and tests, moved with `git mv` so their history is kept, plus `extracted_search_integration.txt`, which captures the code that wired them into search. The imports still point at `clio_agentic_search.*`, so none of it runs as-is.

Getting to a minimal runnable state is tracked in [`../handoffs/clio-converter-bootstrap.md`](../handoffs/clio-converter-bootstrap.md).

## Boundary with clio-agentic-search

- Search keeps document-structure chunking (sections, captions, tables, table cells with `#table=N&row=R&column=C` citation fragments, equation blocks) and generic retrieval metrics (`precision_at_k`).
- Converter owns unit tables, measurement extraction, formula normalization, scientific operators, scientific metrics (`numeric_exactness`, `unit_consistency`), and the scientific quality gate.
- Search does not import converter. Any future integration goes through a service or plugin contract that has not been designed yet.
