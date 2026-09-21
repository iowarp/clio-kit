---
title: Reviewing the marketplace rework
---

# Reviewing the marketplace rework

Review the branch in these groups. Renames, imported content and authored behavior
need different checks; the combined diff alone obscures the changes that affect
installation and execution.

| Review area | Source | What to check |
| --- | --- | --- |
| Server directory move | `mcp-servers/`, build paths and CI | Use Git rename detection; distinguish moved code from actual server fixes |
| Imported Clio Coder skills | `skills/clio-coder-skills/import-lock.json`, `scripts/import_clio_coder_skills.py` | Upstream revision, adaptations, hashes and resource independence |
| Discovery and composition | `local_plugins.py`, `workflow_plugins.py`, `federation.py` | Folder validation, dependency resolution, external provenance and host boundaries |
| Selective distribution | `component_store.py`, `release_components.py`, `scripts/package_components.py` | Selected downloads, integrity, matching source/release resolution and publication order |
| Client installation | `client_install.py`, `install_transaction.py` | Supported configuration formats, conflict handling, idempotence and rollback |
| Cache lifecycle | `component_cache.py` | Keep installed catalogue and project references; retain untracked legacy entries |
| Website | `scripts/generate_website_catalogue.py`, `scripts/generate_docs.py`, `clio-kit-website/src/components/Marketplace/` | Generated client settings/icons, valid memberships and missing featured entries |

Python module names above are under `src/clio_kit/` unless a path is given.
Shared resources inside packed imported skills are intentionally copied so a
single skill can be installed independently. The importer and regression suite
check their consistency; update them through the importer, preserving provenance.

## Repeatable checks

The repository declares checks in `AGENTS.md` and `.github/workflows/`.
No tool-specific local session configuration is needed to run them.

```bash
uv sync --frozen --all-extras --dev
uv run --frozen pytest tests -q
uv run --frozen ruff check scripts src tests evals
uv run --frozen ruff format --check scripts src tests evals
uv run --frozen mypy src --ignore-missing-imports
uv run --frozen python scripts/check_file_size.py
npm --prefix clio-kit-website run build
```

Use a fresh output directory for each acceptance run:

```bash
uv run --frozen python scripts/verify_marketplace_install.py --all-servers --output /tmp/clio-review-marketplace
uv run --frozen python scripts/verify_partial_install.py --output /tmp/clio-review-partial
uv run --frozen python scripts/verify_local_components.py --output /tmp/clio-review-components
```

These build and install the distribution, exercise MCP calls and component
installation, and retain evidence. Client binaries and scientific backends have
separate prerequisites. Run changed servers' own tests in their locked server
environments. Root pytest deliberately targets `tests/`.

A passing connection proves transport and discovery, not every scientific
operation or live model workflow. Read the
[acceptance boundaries](marketplace.md#scientific-acceptance-boundaries) and the
[distribution publication requirements](installation.md#verification-and-publication)
before interpreting the results as release readiness.
