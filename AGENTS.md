# Working on CLIO Kit

CLIO Kit is a meta-marketplace for scientific computing. Start with
[setup.md](setup.md) for installation, [docs/README.md](docs/README.md) for
human guides, and [docs/authoring.md](docs/authoring.md) to define a plugin,
MCP server, skill or hook. Detailed contribution rules are in
[CONTRIBUTING.md](CONTRIBUTING.md).

## Repository map

- `src/clio_kit/`: launcher, runtime isolation, skills and marketplace commands.
- `mcp-servers/<name>/`: independent server packages, descriptors, locks and tests.
- `skills/`: portable skill collections, including adapted Clio Coder skills.
- `plugins/`: native Claude Code plugins and workflow bundles.
- `community/entries/`: external plugin and marketplace TOML entries.
- `.claude-plugin/`: generated marketplace and federation snapshot.
- `docs/`: documentation source shared by GitHub and Docusaurus.
- `website/`: website configuration, components and static assets.

## Choose the contribution route

Use the product distinction consistently: MCPs, skills, agents and hooks are
components; plugins bundle components for a workflow. Native Claude Code wrappers
are installation packages and must not inflate workflow-plugin counts. The
catalogue derives maintained types from contents and dependencies; do not guess
external component types from package names.

For repository-owned component packages, add a named folder under `plugins/`,
`skills/`, `agents/` or `hooks/`, with `.claude-plugin/plugin.json` and the needed
components. Website start/build and CI sync both catalogues automatically;
discovery needs no central TOML entry or manual generator commands. CI syncs the
catalogues before the root suite; do the same locally with
`clio-kit marketplace sync --root .`, which is also what native client testing
before a build needs.
Validate the package, then run `scripts/verify_local_components.py --output`
with a new temporary directory for native installation and usage checks.

For external plugins, run `clio-kit plugin init my-plugin`, edit the generated
components, then `clio-kit plugin validate my-plugin`. Add `--hook` to scaffold
a read-only SessionStart command hook. Follow the authoring guide for MCP
configuration, skill metadata and evaluation scenarios. Generate a community
entry with `clio-kit plugin submit my-plugin --repo owner/name`; `--open-pr`
performs the separate public submission action.

For a maintained server, add it under `mcp-servers/` with its runtime descriptor,
lock file and tests, and register it in `mcp-server-versions.toml`. Python, Node
and Go have different build prerequisites; use the contributor guide's runtime
contract. Keep each server's dependencies isolated from the launcher.

Portable skills do not configure MCP servers. Native plugin manifests, agents
and hooks use their host's format; do not imply universal client compatibility.
Validate hook configuration without executing it, then exercise the hook in an
isolated supported client and check its effects. Indexing an external plugin
does not certify its commands or future updates.

## Verification

From the repository root:

```bash
uv sync --frozen --all-extras --dev
uv run --frozen pytest tests -q
uv run --frozen ruff check scripts src tests evals
uv run --frozen ruff format --check scripts src tests evals
uv run --frozen mypy src --ignore-missing-imports
uv run --frozen python scripts/check_file_size.py
```

Run a changed server's own suite from `mcp-servers/<name>/` using its own locked
environment. Root pytest targets `tests/`; recursive collection mixes independent
server dependencies. For launcher or packaging changes, also run
`uv run --frozen python scripts/verify_marketplace_install.py --all-servers`.
For distribution changes, also run `uv run --frozen python scripts/verify_partial_install.py --output /tmp/clio-partial-install` with a new output directory.
For hooks, run `uv run --frozen python scripts/verify_plugin_hooks.py --output /tmp/clio-hook-acceptance` with a new output directory.
These acceptance scripts create temporary client configurations and evidence.
Check [acceptance boundaries](docs/marketplace.md#scientific-acceptance-boundaries)
before interpreting a successful connection as scientific workflow validation.

For documentation changes, run `npm --prefix website run build` after
`npm --prefix website ci` if dependencies are absent. Server references
and catalogue data regenerate with
`uv run python scripts/generate_docs.py mcp-servers website`.
Keep reviewed usage inside the generator's existing usage markers.

Preserve upstream provenance and hashes when updating imported skills; use
`scripts/import_clio_coder_skills.py` rather than silently changing locked imports.
Keep credentials, local `.clio-coder` sessions and temporary acceptance output
out of commits. Report what was exercised, skipped or blocked without claiming
that static validation proves live agent behavior.
