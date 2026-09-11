---
sidebar_position: 2
title: Marketplace and Contributions
---

# Marketplace features and acceptance

Use the checkout installation in the root README while this work remains on
`feat/360-meta-marketplace`. The public PyPI package and default GitHub branch
are not a coordinated release of this feature branch. A maintainer must review
and release that transition separately. The coordinated candidate version is
`2.11.0`; source installation is required until that version is published.

## Installable components

- 22 scientific/general MCP server plugins and six workflow bundles.
- 20 skills, available per workflow or together as `clio-skills`.
- `clio-agents`: a scientific workflow planner and an evidence reviewer.
- Direct external plugins and compiled external marketplace collections.

Skills use the [Agent Skills format](https://agentskills.io/specification),
independently of the client. Install them from the packaged CLI:

```bash
clio-kit skill list
clio-kit skill install --bundle clio-scientific-io --target /path/to/project/.agents/skills
```

Use `.agents/skills` for a Codex project, or another agent's documented discovery
directory. Required MCP servers are configured separately through the client's
stdio MCP settings. The existing `.claude-plugin` marketplace, dependency bundles
and two agent definitions currently target Claude Code. Portable skill support
does not imply that other clients accept those native manifests.

All 22 shipped scientific servers are Python projects. The launcher also
supports contributed Node/TypeScript and Go projects. Real MCP SDK fixtures
under `tests/fixtures/mcp-servers/` verify those adapters in CI; they are not
installed marketplace products or bundled wheel components. Hosting an
additional server requires its own reviewed source, locks and CI coverage.

## Supported and deployment-dependent paths

| Capability | Release scope |
|---|---|
| Python server packaging and stdio launcher | 22 shipped servers; individual backend prerequisites still apply |
| Portable skills | 20 standard skill folders; Codex discovery and Claude plugin loading are tested |
| Native bundles and agent definitions | Claude Code; other clients use portable skills and explicit MCP configuration |
| Node/TypeScript and Go | Locked local-project adapters tested with real SDK fixtures; no shipped non-Python scientific server |
| External plugins and marketplaces | Entry validation, snapshot compilation and client installation; third-party code remains externally maintained |
| GitHub contribution submission | Local entry/commit construction and API contract tests; acceptance does not open a public test PR |
| Web fetch | Requires an MCP task-capable client; document conversion needs its backend service |
| Scientific workflows and model behavior | Validate against the target data, site software, client and model; discovery is not a quality guarantee |

## Contributing and updating

For a standalone portable skill, use
`clio-kit skill validate /path/to/your-skill`. Put CLIO-specific frontmatter
values under standard `metadata`, with string values. The following authoring
commands scaffold and submit **Claude Code native plugins**:

```bash
clio-kit plugin init my-plugin --agent
clio-kit plugin validate my-plugin
claude plugin validate my-plugin --strict
clio-kit plugin submit my-plugin --repo my-lab/my-plugin --open-pr
```

For an MCP wrapper, supply `--mcp-command` and repeat `--mcp-arg`. Avoid
pointing an npm plugin entry at a raw MCP package: it must contain an actual
plugin manifest and `.mcp.json`. To launch a locked project packaged inside the
plugin, use `clio-kit server run ${CLAUDE_PLUGIN_ROOT}/server` in its config.

`--open-pr` requires the GitHub CLI, authentication, and Git commit identity.
Without it, submission only renders the entry; `--output` writes a local file.
No contribution PR is created during acceptance tests.

`clio-kit marketplace refresh --root .` merges indexed external collections.
It pins fetched relative sources to a commit and records provenance in
`.claude-plugin/federation.lock.json`. Conflicting names or malformed sources
fail before replacing the catalogue. The scheduled/manual GitHub workflow can
refresh this metadata without a package release. Ordinary generation reuses
the snapshot. Publishers must bump plugin versions for content updates; users
refresh the marketplace, update installed plugins, and reload the client.

Removal from a catalogue prevents new discovery; it does not forcibly remove
already installed code. Uninstalling a bundle can leave its dependencies;
`claude plugin prune` handles orphaned dependencies explicitly.

## Runtime and registry metadata

```bash
clio-kit server run /path/to/your/server
clio-kit server inspect /path/to/your/server --output /tmp/go-tools.json
clio-kit doctor --server hdf5 --connect
```

Inspection and connection checks require `clio-kit[verification]`. Registry
generation uses real stdio discovery for selected Python, Node and Go projects.
The descriptor's optional `[registry]` table describes the distribution package,
not its source language. Supported `registryType` values are `npm`, `pypi`,
`oci`, `nuget`, and `mcpb`. The publisher remains responsible for uploading the
referenced artifact and meeting that registry's ownership requirements.

## Reproduce the installed-system checks

```bash
uv sync --frozen --dev --extra verification
uv run --frozen pytest -q tests
uv run --frozen python scripts/verify_marketplace_install.py --all-servers --codex
```

Install Codex and Claude Code to run both client checks. The script builds a
source distribution and wheel, installs all 20 portable skills from that wheel,
and uses Codex's actual `skills/list` discovery API to verify they are enabled.
It also registers an isolated marketplace, installs bundles,
skills, agents and real external plugins, then exercises actual MCP
sessions. It launches the Go and TypeScript test fixtures through the installed
launcher and verifies cold/warm tool results, decompression,
independently calculated grouped means, and a plot of the transformed data.
`--all-servers` additionally initializes every shipped scientific MCP server.
Logs, tool responses, and generated artifacts are retained in the printed
output directory. It does not modify the user's client configuration.
`--skip-client` skips Claude-specific plugin checks; `--codex` independently
selects Codex discovery. These checks send no model requests. Discovery proves
that a client can load skills, not that a model follows them correctly.

## Scientific acceptance boundaries

A handshake proves a server speaks MCP; it does not prove its scientific
backends work. Full HPC workflows need Spack, Lmod, JARVIS packages and a working
scheduler. Chronolog needs its native client/service; ParaView needs a compatible
ParaView Python environment. Darshan diagnosis needs a genuine profile log.
The catalogue service needs `SCIENTIFIC_CATALOG_FILE` pointing to real metadata.

Pandas interpolation now fills interior numeric gaps linearly by row position;
endpoint and non-numeric gaps remain missing. HDF5 now labels sampled statistics
with actual coverage and omits cross-dataset totals when sampling is involved.
Forward/backward fill now follows row order for numeric and categorical data,
and mode fills use observed values with accurate fill counts. Parallel-sort
accepts bare and bracketed log levels consistently across filtering, statistics
and pattern detection. Slurm submission and allocation diagnostics use stderr
to preserve the MCP protocol stream. These fixes have targeted regressions;
installation and connection results alone still do not verify a scientific
workflow.

Model-driven skill trigger and quality evaluation must be recorded per agent
and model. A quota failure in one client says nothing about another client or account. Deterministic installation, discovery
and MCP checks do not require model quota. Record fresh behavioral evidence
separately for each tested client instead of treating historical results or
another client's results as proof of the current revision.

## Skill maintenance

The six workflow packages contain 20 skills. Public skill IDs remain stable;
titles describe the professional task, while descriptions identify when to use
it and distinguish adjacent workflows. Each skill has completion criteria and
an `evals.md` file. Names/descriptions are used for discovery; the full body loads
on invocation, following the [Agent Skills specification](https://agentskills.io/specification).

The current review checks tool names, argument shapes, state and file handoffs,
size limits, provenance and interpretation. Evaluation metadata is deliberately
`scenarios-recorded` until new model runs validate this revision. Historical
results remain labeled in the scenario files.

Specific limits matter: HDF5 aggregate statistics may sample data above 500 MiB.
A live 70-million-element array of ones returns sample sum/count 700,000, now
explicitly labeled as 1% coverage rather than full-dataset totals. Stream summaries can cover only part of a dataset; CSV profiles retain a bounded
sample. None should be presented as exact full-data calculations without checking
coverage. Skill instructions describe these limits instead of changing MCP code.
