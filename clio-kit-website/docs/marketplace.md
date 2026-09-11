---
sidebar_position: 2
title: Marketplace and Contributions
---

# Marketplace features and acceptance

Follow [Getting Started](./intro.md) to install the launcher, register the
marketplace and configure your agent. Source installation uses one checkout
for the launcher and catalogue; publishing a package is a separate release step.

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

See [Agent integrations](./intro.md#agent-integrations) for Codex, Claude Code,
Cursor, VS Code / GitHub Copilot, Antigravity, and Claude Desktop.
That guide lists the client-specific MCP schemas and skill directories, with
complete scientific I/O configuration examples. VS Code Copilot and the Codex
extension use separate MCP configurations.

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
| GitHub contribution submission | Regression tests plus a manually verified PR in a contributor-owned fork; automated acceptance does not create public PRs |
| Web fetch | Ordinary calls and optional task execution; document conversion needs its backend service |
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
not its source language. Registry server patch versions also advance for runtime
and dependency fixes, even when tool schemas stay compatible. This release
selects all 22 servers so their registry entries point to the patched `2.11.0`
wheel. Supported `registryType` values are `npm`, `pypi`,
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

Model evaluation is specific to the client, model and task. The September 10
follow-up ran all 20 skills through Codex with real attached MCP services and
retained their traces and outputs. These are one-scenario behavioral checks,
not comparative quality scores or coverage of every workflow. A separate
23-case skill-selection test matched 22 expected choices; the ambiguous slow-HDF5
request selected the large-dataset skill instead of storage-format advice.
Claude model calls were quota-blocked in this follow-up; Cursor requested login,
and Antigravity's GUI workflow was not automated. These results do not establish
behavioral support in those clients. Native plugin installation and deterministic
checks do not require a model quota.

## Skill maintenance

The six workflow packages contain 20 skills. Public skill IDs remain stable;
titles describe the professional task, while descriptions identify when to use
it and distinguish adjacent workflows. Each skill has completion criteria and
an `evals.md` file. Names/descriptions are used for discovery; the full body loads
on invocation, following the [Agent Skills specification](https://agentskills.io/specification).

The current review checks tool names, argument shapes, state and file handoffs,
size limits, provenance and interpretation. The `scenarios-recorded` metadata identifies scenario definitions, not a
quality certification. The dated behavioral checks above provide separate,
limited evidence; retain the client, revision and outcome when repeating them.

Specific limits matter: HDF5 aggregate statistics may sample data above 500 MiB.
A live 70-million-element array of ones returns sample sum/count 700,000, now
explicitly labeled as 1% coverage rather than full-dataset totals. Stream summaries can cover only part of a dataset; CSV profiles retain a bounded
sample. None should be presented as exact full-data calculations without checking
coverage. Skill instructions require checking these coverage labels.

## Recorded validation, 2026-09-10

The tested checkout installed 22 MCP servers, six bundles and 20 portable skills.
The installed-wheel acceptance script passed Claude plugin installation/update,
Codex skill discovery, all 22 stdio handshakes, TypeScript/Go cold and warm calls,
compression round-trip, grouped means and plotting, Pandas imputation,
bracketed log filtering, plain-client Web fetch and sampled HDF5 coverage. A separate real Slurm job
completed with exact expected output and valid JSON-RPC stdout. The setup guide's
source installation, HDF5 connection and compression commands also passed.

All 22 server dependency environments, the launcher and agentic-search passed
`pip-audit` after updating HTTP dependencies. Locally built packages and the
Git-hosted JARVIS dependency are not covered by PyPI advisory matching.
The website build has an unpatched upstream image-parser advisory with a
pre-build format restriction; see the
[website maintenance notes](https://github.com/iowarp/clio-kit/blob/main/clio-kit-website/README.md#image-parser-advisory).
This is a disclosed build dependency limitation, not a clean npm audit.

Use the GitHub Actions results for the exact commit being reviewed. Public
publication and target-site backend acceptance remain separate steps; the
checks above do not establish universal agent or scientific-workflow support.


## MCP SDK v2 migration, 2026-09-10

All 22 Python servers lock MCP SDK 2.2.0 and stable FastMCP 4.0.3. The shared
verification client follows the [official SDK migration guide](https://py.sdk.modelcontextprotocol.io/migration/)
and [FastMCP upgrade guide](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3).
SDK versions and wire protocol versions are separate: modern connections use
`2026-07-28`, while legacy clients can still negotiate `2025-11-25`.

HDF5 `export_dataset` accepts `export_format` (`csv`, `json`, or `numpy`).
Modern connections default to JSON without asking a mid-call question; legacy
clients retain elicitation when the argument is omitted. HDF5 operational logs
use standard stderr logging. Existing JARVIS user-tool schemas are unchanged;
stable FastMCP no longer infers output schemas for its untyped admin-tool lists.

Validation includes fresh-wheel installation, all 22 server tool inventories in
both protocol eras, and an actual SDK v1.30.0 client connecting to all 22 servers.
Real tool workflows cover ADIOS, arXiv, compression, Geo, HDF5, NDP, node hardware,
Pandas, parallel-sort, Parquet, Plot, scientific catalogue, seismology, Slurm,
terrain and Web. Checks include actual file contents and generated images;
arXiv also returned the expected paper from its public API. TypeScript and Go
launcher fixtures still return the expected results after cold and warm starts.

The native-backend follow-up also passed real installed-wheel workflows:

| Backend | Observed result |
| --- | --- |
| Darshan 3.5.0 | A native instrumented workload produced a log with 4 reads, 4 writes, and 16,384 bytes in each direction; MCP totals matched the native parser. |
| Lmod 8.6.19 | All seven tools ran against real modulefiles, including saving a collection and restoring it in a new MCP process. |
| Spack 0.21.2 | Built zlib-ng 2.1.4, located its exact prefix, and confirmed reuse using a qualified package/hash spec. |
| JARVIS 1.8.1 | Created and executed the built-in echo pipeline and inspected its completed execution and stdout. |
| ParaView 6.0.0 | Connected to native pvserver, created a sphere, computed surface area, and rendered a PNG through Xvfb. |
| ChronoLog | Ran visor/keeper/grapher/player with the Python 3.11 native client and recovered the exact quoted, multiline interaction from its HDF5 archive. |

These tests exposed and fixed Lmod shell integration, ParaView render-thread
affinity, native Darshan text parsing, and ChronoLog reader build/retrieval bugs.
Darshan request-size estimates now label their basis and use bounded memory;
subsecond job timing remains precise. ChronoLog's native suite includes the
previously skipped archive round trip.

The broader follow-up exercised every default tool at least once, including
all 26 ParaView tools, two-rank Darshan MPI-IO on one host, real SSH collection,
Slurm submission/cancellation, arXiv PDF downloads and NDP CSV staging. This
counts attempted tool coverage; it does not mean every operation or parameter
passed. Web's optional remote conversion-events service was unconfigured.
The community submission function also created a real one-file PR in the
contributor's existing renamed fork; the test PR was closed without merging.

These checks fixed tool pagination that hid later tools from Codex, ADIOS stdout
contamination, ParaView first-file loading/presets/histograms/image previews,
remote SSH command quoting, and Pandas hypothesis serialization and string-pattern
validation. Pandas no longer calls p-value thresholds effect sizes; its result
explicitly says `not_computed`. Node Hardware's health endpoint reports process
availability, with unmeasured hardware/security indicators labeled accordingly.

Remaining release and deployment limits:

- JARVIS 1.8.1 scheduler execution failed when Slurm supplied the literal
  `SLURM_CLUSTER_NAME=(null)`. An audit-only backend copy with that handling fixed
  completed job 14. The dependency shipped by CLIO is unchanged and still needs
  an upstream fix and release; the patched-copy result is not a product pass.
- This machine's ParaView build exposed one data partition under a two-process
  MPI launch. Distributed rendering needs an MPI-enabled build and another test.
- Darshan's timeline provides summary duration, not event-level peak/idle phases.
  Multi-node scaling, production fault recovery and arbitrary external recipes
  remain unverified.
- An empty-cache install exceeded the acceptance test's 300-second limit while
  downloading the Pandas stack. Dependency installation succeeded on retry.
  Prepare large servers before starting an agent with a short startup deadline;
  see the [setup guide](https://github.com/iowarp/clio-kit/blob/main/setup.md).
- Native bundle/agent manifests remain Claude Code-specific. Portable skills
  and explicit MCP registrations are the supported path for other clients.
- The website's upstream image-parser advisory remains unpatched; the build
  guard restricts affected formats, but this is not a clean dependency audit.

To repeat installed-wheel acceptance with the verification extra installed:

```bash
uv run --frozen python scripts/verify_marketplace_install.py --all-servers --protocol-mode 2026-07-28 --output /tmp/clio-v2-modern
uv run --frozen python scripts/verify_marketplace_install.py --all-servers --protocol-mode legacy --skip-client --output /tmp/clio-v2-legacy
```

Use `--skip-client` on systems without Claude Code; MCP checks require no model
credentials. Detailed per-server logs belong with the local audit evidence,
not in the package or website assets.
