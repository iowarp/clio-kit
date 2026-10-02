---
sidebar_position: 2
title: Marketplace and Contributions
---

# Marketplace features and acceptance

CLIO Kit is a meta-marketplace: it combines its own MCP servers, skills,
workflow plugins, agents and hooks with contributions from external repositories and marketplaces.

Follow [Getting Started](./intro.md) to install the launcher, register the
marketplace and configure your agent. Source installation uses one checkout
for the launcher and catalogue; publishing a package is a separate release step.

Release installations download selected components separately from the small launcher;
see [selective installation](installation.md) for the no-clone routes and checks.

## Installable components

See [Choosing and composing plugins](plugins.md) for the six primary bundles
and optional task plugins that reuse components across them.
Repository-owned packages in `plugins/`, `skills/`, `agents/` and `hooks/` are
discovered during generation without a central TOML entry; see
[folder contributions](authoring.md#add-a-component-folder-to-clio-kit).

- Individual scientific/general MCP servers and six broad workflow plugins (bundles).
- `clio-dataset-report`: a task plugin combining HDF5/Pandas/Plot, a report skill,
  evidence review and a verification hook; see [inputs and limits](plugins.md#scientific-dataset-report).
- Scientific workflow skills, available per workflow or together as `clio-skills`, plus the optional imported `clio-coder-skills` collection.
- `clio-agents`: a scientific workflow planner and an evidence reviewer.
- Direct external plugins and compiled external marketplace collections.

MCPs, skills, agents and hooks are components; plugins bundle them for a workflow.
Skill and agent collections remain separate from workflow plugins. Claude Code
uses native plugin packages to distribute both. Catalogue filters reflect the
[component model](plugins.md#catalogue-types-and-installation-packages), not the
number of entries in Claude Code's `plugins` array.

Skills use the [Agent Skills format](https://agentskills.io/specification),
independently of the client. Install them from the packaged CLI:

```bash
clio-kit skill list
clio-kit skill install --bundle clio-scientific-io --target /path/to/project/.agents/skills
```

Use `.agents/skills` for a Codex project, or another agent's documented discovery
directory. Required MCP servers are configured separately through the client's
stdio MCP settings. The shared project installer resolves the same maintained,
external and federated packages for each supported client. Claude retains its
native marketplace export; Codex and OpenCode have explicit agent/hook adapters.
See [component support](clients.md#component-support) for the limits.

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

## Clio Coder integration

CLIO Kit includes an optional collection imported from Clio Coder revision
`c841a46101d6d9df5fd3bcb1d337e59d92fb660d`: 33 coding/research skills and
six Materio skills. The original scientific workflow bundles remain unchanged.

```bash
clio-kit skill list --bundle clio-coder
clio-kit skill install --bundle clio-coder --target .agents/skills
clio-kit skill install clio-kit-scientific-debugging --target .agents/skills
```

Choose either the collection or individual skills. Through Claude Code:

```bash
claude plugin install clio-coder-skills@clio-kit
```

The collection also includes a portable root plugin manifest. With Clio Coder
0.4.8, from a working project, install the collection from your kit checkout:

```bash
clio-coder library install /path/to/clio-kit/skills/clio-coder-skills --project
clio-coder library pin clio-coder-skills --project
```

Collection version 1.1.0 gives adapted skills a `clio-kit-` prefix, such as
`clio-kit-scientific-debugging`. Clio Coder compares skill names against its
audited catalogue; using the original name for changed content caused a drift
warning. Distinct names preserve that distinction without disabling integrity
checks. Original names, revisions and file hashes remain in `import-lock.json`.

If you installed the earlier unprefixed collection, review local edits and remove
only those old imported copies before reinstalling. The portable installer does
not delete renamed folders automatically. Native plugin users should update or
reinstall `clio-coder-skills`, then restart their session. Upstream marketplace
package names below remain unchanged.

The imported instructions retain upstream attribution and license declarations.
Each independently installed Materio skill includes its shared references,
templates and scientific helper scripts. Native Clio agents, prompt commands,
fleets and runtime gates are not provided by those skill folders. Host-specific
instructions are identified by a compatibility note. Upstream evaluation labels are preserved
as provenance; the kit does not reinterpret them as its own successful tests.

Install a referenced companion skill too, or select the complete collection;
for example, `clio-kit-tech-spec` uses `clio-kit-tdd`. Herdr requires its executable
and an actual Herdr pane (`HERDR_ENV=1`); installing the skill does not start a
session. Its instructions distinguish JSON responses from actions that succeed
with empty output. Archify likewise requires its separate renderer.

### Upstream marketplace and packed plugin

The community entry `iowarp-clio-coder.toml` federates the pinned Clio Coder
marketplace. Its upstream packages are an alternative to the adapted collection:

```bash
claude plugin install scientific-debugging@clio-kit
claude plugin install materio@clio-kit
```

Avoid installing upstream and adapted copies of the same skill together.
The upstream `materio` marketplace entry exposes six portable skills. Its native
agents, prompts and fleet belong to the complete Clio Coder package:

```bash
clio-coder library install /path/to/clio-coder/library/plugins/materio --project
```

Materio also supplies `assets/scripts/project_plugin.py` for generating native
peer exports. Its optional `wtf-p` action prepares a handoff; it does not install
an MCP server or execute the receiving paper workflow. Foreign MCP/hook import
is omitted by Clio Coder's interoperability importer, so installing these skills
does not establish native MCP integration in that host.

### Refresh and validation

From the CLIO Kit checkout, reproduce the import using a clean Clio Coder
checkout at the recorded revision:

```bash
uv run --frozen python scripts/import_clio_coder_skills.py /path/to/clio-coder
clio-kit marketplace sync
```

`skills/clio-coder-skills/import-lock.json` records original metadata, source
hashes, adaptations and hashes of imported files. Refresh refuses local changes
and an unexpected source revision. A deliberate upstream upgrade uses
`--revision <reviewed-commit> --version <new-collection-version>` and updates the
community entry's pin separately. Review both diffs; clients use the collection
version for plugin cache identity.

The external transport acceptance test builds isolated Git repositories and a
temporary npm registry. It tests GitHub, Git URL, Git subdirectory, npm, and
GitHub/URL marketplace federation with a real imported skill and an MCP query:

```bash
uv run --frozen python scripts/verify_external_contributions.py --output /tmp/clio-external-check
uv run --frozen python scripts/verify_imported_skill_behavior.py --output /tmp/clio-skill-behavior
```

Use new output directories. The first test needs Claude CLI but no model login;
its synthetic GitHub URLs are redirected to temporary repositories. A separate
live check installed `scientific-debugging` and `materio` from the pinned public
repository. The second test needs authenticated Codex model access and checks
skill loading, an actual MCP query, preserved source files, and written artifacts.
Read its diagnosis and protocol before accepting scientific claims.

Installation and discovery cover all imported skills. The 2026-09-16 follow-up
exercised each of the 39 imported procedures in a bounded task, then tested
previously missing live paths. These checks establish the listed behavior, not
universal compatibility or scientific correctness for arbitrary inputs.

| Check | Verified behavior |
|---|---|
| Package and installers | A fresh wheel installed all 39 adapted skills with matching resource bytes. The npm skills CLI discovered and copied all 60 skills, refreshed resources and removed them for Codex, Claude Code and Antigravity targets. |
| External contributions | All six isolated transport routes installed, queried an actual MCP, refreshed and uninstalled successfully. A real one-file contribution [PR on the contributor's fork](https://github.com/SIslamMun/scientific-mcps/pull/2) was opened, inspected and closed without merging. |
| Claude plugin hooks | Default, custom-file and inline hook validation is covered. Installed hooks ran at session start, observed an allowed write and denied a protected write. Both the CI-pinned client's scripted-model run and an authenticated model run passed. |
| Codex 0.154.0 | Loaded the renamed debugging and experiment-protocol skills, queried a numerical MCP and wrote evidence/protocol artifacts while preserving source files. |
| Claude Code 2.1.269 | Loaded the imported literature skill through its plugin, fetched an actual arXiv abstract and full PDF, and wrote a source-linked review with retrieval records. |
| Antigravity CLI 1.2.4 | Loaded the renamed debugging skill and reproduced/diagnosed the numerical failure while preserving source files. |
| Herdr 0.8.0 | From a real pane in an isolated session, created a sibling pane, ran a command, waited for and read its output, preserved focus and cleaned up. |
| Clio Coder 0.4.8 | Discovered and activated the adapted debugging skill without an upstream audit-hash collision. Generated diagnostics still require evidence review; a causal explanation needed correction. |
| Native Materio on Clio Coder 0.4.8 | Its executor and read-only verifier completed a protocol-preparation task within their write boundaries. No physical measurements were performed. |

Use the current CI result for the root test suite. Materio's earlier offline helper suite passed
92 tests and its generated Claude export passed strict validation. One native
Clio model run exceeded the test timeout; resuming that interrupted session
returned an upstream tool-history error. A fresh bounded run completed. Drafting and
interview skills remain bounded evaluations; real publication, independent review
councils and experiments require their own inputs and authorization. The exact
`wtf-p` receiving workflow and desktop/IDE GUI flows are not verified by these
CLI checks. Cursor remains outside this follow-up's scope.

## Optional skills CLI

The [open-source `skills` CLI](https://github.com/vercel-labs/skills) is an
alternative way to install CLIO's existing skill folders. Version `1.5.25`
requires Node.js 22.20.0 or newer. The Python installer above remains available
without Node.js and downloads only the selected skills from the release catalogue.

From your working project, use the absolute path to your CLIO checkout:

```bash
npx skills@1.5.25 add /path/to/clio-kit --list
npx skills@1.5.25 add /path/to/clio-kit --skill dataset-explore large-data-read storage-format --agent codex --copy
```

Use `--agent claude-code` or `--agent antigravity` for those clients; multiple
agent names can follow `--agent`. Use `--skill '*'` for all available skills, including the Clio Coder collection. Codex and
Antigravity share `.agents/skills`; Claude Code uses `.claude/skills`. Other
agents that read the shared directory will also discover those skills.
Antigravity CLI still needs the project selection described in
[Getting Started](./intro.md#agent-integrations).

The CLI also accepts `iowarp/clio-kit` as a GitHub source for skills on its
default branch. A checkout source keeps skills aligned with a source-installed
launcher. Neither route registers MCP servers, installs bundle dependencies,
or installs agent definitions. Continue with the MCP setup in Getting Started.

To refresh, update your source checkout to the reviewed revision and repeat
the same `add` command with the same agent names and `--copy`. Review local
edits first: refreshing replaces installed files. This explicit route avoids
`skills@1.5.25 update` leaving another agent's copied directory stale. Choose
one installer to manage each installed skill; avoid duplicating a native
Claude plugin's skills with a separate portable installation.

To remove one skill from the project, including its shared copy:

```bash
npx skills@1.5.25 remove dataset-explore --yes
```

Omit agent filters for complete removal; this also affects other agents reading
the shared copy. The pinned version rejects `remove --agent '*'`.
Set `DISABLE_TELEMETRY=1` to opt out of upstream telemetry. A source reference
and content hash in `skills-lock.json` do not make a moving branch immutable.

CI checks discovery, complete installed file contents, explicit refresh after a
source change, and removal in temporary projects with the pinned CLI. To repeat:

```bash
npm install --global skills@1.5.25
uv run --frozen python scripts/verify_skills_cli.py
```

## Supported and deployment-dependent paths

| Capability | Release scope |
|---|---|
| Python server packaging and stdio launcher | 22 shipped servers; individual backend prerequisites still apply |
| Portable skills | Primary workflow, dataset-report and adapted Clio Coder skill folders; installation checks are separate from each skill's model evaluation |
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

Add `--hook` for an optional read-only Claude `SessionStart` hook requiring
`python3`. Hook files and inline definitions are validated without execution;
run the native client validator too. Hooks remain specific to their host and
are not portable skill files. See the [hook guide](https://github.com/iowarp/clio-kit/blob/main/community/README.md#hooks).

`--open-pr` requires the GitHub CLI, authentication, and Git commit identity.
Without it, submission only renders the entry; `--output` writes a local file.
Automated acceptance tests do not create public PRs. Explicit public-submission
checks use a contributor's fork and close the test PR without merging it.

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
source distribution and wheel, uses its release catalogue to download and install all portable skills,
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

### Project client installer

The September 17 project-installer checks cover Codex, OpenCode, Cursor,
Antigravity, Claude Code and VS Code configuration generation, preservation of
existing settings, explicit partial-install limits, and real fixture MCP calls
using each generated configuration. Native Codex loaded the scientific-I/O
skills and four server tool inventories in a trusted project. OpenCode 1.2.6
discovered the three skills and connected to all four servers; its generated
timeout allows two minutes for startup, but cold dependency downloads may still
need a prior `clio-kit doctor --server NAME --connect` run.

Antigravity 1.2.0 discovered the skills and compression tools. Its headless
operational attempt was first permission-denied, then returned an incomplete
model/tool interaction; that is not a successful scientific workflow test.
The installed older Cursor CLI did not discover the project configuration;
Cursor and VS Code GUI behavior remain unverified in this follow-up. Native
Claude hooks and agents are not translated by the project installer. These
limits concern the tested integration routes, not whether those clients offer
their own plugin systems.

### Scientific workflows

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
and pattern detection, and preserves records at chunk boundaries. Parquet
rejects invalid filters instead of reporting an unfiltered result as filtered.
Slurm submission and allocation diagnostics use stderr
to preserve the MCP protocol stream. These fixes have targeted regressions;
installation and connection results alone still do not verify a scientific
workflow.

Model evaluation is specific to the client, model and task. Use fresh temporary
projects and preserve the prompts, tool traces and output artifacts. Compare
runs with and without skill instructions against the same scientific checks;
installation success, skill discovery and token counts alone do not establish
scientific correctness or a better workflow. Draft-only and unavailable-backend
checks must be reported separately from operational tests. See the
[authoring guide](authoring.md) for evaluation guidance.

The dataset-report acceptance check installs a fresh wheel and exercises
HDF5 → Pandas → Plot with checked CSV values, numerical summaries, a PNG and an
unchanged input. Its native-client test also exercises the evidence reviewer,
verification hook, failure recovery and shared-dependency removal. Scripted
model responses make those integration checks repeatable; use separate live
model runs to assess scientific interpretation and unnecessary tool calls.

Independent numerical checks remain necessary: a model reviewer can itself
suggest an incorrect correction. Summary statistics alone do not establish
time order, a fitted relationship or a scientific cause.

## Skill maintenance

Owned skills use concise capability names matching their folders and
frontmatter. Titles describe the professional task; descriptions identify when
to use it and distinguish adjacent workflows. Each skill has completion criteria and
an `evals.md` file. Names/descriptions are used for discovery; the full body loads
on invocation, following the [Agent Skills specification](https://agentskills.io/specification).

The current review checks tool names, argument shapes, state and file handoffs,
size limits, provenance and interpretation. The `scenarios-recorded` metadata identifies scenario definitions, not a
quality certification. Retain the client, revision, inputs and observed outcome in local evaluation
evidence when repeating a behavioral check.

Specific limits matter: HDF5 aggregate statistics may sample data above 500 MiB.
A live 70-million-element array of ones returns sample sum/count 700,000, now
explicitly labeled as 1% coverage rather than full-dataset totals. Stream summaries can cover only part of a dataset; CSV profiles retain a bounded
sample. None should be presented as exact full-data calculations without checking
coverage. Skill instructions require checking these coverage labels.

## Validation scope

The installed-wheel acceptance suite checks plugin installation/update, server
handshakes, Node/Go cold and warm launches, and bounded scientific fixtures.
Partial-installation checks verify the component download list and cached reuse.
Repeat these checks for the commit being reviewed; previous results do not
certify a later release or a different scientific environment.

Dependency advisory scans are separate from installation tests. Locally built
packages and Git-hosted dependencies may not be covered by PyPI matching.
The website build keeps a pre-build image-format restriction introduced for an
upstream image-parser advisory; the locked dependency is now patched. See the
[website maintenance notes](https://github.com/iowarp/clio-kit/blob/main/website/README.md#image-parser-advisory).

Use the GitHub Actions results for the exact commit being reviewed. Public
publication and target-site backend acceptance remain separate steps; the
checks above do not establish universal agent or scientific-workflow support.


## MCP SDK v2 migration

All 22 Python servers lock MCP SDK 2.2.0 and stable FastMCP 4.0.3. The shared
verification client follows the [official SDK migration guide](https://py.sdk.modelcontextprotocol.io/migration/)
and [FastMCP upgrade guide](https://gofastmcp.com/getting-started/upgrading/from-fastmcp-3).
SDK versions and wire protocol versions are separate: modern connections use
`2026-07-28`, while legacy clients can still negotiate `2025-11-25`.

### Protocol compatibility and application state

MCP v2 removes transport-session requirements; it does not automatically remove
the state used by a scientific workflow. The standard permits application state
through explicit handles. See the [protocol announcement](https://blog.modelcontextprotocol.io/posts/2026-07-28/).

| Component | State to account for when deploying |
| --- | --- |
| HDF5 | `open_file` selects a process-local current file for subsequent tools. Keep that sequence in the same server process. |
| ParaView | The active visualization pipeline and render context belong to one process/backend connection. |
| ChronoLog | Recording uses an active story handle in the process. Archive retrieval uses explicit chronicle/story names. |
| Lmod | Restoring a collection changes the server process's retained module environment; it does not change the client's shell. Named collections persist on disk. |
| Web tasks | The default in-memory task backend does not survive a process restart. Configure the supported shared task backend for durable tasks. |
| JARVIS, Slurm and Spack | Pipelines, jobs and installations live in their configured backends; tools use explicit identifiers. Independent instances need access to the same backend/configuration. |

These servers can speak MCP v2 without being application-stateless. In particular,
do not round-robin the implicit-state workflows above between independent workers
or share them between unrelated users. The supplied stdio configuration gives
each launched server its own process. File-based tools still require access to
the same input/output paths when moving work between machines.

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

To repeat installed-wheel acceptance with the verification extra installed:

```bash
uv run --frozen python scripts/verify_marketplace_install.py --all-servers --protocol-mode 2026-07-28 --output /tmp/clio-v2-modern
uv run --frozen python scripts/verify_marketplace_install.py --all-servers --protocol-mode legacy --skip-client --output /tmp/clio-v2-legacy
```

Use `--skip-client` on systems without Claude Code; MCP checks require no model
credentials. Detailed per-server logs belong with the local audit evidence,
not in the package or website assets.
