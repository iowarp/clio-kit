---
title: Choosing and composing plugins
---

# Plugins for scientific work

MCP servers, skills, agents and hooks are individual **components**. A **plugin**
bundles the components needed for a workflow; it can include any of these types
without requiring all four. The six primary bundles and task-specific packages
are both plugins. A **component collection** groups components of one type,
such as the scientific I/O skills or the planning/review agents.

| Choose | Example | Provides |
| --- | --- | --- |
| Individual component | HDF5 MCP | Tools for HDF5 files |
| Component collection | `clio-scientific-io-skills` | Scientific I/O procedures, without MCPs |
| Broad workflow plugin | `clio-scientific-io` | Scientific I/O MCPs and skills |
| Task plugin | `clio-dataset-report` | MCPs, report skill, agents and a verification hook |

### Catalogue types and installation packages

Claude Code calls every native installation package a plugin, including a single
MCP wrapper. CLIO Kit keeps these wrappers for individual installation and shared
dependencies, but does **not** count them as workflow plugins. The website's
Plugins filter combines broad bundles and task plugins. MCPs, skills, agents,
hooks and component collections have their own filters. An agent or hook detail
page names its containing package when installation also brings other components.

The catalogue derives maintained component types from local manifests and their
dependencies. Mixed-component packages and declared workflows are plugins;
single-type packages are component collections (or MCP entries). It does not infer
external contents from names or descriptions: indexed upstream packages appear
under External packages until their contents are classified. Catalogue entry
counts include collections and their browsable members, so they are not counts
of installed packages or unique executables.

This follows the host's distinction between
[standalone components and distributable plugin packages](https://code.claude.com/docs/en/plugins#when-to-use-plugins-vs-standalone-configuration).
Native manifests, agents and hooks currently target Claude Code; other clients
can install shared skills and MCPs together with the
[project installer](intro.md#install-a-workflow-for-your-client). This does not
convert native agents or hooks. Install the launcher
first using [Getting Started](intro.md).

## Primary bundles

The six primary bundles provide convenient starting points. They reuse server
and skill plugins through dependencies rather than copying their content.

| Bundle | Purpose | MCP servers |
| --- | --- | --- |
| `clio-hpc` | Prepare software and run work on a cluster | jarvis, lmod, node-hardware, slurm, spack |
| `clio-performance` | Investigate job performance and logs | chronolog, darshan, parallel-sort |
| `clio-scientific-io` | Inspect scientific files and read safely | adios, compression, hdf5, parquet |
| `clio-analysis` | Summarize results and create figures | pandas, paraview, plot |
| `clio-geoscience` | Inspect maps, terrain and waveforms | geo, seismology, terrain |
| `clio-research` | Find papers and research datasets | arxiv, ndp, scientific-catalog, web |

```bash
claude plugin install clio-hdf5@clio-kit          # one MCP
claude plugin install clio-scientific-io@clio-kit # primary bundle + skills
```

Each server belongs to exactly one **primary bundle** in
`mcp-server-versions.toml`. This coverage check catches forgotten servers and
accidental changes to those six collections. It does not restrict how task
plugins reuse components.

## Task plugins

You can now add a handwritten package directly under `plugins/<name>/`, with
its native manifest and whichever components it needs. The marketplace generator
discovers it without a TOML entry or mandatory dependencies. Skills-only,
agents-only and hooks-only packages use the same route in their corresponding
directories. See [folder contributions](authoring.md#add-a-component-folder-to-clio-kit).

### Scientific dataset report

`clio-dataset-report` combines HDF5, Pandas and Plot with the
`creating-dataset-report` skill, the `clio-agents` reviewer, and a read-only
PostToolUse verification hook. It produces a CSV, PNG, Markdown report and
evidence manifest from a small numeric HDF5 table.

```bash
claude plugin install clio-dataset-report@clio-kit
```

Invoke `/clio-dataset-report:creating-dataset-report` with the source file,
dataset path, column meanings/units and a new output directory. Install the
launcher first; the helper and native hook also require `python3`.

The helper verifies count, mean, median, min/max, unchanged source and output
presence. Limits are 64 MiB per source/CSV and 100,000 CSV rows. The hook gives
feedback after Write/Edit of this workflow's manifest or report. It does not
block writes, run on shell/MCP writes or certify report prose; the skill requires
an explicit final check. Review HDF5-to-CSV fidelity, figure meaning and
scientific interpretation separately.

The skill and helper also ship in the wheel:

```bash
clio-kit skill install creating-dataset-report --target .agents/skills
```

Configure HDF5/Pandas/Plot MCPs separately outside the native Claude plugin
route. Other hosts need explicit verification and their own review integration;
installing portable instructions does not install Claude hooks or agents.

Run the installed workflow and native hook acceptance check:

```bash
uv run --frozen python scripts/verify_dataset_report.py --output /tmp/clio-dataset-report-check
```

It performs actual MCP calls and uses a scripted model to exercise the real
client's hook feedback. Add `--live` for a separate authenticated model run.
The September 17 deterministic/native-client checks passed; the fresh live
composite-skill evaluation was blocked by account quota before execution.
Add `--external` to test the same packed components from a separate local Git
publisher, indexed through a community entry. This also validates the submission
entry, native skill/reviewer loading, hook feedback and shared-dependency removal.
It does not open a public PR.

### Author another task

A maintained task plugin may depend on components from several primary bundles.
For example, the following **authoring example is not a published plugin**:

```toml
# Add to mcp-server-versions.toml after reviewing the task.
[workflows.clio-inspect-and-plot]
version = "1.0.0"
description = "Inspect scientific data and prepare a bounded summary plot."
dependencies = ["clio-hdf5", "clio-pandas", "clio-plot"]
```

This registers those three MCPs, without pulling in either entire primary
bundle. It does not supply a new procedure or automatically transfer data
between servers. Add the skills needed for a complete workflow, and document
how their inputs and outputs connect before calling it an end-to-end solution.

Dependencies are **plugin names**, not skill folder names. They can name existing
maintained MCP, skill-collection, agent or primary-bundle plugins. Selecting a
collection installs its whole collection; selecting a primary bundle installs
its dependencies. A skills-only dependency does not register its suggested MCPs.
Several task plugins may reuse the same components.

For this TOML-generated route, task-to-task and external plugin dependencies
are rejected. Unknown names, collisions, repeated dependencies and malformed
versions fail generation. External authors continue to use the
[community contribution route](authoring.md#index-an-external-contribution).

Website start/build and CI regenerate both catalogues automatically. To inspect
a local native installation before either runs, use the optional command:

```bash
clio-kit marketplace sync --root .
```

This lightweight generator leaves server code and metadata untouched. It writes
`plugins/<name>/.claude-plugin/plugin.json` and the marketplace entry from the
TOML definition. Edit the TOML for these generated tasks, not their generated
manifest. Handwritten package manifests are preserved. README
files and other component files are preserved. When retiring a task, remove
both its TOML definition and its generated plugin directory, then regenerate.

## What a workflow should explain

A complete task plugin can combine MCP tools, a skill that connects their inputs
and outputs, an agent that reviews the evidence, and hooks for useful lifecycle
checks. These components are optional: include each for a specific purpose.
The dataset-report plugin uses HDF5, Pandas and Plot,
guides the data handoff, and provides evidence for review. Deterministic checks
can verify artifacts and calculations; an agent review alone cannot certify
scientific conclusions. A task definition composes dependencies;
local skill and hook files can add the task behavior.

Keep a short README beside each workflow plugin: its intended task, required
inputs and system prerequisites, included components, expected output and
verification limits. Add a new skill, agent or hook only when the task needs it;
see [Authoring components](authoring.md).

Before publishing a task, use the automatic CI checks or local sync above, then
`claude plugin validate plugins/<name> --strict`. Use
`clio-kit plugin validate plugins/<name> --maintained` for the kit's structural
checks when the package uses the reserved `clio-` prefix.
Install the task in a disposable client profile and check that only
the intended dependencies are enabled. Exercise the procedure on known data,
check its result against expected values, and test refresh/removal alongside
another plugin that shares a dependency. Successful installation alone does
not establish scientific correctness.
