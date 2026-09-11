---
sidebar_position: 1
---

# Getting Started

CLIO Kit brings scientific MCP servers, workflow skills, agents and contributed
plugins into one marketplace. This branch contains 22 server plugins, six
workflow bundles, 20 skills and two planning/review agents. External plugin
repositories and marketplace collections remain under their maintainers' control.

The shipped scientific servers are Python projects. The launcher also supports
Node/TypeScript and Go descriptors with runtime-specific dependency locks.
See [Marketplace and Contributions](./marketplace.md) for supported components,
contribution commands, update behavior and tested runtime boundaries.

## Install this branch

Use the same checkout for the launcher and marketplace. The public PyPI release
and default GitHub branch currently have a different catalogue.

```bash
git clone --branch feat/360-meta-marketplace https://github.com/iowarp/clio-kit.git
cd clio-kit
uv tool install --force --reinstall ".[verification]"
clio-kit mcp-servers
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) first.
If the launcher is not on PATH, run `uv tool update-shell` and open a new shell.
First server launches may download dependencies.

### Portable skills: Codex and other compatible agents

```bash
clio-kit skill install --bundle clio-scientific-io --target /path/to/project/.agents/skills
```

For Codex, `.agents/skills` is a project skill directory; `~/.agents/skills` is
user-wide. Other agents use their own documented discovery path as `--target`.
Omit `--bundle` for all 20 skills. Configure required MCP servers separately;
for example `codex mcp add clio-hdf5 -- clio-kit mcp-server hdf5` registers HDF5.
The scientific I/O workflow also requires ADIOS, Parquet and compression.

### Claude Code native plugins

```bash
claude plugin marketplace add "$PWD"
claude plugin install clio-scientific-io@clio-kit
claude mcp list
```

This route requires Claude Code with plugin dependency support. Reload plugins
or restart an existing session before using its new tools. Plugin registration
alone does not establish that servers connect.

The repository's [setup guide](https://github.com/iowarp/clio-kit/blob/feat/360-meta-marketplace/setup.md)
includes an actual compression round-trip check and troubleshooting steps.

## Choose a workflow

| Bundle | Work |
|---|---|
| `clio-hpc` | Software discovery, JARVIS execution and Slurm scheduling |
| `clio-performance` | I/O diagnosis, log investigation and session provenance |
| `clio-scientific-io` | Inspect and read HDF5, ADIOS BP5, Parquet and compressed files |
| `clio-analysis` | Tabular data, statistics, charts and ParaView |
| `clio-geoscience` | Geospatial, terrain and seismic analysis |
| `clio-research` | Literature, public datasets and operator catalogues |

In Claude Code, a bundle installs its member servers and matching skills. Install a server alone
with `claude plugin install clio-hdf5@clio-kit`, a workflow's procedures with
`clio-scientific-io-skills@clio-kit`, or all procedures with `clio-skills@clio-kit`.
`clio-agents@clio-kit` provides a workflow planner and evidence reviewer.

## Other MCP clients

The launcher also supports clients that accept stdio MCP configurations:

```json
{
  "mcpServers": {
    "hdf5": {
      "command": "clio-kit",
      "args": ["mcp-server", "hdf5"]
    }
  }
}
```

Use the client's documented configuration location and schema. Skills are
installed separately with `clio-kit skill install`. Native `.claude-plugin`
bundles, dependency resolution and agent definitions currently target Claude
Code; a plain MCP configuration starts only the named server.

## Verify the workflow

`clio-kit doctor --server hdf5 --connect` checks prerequisites and a real MCP
connection. Follow it with a representative tool call on known data and inspect
the result. HPC workflows require site software and scheduler access; ParaView
and ChronoLog require compatible native backends. A successful connection does
not establish that those dependencies are available or that a scientific result
is correct.

The [marketplace guide](./marketplace.md#reproduce-the-installed-system-checks)
describes fresh-wheel installation tests, real client installation, Node/Go
fixture execution, scientific data checks and remaining acceptance boundaries.

## Repository structure

```text
clio-kit/
├── .claude-plugin/marketplace.json
├── clio-kit-mcp-servers/    # Independent scientific server projects
├── plugins/                # Workflow bundles, aggregate skills and agents
├── skills/                 # Six skill packages, each with scenario records
├── community/entries/      # External plugin and marketplace entries
├── src/clio_kit/            # Launcher, authoring, discovery and cache management
├── tests/fixtures/         # Runtime acceptance fixtures
└── clio-kit-website/docs/  # Documentation
```

Server environments are isolated and identified by source and lock contents.
Python projects use `uv.lock`, Node projects use `package-lock.json`, and Go
projects use module versions and checksums. Source in the wheel does not imply
a cold installation works offline.

## Contribute and get help

Read the [contributor guide](https://github.com/iowarp/clio-kit/blob/feat/360-meta-marketplace/CONTRIBUTING.md)
for local skills, hosted servers and indexed contributions. Use
`clio-kit plugin init`, `plugin validate` and `plugin submit` for plugin authoring.
The marketplace guide explains what each command validates and publishes.

CLIO Kit is developed by the [Gnosis Research Center](https://grc.iit.edu/) at
[Illinois Institute of Technology](https://www.iit.edu/) as part of
[IoWarp](https://iowarp.ai), with National Science Foundation support.
The repository is licensed under BSD-3-Clause.

Report issues on [GitHub](https://github.com/iowarp/clio-kit/issues) or join the
[community discussion](https://iowarp.zulipchat.com/#narrow/channel/543872-Agent-Toolkit).
