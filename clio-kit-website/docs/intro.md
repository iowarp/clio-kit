---
sidebar_position: 1
---

# Getting Started

CLIO Kit brings scientific MCP servers, workflow skills, agents and contributed
plugins into one marketplace. The catalogue contains 22 server plugins, six
workflow bundles, 20 skills and two planning/review agents. External plugin
repositories and marketplace collections remain under their maintainers' control.

The shipped scientific servers are Python projects. The launcher also supports
Node/TypeScript and Go descriptors with runtime-specific dependency locks.
See [Marketplace and Contributions](./marketplace.md) for supported components,
contribution commands, update behavior and tested runtime boundaries.

## Install from source

Use the same checkout for the launcher and marketplace. The commands below
install the checked-out code; package releases are published separately.

```bash
git clone https://github.com/iowarp/clio-kit.git
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

The repository's [setup guide](https://github.com/iowarp/clio-kit/blob/main/setup.md)
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

## Agent integrations

MCP servers provide tools; skills provide procedures for using them. Configure
both for a complete workflow. The scientific I/O workflow needs `hdf5`, `adios`,
`parquet`, and `compression`, plus its three skills.

| Client | Register MCP servers | Project skill target |
|---|---|---|
| Codex CLI / IDE extension | `codex mcp add`; shared `~/.codex/config.toml` | `.agents/skills` |
| Claude Code | Native marketplace above, or `claude mcp add --scope project` | Bundle-managed, or `.claude/skills` |
| Cursor | `.cursor/mcp.json` with `mcpServers` | `.cursor/skills` |
| VS Code / GitHub Copilot | `.vscode/mcp.json` with `servers` | `.github/skills` |
| Antigravity | MCP settings → raw config; workspace `.agents/mcp_config.json` | `.agents/skills` |
| Claude Desktop | Developer settings → `claude_desktop_config.json` with `mcpServers` | Local MCP setup does not install skills |

For example, in Codex:

```bash
codex mcp add clio-hdf5 -- clio-kit mcp-server hdf5
codex mcp add clio-adios -- clio-kit mcp-server adios
codex mcp add clio-parquet -- clio-kit mcp-server parquet
codex mcp add clio-compression -- clio-kit mcp-server compression
clio-kit skill install --bundle clio-scientific-io --target .agents/skills
codex mcp list
```

For Cursor, Antigravity, or Claude Desktop, merge these entries into the
configuration file listed above:

```json
{
  "mcpServers": {
    "clio-hdf5": { "command": "clio-kit", "args": ["mcp-server", "hdf5"] },
    "clio-adios": { "command": "clio-kit", "args": ["mcp-server", "adios"] },
    "clio-parquet": { "command": "clio-kit", "args": ["mcp-server", "parquet"] },
    "clio-compression": { "command": "clio-kit", "args": ["mcp-server", "compression"] }
  }
}
```

For VS Code / GitHub Copilot, use this complete `.vscode/mcp.json` structure:

```json
{
  "servers": {
    "clio-hdf5": { "type": "stdio", "command": "clio-kit", "args": ["mcp-server", "hdf5"] },
    "clio-adios": { "type": "stdio", "command": "clio-kit", "args": ["mcp-server", "adios"] },
    "clio-parquet": { "type": "stdio", "command": "clio-kit", "args": ["mcp-server", "parquet"] },
    "clio-compression": { "type": "stdio", "command": "clio-kit", "args": ["mcp-server", "compression"] }
  }
}
```

Run **MCP: List Servers** to start the servers, then use Copilot's Agent mode.
The Codex extension uses Codex's own configuration, even when running in VS Code.

Install skills using the appropriate target from the table:

```bash
# Cursor
clio-kit skill install --bundle clio-scientific-io --target .cursor/skills
# VS Code / GitHub Copilot
clio-kit skill install --bundle clio-scientific-io --target .github/skills
# Antigravity
clio-kit skill install --bundle clio-scientific-io --target .agents/skills
```

Current Cursor, VS Code, and Antigravity also discover `.agents/skills`;
reuse a shared installation there instead of making duplicate copies.
Antigravity also supports the older `.agent/skills` directory. Its current global
locations are `~/.gemini/config/mcp_config.json` for servers and
`~/.gemini/config/skills` for skills. Use the IDE's **MCP Servers → Manage MCP
Servers → View raw config** to locate the configuration for your installed version.
For Antigravity CLI, start the first session from your project with
`agy --new-project` so its local skills and MCP configuration are loaded.
Reopen it with `agy --project <project-name-or-id>`.

CLIO's native `.claude-plugin` bundles and agent definitions currently target
Claude Code. Other clients have their own plugin systems; the portable routes
above install CLIO skills and tools without claiming native plugin compatibility.
Claude Desktop's MCP configuration installs only servers.
For another local stdio client, use command `clio-kit` with arguments
`["mcp-server", "SERVER_NAME"]` in its documented schema. Remote-only clients
need a separately deployed MCP endpoint.

Install the launcher in the environment where your client starts its MCP
processes, including SSH, WSL, or a container. If `clio-kit` is not on the
client's PATH, use its absolute executable path. Reload or restart the client,
inspect its discovered tools and skills, and continue with verification below.

Client references: [Codex MCP](https://developers.openai.com/codex/mcp) and
[skills](https://developers.openai.com/codex/skills),
[Cursor MCP](https://cursor.com/docs/mcp) and [skills](https://cursor.com/docs/skills),
[VS Code MCP](https://code.visualstudio.com/docs/agent-customization/mcp-servers) and
[skills](https://code.visualstudio.com/docs/agent-customization/agent-skills),
[Antigravity MCP](https://antigravity.google/docs/mcp) and
[skills](https://antigravity.google/docs/skills/), and
[Claude Desktop MCP](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

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

Read the [contributor guide](https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md)
for local skills, hosted servers and indexed contributions. Use
`clio-kit plugin init`, `plugin validate` and `plugin submit` for plugin authoring.
The marketplace guide explains what each command validates and publishes.

CLIO Kit is developed by the [Gnosis Research Center](https://grc.iit.edu/) at
[Illinois Institute of Technology](https://www.iit.edu/) as part of
[IoWarp](https://iowarp.ai), with National Science Foundation support.
The repository is licensed under BSD-3-Clause.

Report issues on [GitHub](https://github.com/iowarp/clio-kit/issues) or join the
[community discussion](https://iowarp.zulipchat.com/#narrow/channel/543872-Agent-Toolkit).
