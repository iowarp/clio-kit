# CLIO Kit

<!-- mcp-name: io.github.iowarp/adios-mcp -->
<!-- mcp-name: io.github.iowarp/arxiv-mcp -->
<!-- mcp-name: io.github.iowarp/chronolog-mcp -->
<!-- mcp-name: io.github.iowarp/compression-mcp -->
<!-- mcp-name: io.github.iowarp/darshan-mcp -->
<!-- mcp-name: io.github.iowarp/geo-mcp -->
<!-- mcp-name: io.github.iowarp/hdf5-mcp -->
<!-- mcp-name: io.github.iowarp/jarvis-mcp -->
<!-- mcp-name: io.github.iowarp/lmod-mcp -->
<!-- mcp-name: io.github.iowarp/ndp-mcp -->
<!-- mcp-name: io.github.iowarp/node-hardware-mcp -->
<!-- mcp-name: io.github.iowarp/pandas-mcp -->
<!-- mcp-name: io.github.iowarp/parallel-sort-mcp -->
<!-- mcp-name: io.github.iowarp/paraview-mcp -->
<!-- mcp-name: io.github.iowarp/parquet-mcp -->
<!-- mcp-name: io.github.iowarp/plot-mcp -->
<!-- mcp-name: io.github.iowarp/seismology-mcp -->
<!-- mcp-name: io.github.iowarp/scientific-catalog-mcp -->
<!-- mcp-name: io.github.iowarp/slurm-mcp -->
<!-- mcp-name: io.github.iowarp/spack-mcp -->
<!-- mcp-name: io.github.iowarp/terrain-mcp -->
<!-- mcp-name: io.github.iowarp/web-mcp -->

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://img.shields.io/pypi/v/clio-kit.svg)](https://pypi.org/project/clio-kit/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/FastMCP-4.0-purple)](https://github.com/jlowin/fastmcp)
[![CI](https://github.com/iowarp/clio-kit/actions/workflows/quality_control.yml/badge.svg)](https://github.com/iowarp/clio-kit/actions/workflows/quality_control.yml)
[![Coverage](https://codecov.io/gh/iowarp/clio-kit/branch/main/graph/badge.svg)](https://codecov.io/gh/iowarp/clio-kit)

[![MCP Servers](https://img.shields.io/badge/MCP%20Servers-22-green)](https://github.com/iowarp/clio-kit/tree/main/clio-kit-mcp-servers)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![Type Checked](https://img.shields.io/badge/mypy-type%20checked-blue)](http://mypy-lang.org/)
[![Package Manager](https://img.shields.io/badge/uv-package%20manager-orange)](https://github.com/astral-sh/uv)
[![Security Audit](https://img.shields.io/badge/pip--audit-security%20scanned-green)](https://github.com/pypa/pip-audit)

**CLIO Kit** - Part of the IoWarp platform's tooling layer for AI agents. A comprehensive collection of tools, skills, plugins, and extensions. It ships 22 Model Context Protocol (MCP) servers for scientific computing and enables AI agents to interact with HPC resources, scientific data formats, and research datasets.

[**Website**](https://docs.iowarp.ai/) | [**IOWarp**](https://iowarp.ai)

Chat with us on [**Zulip**](https://iowarp.zulipchat.com/#narrow/channel/543872-Agent-Toolkit) or [**join us**](https://iowarp.zulipchat.com/join/e4wh24du356e4y2iw6x6jeay/)

Developed by <img src="https://grc.iit.edu/img/logo.png" alt="GRC Logo" width="18" height="18"> [**Gnosis Research Center**](https://grc.iit.edu/)

---

## ❌ Without CLIO Kit

Working with scientific data and HPC resources requires manual scripting and tool-specific knowledge:

- ❌ Write custom scripts for every HDF5/Parquet file exploration
- ❌ Manually craft Slurm job submission scripts
- ❌ Switch between multiple tools for data analysis
- ❌ No AI assistance for scientific workflows
- ❌ Repetitive coding for common research tasks

## ✅ With CLIO Kit

AI agents handle scientific computing tasks through natural language:

- ✅ **"Analyze the temperature dataset in this HDF5 file"** - HDF5 MCP does it
- ✅ **"Submit this simulation to Slurm with 32 cores"** - Slurm MCP handles it
- ✅ **"Find papers on neural networks from ArXiv"** - ArXiv MCP searches
- ✅ **"Plot the results from this CSV file"** - Plot MCP visualizes
- ✅ **"Optimize memory usage for this pandas DataFrame"** - Pandas MCP optimizes
- ✅ **"Find all documents where pressure exceeds 200 kPa"** - Agentic Search retrieves

**One unified interface. 22 MCP servers. Hybrid search engine. 150+ specialized tools. Built for research.**

CLIO Kit is part of the IoWarp platform's comprehensive tooling ecosystem for AI agents. It brings AI assistance to your scientific computing workflow—whether you're analyzing terabytes of HDF5 data, managing Slurm jobs across clusters, or exploring research papers. Built by researchers, for researchers, at Illinois Institute of Technology with NSF support.

> **Part of IoWarp Platform**: CLIO Kit is the tooling layer of the IoWarp platform, providing skills, plugins, and extensions for AI agents working in scientific computing environments.

> Scientific MCP servers, workflow skills, agents, and external collections in one marketplace.

## 🚀 Quick Installation

**AI agent (recommended)** - clone the repo, then tell your agent:

```
Read setup.md and set up CLIO Kit for me.
```

The agent will check prerequisites, use your stated work (or ask if it is unknown),
install skills and MCP tools using its supported configuration, and verify real results.

**Codex and other agents with Agent Skills support.** Install this branch's
launcher from a checkout (clone commands below), then install portable skills:

```bash
uv tool install --force --reinstall ".[verification]"
clio-kit skill list
clio-kit skill install --bundle clio-scientific-io --target /path/to/project/.agents/skills
```

`.agents/skills` is Codex's project discovery directory. Other agents can use
`--target` with their own documented skill directory. All 20 standard `SKILL.md`
folders ship in the Python package. Configure the required MCP servers separately
in the client; [setup.md](setup.md) gives complete Codex and Claude Code routes.
Native `.claude-plugin` bundles and the two agent definitions currently target
Claude Code; their manifest format is not shared by every agent.

**Claude Code users — install this feature branch from a matching checkout.**
The default GitHub branch and published PyPI package do not yet contain this
marketplace. Install the launcher and catalogue from the same checkout:

```bash
git clone --branch feat/360-meta-marketplace https://github.com/iowarp/clio-kit.git
cd clio-kit
```

```bash
uv tool install --force --reinstall ".[verification]"
claude plugin marketplace add "$PWD"
claude plugin install clio-hpc@clio-kit      # see the table below for other workflows
```

The first line is not optional. A plugin is a manifest that runs `clio-kit`; it
does not contain the server. Install plugins without the launcher and every one
of them reports `enabled` while every server fails with `ENOENT: Executable not
found in $PATH: "clio-kit"`.

Optional installations and checks:

```bash
claude plugin install clio-skills@clio-kit       # all 20 skills, no MCP servers
claude plugin install clio-agents@clio-kit       # planning and evidence review
clio-kit doctor --server hdf5 --connect
```

First builds download locked dependencies; vendored source alone does not
enable a cold offline installation. Contributed Node and Go projects also
require their corresponding toolchains.
Spack, Lmod, Slurm, ParaView, Chronolog, and site catalogue workflows have additional
system prerequisites; `doctor` reports basic prerequisites separately from MCP
connections. See [the feature and acceptance guide](clio-kit-website/docs/marketplace.md).

Reload plugins or restart Claude Code, then confirm the servers actually connected:

```bash
claude mcp list      # every plugin:clio-* line must say ✔ Connected
```

Use `claude mcp list`, not `claude plugin list` — the latter reports `enabled`
for plugins whose servers are completely broken.

<details>
<summary>or install servers individually</summary>

### One Command for Any Server

```bash
# From this checkout, install its CLI into a persistent tool environment
uv tool install --force --reinstall .
# If uv reports that its executable directory is not on PATH:
uv tool update-shell

# List all 22 available MCP servers
clio-kit mcp-servers

# Run any installed server
clio-kit mcp-server hdf5
clio-kit mcp-server pandas
clio-kit mcp-server slurm

# Agentic search — hybrid retrieval for scientific corpora
clio-kit search serve               # Start search API server
clio-kit search query --namespace local_fs --q "pressure > 200 kPa"

```

`uv tool install` keeps CLIO Kit in a persistent, isolated tool environment.
Use `uvx --from clio-kit clio-kit ...` only for a temporary, one-shot
invocation. Pin a version (`clio-kit==2.11.0`) only when you need one; unpinned
installs track the current release.

Released `clio-kit` wheels execute each embedded MCP server from that server's
shipped `uv.lock`. The launcher uses a source-and-lock-addressed environment
under the user cache, installs only production dependencies, and refuses to
resolve an embedded server whose lock is missing. The `--branch` launcher
option is an explicit development path and is not an immutable
release-artifact path.

The root wheel also ships machine-readable user contracts for the locked
JARVIS, SLURM, Spack, and Scientific Catalog servers. These artifacts are generated from real stdio
`tools/list` exchanges and include canonical SHA-256 digests for downstream
federation gates:

```bash
clio-kit mcp-contracts
clio-kit mcp-contract clio-kit-jarvis-user-v3.5
clio-kit mcp-contract clio-kit-slurm-user-v3
clio-kit mcp-contract clio-kit-spack-user-v2.3
clio-kit mcp-contract clio-kit-scientific-catalog-user-v1.1
```
</details>

### Workflow Bundles

In Claude Code, installing a bundle pulls in every server it needs plus the skills written for
that workflow, so you do not have to know which servers go together.

| Bundle | Servers | For |
|---|---|---|
| `clio-hpc` | spack, lmod, jarvis, slurm, node-hardware | Building software and running work on a cluster |
| `clio-performance` | darshan, chronolog, parallel-sort | Working out why a finished job was slow |
| `clio-scientific-io` | hdf5, adios, parquet, compression | Opening scientific data files and reading them safely |
| `clio-analysis` | pandas, plot, paraview | Turning results into statistics and figures |
| `clio-geoscience` | geo, seismology, terrain | Geospatial, terrain and waveform data |
| `clio-research` | arxiv, ndp, scientific-catalog, web | Finding papers and the datasets behind them |

Each bundle is a manifest naming its members, not a copy of them, so a bundle
cannot drift from the servers it bundles.

### Skills

Bundles ship skills: written procedures for tool sequences that are easy to get
wrong. They cover things the tool descriptions cannot say on their own, such as
which of two similar tools to reach for, what order calls have to happen in, and
how to read a number a server hands back.

For any compatible agent, install procedures independently of MCP servers:

```bash
clio-kit skill install --bundle clio-hpc --target /path/to/agent/skills
```

In Claude Code, skills are discovered once their bundle is installed. To install
only the procedures through its native marketplace:

```bash
claude plugin install clio-hpc-skills@clio-kit
```

Skill names and descriptions are available for selection; full instructions
load when a skill is used. Inspect installed components and the client's context
estimate:

```bash
claude plugin details clio-hpc-skills@clio-kit    # lists installed skills and context estimates
```

### Contributing Your Own Servers or Skills

The marketplace indexes work from outside this repository. Your code stays in
your repository, on your release schedule, and your updates reach users without a
release here.

```bash
clio-kit plugin init my-plugin      # scaffold a valid plugin
clio-kit plugin validate my-plugin  # check it before opening anything
claude plugin validate my-plugin --strict   # and the client's own rules
clio-kit plugin submit my-plugin --repo owner/name
```

`submit` prints the entry to add as `community/entries/<name>.toml` in a pull
request. Four source types are accepted — `github`, `git-subdir`, `npm` and
`url` — so a plugin published as an npm package, or living in a subdirectory of
a monorepo, is listable without moving into this repository.

**A server in another language can be indexed or hosted.** To keep it yours,
publish a plugin package containing its manifest and MCP configuration, then
add an `npm` entry: the plugin installs through this marketplace
while its code, dependencies and releases stay in your repository. To have it
ship as part of the kit, contribute it here with a `clio-server.toml` naming
its runtime — the launcher builds and starts node and go servers from their own
lock files exactly as it does Python. See
[CONTRIBUTING.md](CONTRIBUTING.md#contributing-a-server-in-another-language)
for which of the two to choose.

Once merged, an indexed contribution installs exactly like ours:

```bash
claude plugin install materials-lab@clio-kit
```

Indexed entries carry `metadata.indexed`, so the catalogue distinguishes what we
maintain from what we point at.

See [`community/README.md`](community/README.md) for the accepted source types,
what the generator enforces, and how to trial a contribution against a throwaway
config before indexing it.


<a id="agent-integrations"></a>

<details>
<summary><b>Install in Cursor</b></summary>

Add to `.cursor/mcp.json` in your project, or `~/.cursor/mcp.json`:

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

Install the workflow skills from your project:

```bash
clio-kit skill install --bundle clio-scientific-io --target .cursor/skills
```

See [Cursor MCP docs](https://cursor.com/docs/mcp) and
[skills docs](https://cursor.com/docs/skills).

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

From the CLIO Kit checkout, install the workflow plugin with its MCPs and skills:

```bash
claude plugin marketplace add "$PWD"
claude plugin install clio-scientific-io@clio-kit
claude mcp list
```

Skills-only and agent plugins:

```bash
claude plugin install clio-skills@clio-kit    # all 20 skills, no MCP servers
claude plugin install clio-agents@clio-kit    # workflow planner and evidence reviewer
```

See [setup.md](setup.md#claude-code-native-marketplace) for details.

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add to `.vscode/mcp.json` for GitHub Copilot:

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

Install the workflow skills from your project:

```bash
clio-kit skill install --bundle clio-scientific-io --target .github/skills
```

Start the servers with **MCP: List Servers**. The Codex extension uses the
Codex configuration below. See [VS Code MCP docs](https://code.visualstudio.com/docs/agent-customization/mcp-servers)
and [skills docs](https://code.visualstudio.com/docs/agent-customization/agent-skills).

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Edit `claude_desktop_config.json` through **Settings → Developer → Edit Config**:

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

Restart Claude Desktop. This config adds MCP servers only; CLIO's native
skill and agent plugins target Claude Code.
See [Claude Desktop MCP docs](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

</details>

<details>
<summary><b>Install in Codex</b></summary>

Add the MCP servers for Codex CLI or its IDE extension:

```bash
codex mcp add clio-hdf5 -- clio-kit mcp-server hdf5
codex mcp add clio-adios -- clio-kit mcp-server adios
codex mcp add clio-parquet -- clio-kit mcp-server parquet
codex mcp add clio-compression -- clio-kit mcp-server compression
codex mcp list
```

Install the workflow skills from your project:

```bash
clio-kit skill install --bundle clio-scientific-io --target .agents/skills
```

Check `/mcp` and `/skills` in Codex. CLIO's native plugins target Claude Code;
use the MCP and skill commands above for Codex.
See [Codex MCP docs](https://developers.openai.com/codex/mcp) and
[skills docs](https://developers.openai.com/codex/skills).

</details>

<details>
<summary><b>Install in Antigravity</b></summary>

Open **MCP Servers → Manage MCP Servers → View raw config** and add:

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

Install the workflow skills from your project:

```bash
clio-kit skill install --bundle clio-scientific-io --target .agents/skills
```

Reload the MCP configuration. Reuse `.agents/skills` if already installed for
Codex. CLIO does not yet ship a native Antigravity plugin.
See [Antigravity MCP docs](https://antigravity.google/docs/mcp) and
[skills docs](https://antigravity.google/docs/skills/).

</details>

## Available Packages

The version below identifies each MCP server's contract and runtime release,
independently of the containing `clio-kit` wheel version. JARVIS 3.7 and SLURM 3.0 have contracts
redesigned for agent use. Spack is at 2.3, while the other
contracts retain their existing 2.x identities until a focused upgrade.

The Spack install contract makes concretization explicit: `reuse=true` passes
`spack install --reuse`, while `reuse=false` passes `spack install --fresh`.
Agents should discover first, install only when needed, then pass the exact
`spack_locate` result to JARVIS for runtime loading. A find with no installed
match is normal typed data (`count=0`, `packages=[]`); locate reports the
distinct `not_installed` semantic, while real Spack failures remain errors.

<div align="center">

| 📦 **Package** | 📌 **Ver** | 🔧 **System** | 📋 **Description** | ⚡ **Install Command** |
|:---|:---:|:---:|:---|:---|
| **`adios`** | 2.2.4 | Data I/O | Read data using ADIOS2 engine | `clio-kit mcp-server adios` |
| **`arxiv`** | 2.2.4 | Research | Fetch research papers from ArXiv | `clio-kit mcp-server arxiv` |
| **`chronolog`** | 2.0.2 | Logging | Log and retrieve data from ChronoLog | `clio-kit mcp-server chronolog` |
| **`compression`** | 2.2.4 | Utilities | File compression with gzip | `clio-kit mcp-server compression` |
| **`darshan`** | 2.2.4 | Performance | I/O performance trace analysis | `clio-kit mcp-server darshan` |
| **`geo`** | 2.3.0 | Geospatial | Render GeoJSON vector layers with basemaps | `clio-kit mcp-server geo` |
| **`hdf5`** | 2.2.4 | Data I/O | HPC-optimized scientific data with 27 tools, AI insights, caching, streaming | `clio-kit mcp-server hdf5` |
| **`jarvis`** | 3.7.3 | Workflow | Durable pipeline, bounded package discovery, progress, artifact, and service-runtime management | `clio-kit mcp-server jarvis` |
| **`lmod`** | 3.0.0 | Environment | Environment module management | `clio-kit mcp-server lmod` |
| **`ndp`** | 2.2.4 | Data Protocol | Search and discover datasets across CKAN instances | `clio-kit mcp-server ndp` |
| **`node-hardware`** | 2.2.4 | System | System hardware information | `clio-kit mcp-server node-hardware` |
| **`pandas`** | 2.2.5 | Data Analysis | CSV data loading and filtering | `clio-kit mcp-server pandas` |
| **`parallel-sort`** | 2.2.4 | Computing | Large file sorting | `clio-kit mcp-server parallel-sort` |
| **`paraview`** | 2.2.4 | Visualization | Scientific 3D visualization and analysis | `clio-kit mcp-server paraview` |
| **`parquet`** | 2.2.4 | Data I/O | Read Parquet file columns | `clio-kit mcp-server parquet` |
| **`plot`** | 2.2.4 | Visualization | Generate plots from CSV data | `clio-kit mcp-server plot` |
| **`seismology`** | 2.3.0 | Seismology | Analyze SAC waveforms and archives | `clio-kit mcp-server seismology` |
| **`scientific-catalog`** | 1.1.3 | Discovery | Operator-owned scientific dataset discovery | `clio-kit mcp-server scientific-catalog` |
| **`slurm`** | 3.0.1 | HPC | Job submission and management | `clio-kit mcp-server slurm` |
| **`spack`** | 2.3.0 | Package Management | Structured package discovery, installation, and location | `clio-kit mcp-server spack` |
| **`terrain`** | 2.2.4 | Geospatial | Analyze DEMs and terrain point clouds | `clio-kit mcp-server terrain` |
| **`web`** | 2.1.2 | Web | Synchronous search plus durable, queryable, cancellable URL, DOI, and document fetch tasks | `clio-kit mcp-server web` |

</div>

### Agentic Search

Hybrid retrieval engine for scientific corpora — combines lexical (BM25), vector, graph, and scientific search (numeric range, unit matching, formula targeting) over namespaced document collections. DuckDB storage, FastAPI, async job queue, OpenTelemetry tracing, Prometheus metrics.

```bash
# Start the search API server
clio-kit search serve

# Index documents from a namespace
clio-kit search index --namespace local_fs

# Query with scientific operators
clio-kit search query --namespace local_fs --q "pressure between 190 and 360 kPa"

# List indexed documents
clio-kit search list --namespace local_fs
```

**API endpoints**: `/query`, `/jobs/index`, `/documents`, `/health`, `/metrics` — [full docs](clio-agentic-search/README.md)

---

## 📖 Usage Examples

### HDF5: Scientific Data Analysis

```
"What datasets are in climate_simulation.h5? Show me the temperature field structure and read the first 100 timesteps."
```

**Tools used:** `open_file`, `analyze_dataset_structure`, `read_partial_dataset`, `list_attributes`

### Slurm: HPC Job Management

```
"Submit simulation.py to Slurm with 32 cores, 64GB memory, 24-hour runtime. Monitor progress and retrieve output when complete."
```

**Tools used:** `submit_slurm_job`, `check_job_status`, `get_job_output`

### ArXiv: Research Discovery

```
"Find the latest papers on diffusion models from ArXiv, get details on the top 3, and export citations to BibTeX."
```

**Tools used:** `search_arxiv`, `get_paper_details`, `export_to_bibtex`, `download_paper_pdf`

### Pandas: Data Processing

```
"Load sales_data.csv, clean missing values, compute statistics by region, and save as Parquet with compression."
```

**Tools used:** `load_data`, `handle_missing_data`, `groupby_operations`, `save_data`

### Plot: Data Visualization

```
"Create a line plot showing temperature trends over time from weather.csv with proper axis labels."
```

**Tools used:** `line_plot`, `data_info`

### Agentic Search: Scientific Retrieval

```
"Find all chunks mentioning pressure above 200 kPa in the local_fs namespace."
```

**CLI:** `clio-kit search query --namespace local_fs --q "pressure > 200 kPa"`

---

## 🚨 Troubleshooting

<details>
<summary><b>Server Not Found Error</b></summary>

If `clio-kit mcp-server <server-name>` fails:

```bash
# Verify server name is correct
clio-kit mcp-servers

# Common names: hdf5, pandas, slurm, arxiv (not hdf5-mcp, pandas-mcp)
```

</details>

<details>
<summary><b>Import Errors or Missing Dependencies</b></summary>

For development or local testing:

```bash
cd clio-kit-mcp-servers/hdf5
uv sync --all-extras --dev
uv run hdf5-mcp
```

</details>


<details>
<summary><b>uv or clio-kit Command Not Found</b></summary>

Install uv package manager:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

Then install CLIO Kit persistently and expose uv's tool directory:

```bash
uv tool install --force --reinstall .
uv tool update-shell
```

Open a new shell after `update-shell`, then check it resolved:

```bash
clio-kit mcp-servers
```

</details>

---

## Team 

- **[Gnosis Research Center (GRC)](https://grc.iit.edu/)** - [Illinois Institute of Technology](https://www.iit.edu/) | Lead 
- **[HDF Group](https://www.hdfgroup.org/)** - Data format and library developers | Industry Partner    
- **[University of Utah](https://www.utah.edu/)** - Research collaboration | Domain Science Partner

## Sponsored By

<img src="https://www.nsf.gov/themes/custom/nsf_theme/components/molecules/logo/logo-desktop.png" alt="NSF Logo" width="24" height="24"> **[NSF (National Science Foundation)](https://www.nsf.gov/)** - Supporting scientific computing research and AI integration initiatives

 > we welcome more sponsorships. please contact the [Principal Investigator](mailto:grc@illinoistech.edu)

## Ways to Contribute

- **Submit Issues**: Report bugs or request features via [GitHub Issues](https://github.com/iowarp/clio-kit/issues)
- **Develop New MCPs**: Add servers for your research tools ([CONTRIBUTING.md](CONTRIBUTING.md))
- **Improve Documentation**: Help make guides clearer
- **Share Use Cases**: Tell us how you're using CLIO Kit in your research

**Full Guide**: [CONTRIBUTING.md](CONTRIBUTING.md) 

### Community & Support

- **Chat**: [Zulip Community](https://iowarp.zulipchat.com/#narrow/channel/543872-Agent-Toolkit)
- **Join**: [Invitation Link](https://iowarp.zulipchat.com/join/e4wh24du356e4y2iw6x6jeay/)
- **Issues**: [GitHub Issues](https://github.com/iowarp/clio-kit/issues)
- **Discussions**: [GitHub Discussions](https://github.com/iowarp/clio-kit/discussions)
- **Website**: [https://docs.iowarp.ai/](https://docs.iowarp.ai/)
- **Project**: [IOWarp Project](https://iowarp.ai)

---
