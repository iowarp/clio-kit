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

[**Website**](https://toolkit.iowarp.ai/) | [**IOWarp**](https://iowarp.ai)

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

```bash
git clone https://github.com/iowarp/clio-kit.git
cd clio-kit
```

**Recommended:** Ask your agent:

```text
Read setup.md and set up CLIO Kit for me.
```

**Manual setup:** Install the launcher, then choose an option below.

```bash
uv tool install --force --reinstall ".[verification]"
```

### Claude Code

Install a workflow bundle with its MCP servers and skills:

```bash
claude plugin marketplace add "$PWD"
claude plugin install clio-hpc@clio-kit
claude mcp list
```

Restart Claude Code after installation and confirm the servers connect.

### Codex and Other Agents

Install workflow skills into your project:

```bash
clio-kit skill install --bundle clio-scientific-io --target /path/to/project/.agents/skills
```

Configure the required MCP servers using [setup.md](setup.md). Use your agent’s supported skill directory.

### Install Skills with npm

Requires Node.js 22.20.0+. From your working project:

```bash
npx skills@1.5.25 add /path/to/clio-kit --skill '*' --agent codex --copy
```

Replace `codex` with `claude-code` or `antigravity`. This installs all 20 skills; MCP servers require separate configuration.

Choose either the Python or npm installer for each skill. See the [skills CLI guide](clio-kit-website/docs/marketplace.md#optional-skills-cli) for selecting skills, updates and removal.

<details>
<summary><b>One Command for Any Server</b></summary>

```bash
# From the CLIO Kit checkout, install the CLI in a persistent tool environment
uv tool install --force --reinstall .
# If uv's executable directory is not on PATH:
uv tool update-shell

# List all 22 available MCP servers
clio-kit mcp-servers

# Run an individual server
clio-kit mcp-server hdf5
clio-kit mcp-server pandas
clio-kit mcp-server slurm

# Agentic search for scientific corpora
clio-kit search serve
clio-kit search query --namespace local_fs --q "pressure > 200 kPa"
```

To install the published release instead, use `uv tool install clio-kit`.
`uv tool install` creates a persistent, isolated environment; use
`uvx --from clio-kit clio-kit ...` for a temporary, one-shot invocation.

Released wheels run each embedded Python MCP server using its shipped `uv.lock`.
The launcher caches environments by source and lock contents, installs production
dependencies, and refuses to resolve an embedded server whose lock is missing.
The `--branch` option is for development, not an immutable release installation.

The wheel also includes machine-readable tool contracts for JARVIS, Slurm, Spack
and Scientific Catalog, with SHA-256 digests for downstream verification:

```bash
clio-kit mcp-contracts
clio-kit mcp-contract clio-kit-jarvis-user-v3.7.2
clio-kit mcp-contract clio-kit-slurm-user-v3
clio-kit mcp-contract clio-kit-spack-user-v2.3
clio-kit mcp-contract clio-kit-scientific-catalog-user-v1.1
```

</details>

### Workflow Bundles

| Bundle | Purpose | MCP servers |
|---|---|---|
| `clio-hpc` | Build software and run cluster jobs | spack, lmod, jarvis, slurm, node-hardware |
| `clio-performance` | Diagnose slow jobs and inspect logs | darshan, chronolog, parallel-sort |
| `clio-scientific-io` | Explore scientific data files | hdf5, adios, parquet, compression |
| `clio-analysis` | Analyze results and create figures | pandas, plot, paraview |
| `clio-geoscience` | Work with maps, terrain and waveforms | geo, terrain, seismology |
| `clio-research` | Find papers and datasets | arxiv, ndp, scientific-catalog, web |

Some servers require additional system software. See [setup and prerequisites](setup.md#native-backend-setup).

### Skills and Agents

Skills provide workflow instructions and guidance for interpreting results. Claude Code users can install skills or agent definitions separately:

```bash
claude plugin install clio-skills@clio-kit
claude plugin install clio-agents@clio-kit
```

Native plugins and agent definitions currently target Claude Code. Portable skills work with compatible agents.

### Contribute Servers, Skills or Plugins

Add skills here, or list a plugin maintained in your own repository:

```bash
clio-kit plugin init my-plugin
clio-kit plugin validate my-plugin
clio-kit plugin submit my-plugin --repo owner/name
```

`submit` generates a community entry for a pull request. External servers can use any language; servers hosted inside CLIO currently support Python, Node.js and Go.

See the [contribution guide](CONTRIBUTING.md) and [community guide](community/README.md) for requirements and submission details.

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

Reload the MCP configuration. For Antigravity CLI, start the first session from
this project with `agy --new-project` so its local skills and MCP configuration
are loaded. Reopen it with `agy --project <project-name-or-id>`.
Reuse `.agents/skills` if already installed for Codex.
CLIO does not yet ship a native Antigravity plugin.
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
| **`adios`** | 2.2.5 | Data I/O | Read data using ADIOS2 engine | `clio-kit mcp-server adios` |
| **`arxiv`** | 2.2.5 | Research | Fetch research papers from ArXiv | `clio-kit mcp-server arxiv` |
| **`chronolog`** | 2.0.3 | Logging | Log and retrieve data from ChronoLog | `clio-kit mcp-server chronolog` |
| **`compression`** | 2.2.5 | Utilities | File compression with gzip | `clio-kit mcp-server compression` |
| **`darshan`** | 2.2.5 | Performance | I/O performance trace analysis | `clio-kit mcp-server darshan` |
| **`geo`** | 2.3.1 | Geospatial | Render GeoJSON vector layers with basemaps | `clio-kit mcp-server geo` |
| **`hdf5`** | 2.2.5 | Data I/O | HPC-optimized scientific data with 27 tools, AI insights, caching, streaming | `clio-kit mcp-server hdf5` |
| **`jarvis`** | 3.7.4 | Workflow | Durable pipeline, bounded package discovery, progress, artifact, and service-runtime management | `clio-kit mcp-server jarvis` |
| **`lmod`** | 3.0.1 | Environment | Environment module management | `clio-kit mcp-server lmod` |
| **`ndp`** | 2.2.5 | Data Protocol | Search and discover datasets across CKAN instances | `clio-kit mcp-server ndp` |
| **`node-hardware`** | 2.2.5 | System | System hardware information | `clio-kit mcp-server node-hardware` |
| **`pandas`** | 2.2.6 | Data Analysis | CSV data loading and filtering | `clio-kit mcp-server pandas` |
| **`parallel-sort`** | 2.2.5 | Computing | Large file sorting | `clio-kit mcp-server parallel-sort` |
| **`paraview`** | 2.2.5 | Visualization | Scientific 3D visualization and analysis | `clio-kit mcp-server paraview` |
| **`parquet`** | 2.2.5 | Data I/O | Read Parquet file columns | `clio-kit mcp-server parquet` |
| **`plot`** | 2.2.5 | Visualization | Generate plots from CSV data | `clio-kit mcp-server plot` |
| **`seismology`** | 2.3.1 | Seismology | Analyze SAC waveforms and archives | `clio-kit mcp-server seismology` |
| **`scientific-catalog`** | 1.1.4 | Discovery | Operator-owned scientific dataset discovery | `clio-kit mcp-server scientific-catalog` |
| **`slurm`** | 3.0.2 | HPC | Job submission and management | `clio-kit mcp-server slurm` |
| **`spack`** | 2.3.1 | Package Management | Structured package discovery, installation, and location | `clio-kit mcp-server spack` |
| **`terrain`** | 2.2.5 | Geospatial | Analyze DEMs and terrain point clouds | `clio-kit mcp-server terrain` |
| **`web`** | 2.1.3 | Web | Synchronous search plus durable, queryable, cancellable URL, DOI, and document fetch tasks | `clio-kit mcp-server web` |

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
- **Website**: [https://toolkit.iowarp.ai/](https://toolkit.iowarp.ai/)
- **Project**: [IOWarp Project](https://iowarp.ai)

---
