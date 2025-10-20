# IoWarp MCPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/iowarp-mcps.svg)](https://pypi.org/project/iowarp-mcps/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastMCP](https://img.shields.io/badge/FastMCP-2.12-purple)](https://gofastmcp.com)
[![CI](https://github.com/iowarp/iowarp-mcps/actions/workflows/quality_control.yml/badge.svg)](https://github.com/iowarp/iowarp-mcps/actions/workflows/quality_control.yml)
[![Coverage](https://codecov.io/gh/iowarp/iowarp-mcps/branch/main/graph/badge.svg)](https://codecov.io/gh/iowarp/iowarp-mcps)

[![Tests](https://img.shields.io/badge/Tests-15%20MCP%20Packages-blue)](https://github.com/iowarp/iowarp-mcps/actions/workflows/test-mcps.yml)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![uv](https://img.shields.io/badge/uv-managed-orange)](https://github.com/astral-sh/uv)

**AI Tools for Scientific Computing** - Model Context Protocol servers enabling AI agents to interact with HPC resources, scientific data formats, and research datasets.

**Website**: [https://iowarp.github.io/iowarp-mcps/](https://iowarp.github.io/iowarp-mcps/) | **Chat**: [Zulip Community](https://grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps) | Developed by <img src="https://grc.iit.edu/img/logo.png" alt="GRC Logo" width="18" height="18"> **[Gnosis Research Center (GRC)](https://grc.iit.edu/)**

---

## ❌ Without IoWarp MCPs

Working with scientific data and HPC resources requires manual scripting and tool-specific knowledge:

- ❌ Write custom scripts for every HDF5/Parquet file exploration
- ❌ Manually craft Slurm job submission scripts
- ❌ Switch between multiple tools for data analysis
- ❌ No AI assistance for scientific workflows
- ❌ Repetitive coding for common research tasks

## ✅ With IoWarp MCPs

AI agents handle scientific computing tasks through natural language:

- ✅ **"Analyze the temperature dataset in this HDF5 file"** - HDF5 MCP does it
- ✅ **"Submit this simulation to Slurm with 32 cores"** - Slurm MCP handles it
- ✅ **"Find papers on neural networks from ArXiv"** - ArXiv MCP searches
- ✅ **"Plot the results from this CSV file"** - Plot MCP visualizes
- ✅ **"Optimize memory usage for this pandas DataFrame"** - Pandas MCP optimizes

**One unified interface. 15 MCP servers. 150+ specialized tools. Built for research.**

IoWarp MCPs brings AI assistance to your scientific computing workflow—whether you're analyzing terabytes of HDF5 data, managing Slurm jobs across clusters, or exploring research papers. Built by researchers, for researchers, at Illinois Institute of Technology with NSF support.

> **One simple command.** Production-ready, fully typed, MIT licensed, and beta-tested in real HPC environments.

## 🚀 Quick Installation

### One Command for Any Server

```bash
# List all 15 available MCP servers
uvx iowarp-mcps

# Run any server instantly
uvx iowarp-mcps hdf5
uvx iowarp-mcps pandas
uvx iowarp-mcps slurm
```

<details>
<summary><b>Install in Cursor</b></summary>

Add to your Cursor `~/.cursor/mcp.json`:

```json
{
  "mcpServers": {
    "hdf5-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "hdf5"]
    },
    "pandas-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "pandas"]
    },
    "slurm-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "slurm"]
    }
  }
}
```

See [Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for more info.

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

```bash
# Add HDF5 MCP
claude mcp add hdf5-mcp -- uvx iowarp-mcps hdf5

# Add Pandas MCP
claude mcp add pandas-mcp -- uvx iowarp-mcps pandas

# Add Slurm MCP
claude mcp add slurm-mcp -- uvx iowarp-mcps slurm
```

See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add to your VS Code MCP config:

```json
"mcp": {
  "servers": {
    "hdf5-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["iowarp-mcps", "hdf5"]
    },
    "pandas-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["iowarp-mcps", "pandas"]
    }
  }
}
```

See [VS Code MCP docs](https://code.visualstudio.com/docs/copilot/chat/mcp-servers) for more info.

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Edit `claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "hdf5-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "hdf5"]
    },
    "arxiv-mcp": {
      "command": "uvx",
      "args": ["iowarp-mcps", "arxiv"]
    }
  }
}
```

See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

</details>

## Available Packages

<div align="center">

| 📦 **Package** | 📌 **Ver** | 🔧 **System** | 📋 **Description** | ⚡ **Install Command** |
|:---|:---:|:---:|:---|:---|
| **`adios`** | 1.0 | Data I/O | Read data using ADIOS2 engine | `uvx iowarp-mcps adios` |
| **`arxiv`** | 1.0 | Research | Fetch research papers from ArXiv | `uvx iowarp-mcps arxiv` |
| **`chronolog`** | 1.0 | Logging | Log and retrieve data from ChronoLog | `uvx iowarp-mcps chronolog` |
| **`compression`** | 1.0 | Utilities | File compression with gzip | `uvx iowarp-mcps compression` |
| **`darshan`** | 1.0 | Performance | I/O performance trace analysis | `uvx iowarp-mcps darshan` |
| **`hdf5`** | 2.1 | Data I/O | HPC-optimized scientific data with 27 tools, AI insights, caching, streaming | `uvx iowarp-mcps hdf5` |
| **`jarvis`** | 1.0 | Workflow | Data pipeline lifecycle management | `uvx iowarp-mcps jarvis` |
| **`lmod`** | 1.0 | Environment | Environment module management | `uvx iowarp-mcps lmod` |
| **`ndp`** | 1.0 | Data Protocol | Search and discover datasets across CKAN instances | `uvx iowarp-mcps ndp` |
| **`node-hardware`** | 1.0 | System | System hardware information | `uvx iowarp-mcps node-hardware` |
| **`pandas`** | 1.0 | Data Analysis | CSV data loading and filtering | `uvx iowarp-mcps pandas` |
| **`parallel-sort`** | 1.0 | Computing | Large file sorting simulation | `uvx iowarp-mcps parallel-sort` |
| **`parquet`** | 1.0 | Data I/O | Read Parquet file columns | `uvx iowarp-mcps parquet` |
| **`plot`** | 1.0 | Visualization | Generate plots from CSV data | `uvx iowarp-mcps plot` |
| **`slurm`** | 1.0 | HPC | Job submission simulation | `uvx iowarp-mcps slurm` |

</div>

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

---

## 🚨 Troubleshooting

<details>
<summary><b>Server Not Found Error</b></summary>

If `uvx iowarp-mcps <server-name>` fails:

```bash
# Verify server name is correct
uvx iowarp-mcps

# Common names: hdf5, pandas, slurm, arxiv (not hdf5-mcp, pandas-mcp)
```

</details>

<details>
<summary><b>Import Errors or Missing Dependencies</b></summary>

For development or local testing:

```bash
cd iowarp_mcp_servers/hdf5
uv sync --all-extras --dev
uv run hdf5-mcp
```

</details>


<details>
<summary><b>uvx Command Not Found</b></summary>

Install uv package manager:

```bash
# Linux/macOS
curl -LsSf https://astral.sh/uv/install.sh | sh

# Windows
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or via pip
pip install uv
```

</details>

---

## Team 

- **[Gnosis Research Center (GRC)](https://grc.iit.edu/)** - [Illinois Institute of Technology](https://www.iit.edu/) | Lead 
- **[HDF Group](https://www.hdfgroup.org/)** - Data format and library developers | Industry Partner    
- **[University of Utah](https://www.utah.edu/)** - Research collaboration | Domain Science Partner

## Sponsored By

<img src="https://www.nsf.gov/themes/custom/nsf_theme/components/molecules/logo/logo-desktop.png" alt="NSF Logo" width="24" height="24"> **[NSF (National Science Foundation)](https://www.nsf.gov/)** - Supporting scientific computing research and AI integration initiatives

 > we welcome more sponsorships. please contact the [Principal Investigator](mailto:akougkas@illinoistech.edu)

## Ways to Contribute

- **Submit Issues**: Report bugs or request features via [GitHub Issues](https://github.com/iowarp/iowarp-mcps/issues)
- **Develop New MCPs**: Add servers for your research tools ([CONTRIBUTING.md](CONTRIBUTING.md))
- **Improve Documentation**: Help make guides clearer
- **Share Use Cases**: Tell us how you're using IoWarp MCPs in your research

**Full Guide**: [CONTRIBUTING.md](CONTRIBUTING.md) 

### Community & Support

- **Chat**: [Zulip Community](https://grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps)
- **Issues**: [GitHub Issues](https://github.com/iowarp/iowarp-mcps/issues)
- **Discussions**: [GitHub Discussions](https://github.com/iowarp/iowarp-mcps/discussions)
- **Website**: [https://iowarp.github.io/iowarp-mcps/](https://iowarp.github.io/iowarp-mcps/)
- **Project**: [IOWarp Project](https://iowarp.ai)

---