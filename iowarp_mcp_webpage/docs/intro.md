---
sidebar_position: 1
---

# Getting Started

IoWarp MCPs are Model Context Protocol servers for scientific computing. They enable AI coding assistants (Cursor, Claude Code, VS Code) to interact with HPC resources and scientific data formats through natural language.

**Built by:** Gnosis Research Center, Illinois Institute of Technology
**Supported by:** National Science Foundation
**Technology:** FastMCP 2.12, Python 3.10+, MIT licensed

---

## Quick Start

### Install uv (if needed)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Run a Server

```bash
# List all 15 available servers
uvx iowarp-mcps

# Example: Run HDF5 server
uvx iowarp-mcps hdf5
```

This launches the server via stdio transport. Your AI assistant can now use it.

### Add to Your AI Assistant

**Cursor:** Edit `~/.cursor/mcp.json`
```json
{
  "mcpServers": {
    "hdf5": {"command": "uvx", "args": ["iowarp-mcps", "hdf5"]}
  }
}
```

**Claude Code:**
```bash
claude mcp add hdf5 -- uvx iowarp-mcps hdf5
```

**VS Code:** Add to settings
```json
"mcp": {
  "servers": {
    "hdf5": {"type": "stdio", "command": "uvx", "args": ["iowarp-mcps", "hdf5"]}
  }
}
```

Restart your editor. The MCP server tools will be available in AI assistant context.

---

## What Can You Do?

### With HDF5 MCP (27 tools)

Work with HDF5 scientific data files:
- Explore file structure and datasets
- Read full or partial datasets
- Access metadata and attributes
- Parallel batch processing
- Stream large datasets efficiently
- AI-powered structure analysis

Example prompt: *"Open simulation.h5 and show me the temperature dataset structure"*

### With Slurm MCP (13 tools)

Manage HPC cluster jobs:
- Submit jobs with resource specifications
- Monitor job status and queue position
- Retrieve job output
- Allocate interactive nodes
- Query cluster information

Example prompt: *"Submit train.py to Slurm with 32 cores, 64GB RAM, 24 hours"*

### With Pandas MCP (15 tools)

Process and analyze tabular data:
- Load CSV, Excel, Parquet, HDF5
- Statistical analysis and correlations
- Data cleaning and transformation
- Time series operations
- Save to multiple formats

Example prompt: *"Load data.csv, clean missing values, compute statistics by group"*

### With ArXiv MCP (13 tools)

Search and retrieve research papers:
- Search by author, title, keywords, date
- Download PDFs
- Export citations to BibTeX
- Find similar papers

Example prompt: *"Find recent diffusion model papers and export top 5 to BibTeX"*

### Other Servers

- **ADIOS** (5 tools) - Read ADIOS2 BP5 files
- **Darshan** (10 tools) - Analyze I/O performance traces
- **Lmod** (10 tools) - Manage environment modules
- **Plot** (6 tools) - Generate plots from CSV data
- **Compression** - GZIP compression
- **Jarvis** (27 tools) - Data pipeline management
- **ChronoLog** (4 tools) - Distributed logging
- **NDP** (3 tools) - Dataset discovery via CKAN
- **Node Hardware** (11 tools) - System monitoring
- **Parallel Sort** (13 tools) - Large file sorting
- **Parquet** - Parquet file operations

[Browse all servers →](/)

---

## Architecture

Each MCP server is an independent Python package with its own dependencies. The `iowarp-mcps` launcher uses `uvx` to run servers in isolated environments.

**Repository structure:**
```
iowarp-mcps/
├── src/iowarp_mcps/       # Unified launcher (180 lines, Click only)
├── iowarp_mcp_servers/     # 15 independent server packages
│   ├── hdf5/              # v2.1 - 27 tools, FastMCP 2.12.5, h5py 3.15.1
│   ├── pandas/            # v1.0 - 15 tools
│   ├── slurm/             # v1.0 - 13 tools
│   └── ...                # 12 more servers
└── pyproject.toml         # Launcher config only
```

**Design benefits:**
- Dependency isolation (each server has own requirements)
- Independent development (students work on separate servers)
- Unified user experience (single `uvx iowarp-mcps <name>` command)
- Auto-discovery (launcher scans for servers via pyproject.toml)

---

## HDF5 MCP - Reference Implementation

HDF5 MCP v2.1 demonstrates MCP best practices. Study this server as a template:

**Dependencies:** FastMCP 2.12.5, h5py 3.15.1, numpy 2.3.4 (latest as of Oct 2025)

**MCP Protocol Features:**
- 27 tools with complete annotations (title, readonly, destructive, idempotent, openworld hints)
- Context API: progress reporting, AI-powered insights via LLM sampling
- 3 resource URIs with templates
- 4 workflow prompts for guided analysis

**Code Quality:**
- Full type coverage (MyPy checked)
- 10 passing tests with realistic fixtures
- Educational demo script with sample climate data
- Comprehensive documentation (README, TOOLS.md, ARCHITECTURE.md, EXAMPLES.md, TRANSPORTS.md)

**Performance:**
- LRU caching (100-1000x speedup on repeated queries)
- Parallel processing via ThreadPoolExecutor (4-8x speedup)
- Streaming for datasets larger than RAM
- Adaptive performance monitoring

**Location:** `iowarp_mcp_servers/hdf5/`

Try the demo:
```bash
cd iowarp_mcp_servers/hdf5/examples
uv run python create_demo_data.py
uv run python demo_script.py
```

[View HDF5 documentation →](/docs/mcps/hdf5)

---

## Development

### Clone Repository

```bash
git clone https://github.com/iowarp/iowarp-mcps.git
cd iowarp-mcps
```

### Work on a Server

```bash
cd iowarp_mcp_servers/hdf5
uv sync --all-extras --dev
uv run pytest tests/ -v
uv run hdf5-mcp
```

### Add a New Server

1. Create directory: `iowarp_mcp_servers/my-server/`
2. Add `pyproject.toml` with entry point
3. Implement server with FastMCP decorators
4. Add tests
5. Launcher auto-discovers it

See [CONTRIBUTING.md](https://github.com/iowarp/iowarp-mcps/blob/main/CONTRIBUTING.md) for complete guide.

---

## Support

- **Documentation:** [iowarp.github.io/iowarp-mcps](https://iowarp.github.io/iowarp-mcps/)
- **Community Chat:** [Zulip](https://grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps)
- **Issues:** [GitHub Issues](https://github.com/iowarp/iowarp-mcps/issues)
- **Discussions:** [GitHub Discussions](https://github.com/iowarp/iowarp-mcps/discussions)

**Institutional Links:**
- [Gnosis Research Center](https://grc.iit.edu/)
- [Illinois Institute of Technology](https://www.iit.edu/)
- [IoWarp Project](https://iowarp.ai)

---

## Citation

If you use IoWarp MCPs in your research:

```
IoWarp MCPs: Model Context Protocol Servers for Scientific Computing
Gnosis Research Center, Illinois Institute of Technology
https://iowarp.github.io/iowarp-mcps/
```

**Funding:** This work is supported in part by the National Science Foundation.
