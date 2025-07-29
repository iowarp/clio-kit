# IoWarp MCPs

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/iowarp-mcps.svg)](https://pypi.org/project/iowarp-mcps/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![CI](https://github.com/iowarp/iowarp-mcps/actions/workflows/quality_control.yml/badge.svg)](https://github.com/iowarp/iowarp-mcps/actions/workflows/quality_control.yml)
[![Coverage](https://codecov.io/gh/iowarp/iowarp-mcps/branch/main/graph/badge.svg)](https://codecov.io/gh/iowarp/iowarp-mcps)

[![Tests](https://img.shields.io/badge/Tests-14%20MCP%20Packages-blue)](https://github.com/iowarp/iowarp-mcps/actions/workflows/test-mcps.yml)
[![WRP Framework](https://img.shields.io/badge/WRP-AI%20Testing%20Framework-blue)](https://github.com/iowarp/iowarp-mcps/actions/workflows/wrp-tests.yml)

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![MyPy](https://img.shields.io/badge/mypy-checked-blue)](http://mypy-lang.org/)
[![uv](https://img.shields.io/badge/uv-managed-orange)](https://github.com/astral-sh/uv)
[![pip-audit](https://img.shields.io/badge/security-pip--audit-green)](https://pypi.org/project/pip-audit/)

Collection of MCP servers specifically designed for scientific computing research that enable AI agents and LLMs to interact with data analysis tools, HPC resources, and research datasets through a standardized protocol.

**More info at**: [https://iowarp.github.io/iowarp-mcps/](https://iowarp.github.io/iowarp-mcps/)

## Quick Installation

All our packages are released on [PyPI](https://pypi.org/project/iowarp-mcps/) for easy installation and usage.

### Simple Command

```bash
# Run any MCP server directly
uvx iowarp-mcps <server-name>
```

### List All MCPs

```bash
# See all available MCP servers
uvx iowarp-mcps
```

### Get Started with a Simple Command

```bash
# Example: Run the pandas MCP server
uvx iowarp-mcps pandas

# Example: Run the plot MCP server  
uvx iowarp-mcps plot

# Example: Run the slurm MCP server
uvx iowarp-mcps slurm
```

## Available Packages

<div align="center">

| 📦 **Package** | 🔧 **System** | 📋 **Description** | ⚡ **Install Command** |
|:---|:---:|:---|:---|
| **`adios`** | Data I/O | Read data using ADIOS2 engine | `uvx iowarp-mcps adios` |
| **`arxiv`** | Research | Fetch research papers from ArXiv | `uvx iowarp-mcps arxiv` |
| **`chronolog`** | Logging | Log and retrieve data from ChronoLog | `uvx iowarp-mcps chronolog` |
| **`compression`** | Utilities | File compression with gzip | `uvx iowarp-mcps compression` |
| **`darshan`** | Performance | I/O performance trace analysis | `uvx iowarp-mcps darshan` |
| **`hdf5`** | Data I/O | List HDF5 files from directories | `uvx iowarp-mcps hdf5` |
| **`jarvis`** | Workflow | Data pipeline lifecycle management | `uvx iowarp-mcps jarvis` |
| **`lmod`** | Environment | Environment module management | `uvx iowarp-mcps lmod` |
| **`node-hardware`** | System | System hardware information | `uvx iowarp-mcps node-hardware` |
| **`pandas`** | Data Analysis | CSV data loading and filtering | `uvx iowarp-mcps pandas` |
| **`parallel-sort`** | Computing | Large file sorting simulation | `uvx iowarp-mcps parallel-sort` |
| **`parquet`** | Data I/O | Read Parquet file columns | `uvx iowarp-mcps parquet` |
| **`plot`** | Visualization | Generate plots from CSV data | `uvx iowarp-mcps plot` |
| **`slurm`** | HPC | Job submission simulation | `uvx iowarp-mcps slurm` |

</div>

## Members

**Primary Institution:**
- <img src="https://grc.iit.edu/img/logo.png" alt="GRC Logo" width="24" height="24"> **[Gnosis Research Center (GRC)](https://grc.iit.edu/)** - [Illinois Institute of Technology](https://www.iit.edu/)

**Collaborating Institutions:**
- 📊 **[HDF Group](https://www.hdfgroup.org/)** - Data format and library developers
<!-- - **[University of Utah](https://www.utah.edu/)** - Research collaboration   -->


## Sponsors

<img src="https://www.nsf.gov/themes/custom/nsf_theme/components/molecules/logo/logo-desktop.png" alt="NSF Logo" width="24" height="24"> **[NSF (National Science Foundation)](https://www.nsf.gov/)** - Supporting scientific computing research and AI integration initiatives

## Development & Publishing

### Testing Development Versions

Development versions are automatically published to TestPyPI on every commit to main. To test the latest development version:

```bash
# Install dev version from TestPyPI
uvx --index-url https://test.pypi.org/simple/ iowarp-mcps

# Or install specific dev version
pip install --index-url https://test.pypi.org/simple/ iowarp-mcps==1.2.3.dev20241201123456+abc1234
```

Development versions use the format: `{base_version}.dev{timestamp}+{commit_sha}`

### Creating Releases

To create a stable release on PyPI:

1. **Create and push a version tag:**
   ```bash
   git tag v1.2.3
   git push origin v1.2.3
   ```

2. **GitHub Actions will automatically:**
   - Build the package with the tagged version
   - Publish to PyPI
   - Create a GitHub release with artifacts

3. **Users can then install the stable version:**
   ```bash
   uvx iowarp-mcps  # Latest stable version
   uvx iowarp-mcps@1.2.3  # Specific version
   ```

### Version Management

- **Base version** is maintained in `pyproject.toml`
- **Dev versions** are auto-generated from commits to main
- **Release versions** come from git tags (without the 'v' prefix)
- All versions follow [semantic versioning](https://semver.org/)

## Contributing

We welcome contributions in any form!

### Ways to Contribute:

- **Submit Issues**: Report any problems or bugs you encounter
- **Request Features**: Submit an issue requesting a new MCP server or functionality
- **Develop**: Try your hand at developing new MCP servers

Find our comprehensive **contribution/development/debugging guide** [here](https://github.com/iowarp/iowarp-mcps/wiki/Contribution).

### Get Help & Connect

**Reach out to us on Zulip**: [IoWarp-mcp Community Chat](https://grc.zulipchat.com/#narrow/channel/518574-iowarp-mcps)

---