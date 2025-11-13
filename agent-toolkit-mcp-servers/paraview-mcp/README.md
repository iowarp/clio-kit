# ParaView MCP Server

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://img.shields.io/pypi/v/paraview-mcp.svg)](https://pypi.org/project/paraview-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [Agent Toolkit](https://iowarp.github.io/agent-toolkit/) - Gnosis Research Center**

ParaView MCP is a Model Context Protocol server that enables LLMs to create scientific 3D visualizations using ParaView through natural language commands. Features autonomous visualization, native ADIOS2/BP5 support, visual feedback, and no programming required.

## Quick Start

```bash
uvx agent-toolkit paraview
```

---

## 🛠️ Installation

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- ParaView installation

<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file is the recommended approach. You may also install in a specific project by creating `.cursor/mcp.json` in your project folder. See [Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for more info.

```json
{
  "mcpServers": {
    "paraview-mcp": {
      "command": "uvx",
      "args": ["agent-toolkit", "paraview"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add this to your VS Code MCP config file. See [VS Code MCP docs](https://code.visualstudio.com/docs/copilot/chat/mcp-servers) for more info.

```json
"mcp": {
  "servers": {
    "paraview-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["agent-toolkit", "paraview"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add paraview-mcp -- uvx agent-toolkit paraview
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

```json
{
  "mcpServers": {
    "paraview-mcp": {
      "command": "uvx",
      "args": ["agent-toolkit", "paraview"]
    }
  }
}
```

</details>

<details>
<summary><b>Manual Setup</b></summary>

**Linux/macOS:**
```bash
CLONE_DIR=$(pwd)
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=$CLONE_DIR/agent-toolkit/agent-toolkit-mcp-servers/paraview-mcp run paraview-mcp --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=%CLONE_DIR%\agent-toolkit\agent-toolkit-mcp-servers\paraview-mcp run paraview-mcp --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=$env:CLONE_DIR\agent-toolkit\agent-toolkit-mcp-servers\paraview-mcp run paraview-mcp --help
```

</details>

## How to Install ParaView

### Option 1: Automated Installation

```bash
# One-command setup - installs and configures everything
uv run automate-setup
```

This automatically installs system dependencies, builds ParaView with ADIOS2 support, configures Python integration, and verifies installation.

**Time:** ~1-2 hours (mostly automated)

### Option 2: Quick Install with Conda

```bash
# Install ParaView and ADIOS2 via conda
conda install -c conda-forge paraview adios2
uv run configure-paraview
```

### Option 3: System Packages

```bash
# Ubuntu/Debian
sudo apt install paraview python3-paraview libadios2-dev
uv run configure-paraview
```

## How to Run ParaView Server

### Find Your ParaView Installation
```bash
find ~ -name "pvserver" -type f 2>/dev/null | head -5
find ~ -name "paraview" -type f 2>/dev/null | head -5
```

### Start ParaView Server
```bash
# Method 1: Using project scripts (recommended)
uv run paraview-server

# Method 2: Direct command (replace with your actual path)
PARAVIEW_PATH=$(find ~ -name "pvserver" -type f 2>/dev/null | head -1)
$(dirname "$PARAVIEW_PATH")/pvserver --multi-clients --server-port=11111
```

**Expected output:** `Waiting for client...` and `Accepting connection(s): hostname:11111`

## How to Run ParaView GUI

### Start ParaView GUI
```bash
# Method 1: Using project scripts (recommended)
uv run paraview-gui

# Method 2: Direct command (replace with your actual path)
PARAVIEW_GUI=$(find ~ -name "paraview" -type f 2>/dev/null | head -1)
$PARAVIEW_GUI
```

### Connect GUI to Server
1. Open ParaView GUI
2. Go to **File** → **Connect**
3. Select **Add Server**
4. Name: `localhost`, Host: `localhost`, Port: `11111`
5. Click **Connect**

## Capabilities

- **`load_scientific_data`** - Load VTK, EXODUS, CSV, RAW, BP5, and other scientific data formats
- **`generate_isosurface`** - Create isosurface visualizations for extracting surfaces of constant value
- **`create_data_slice`** - Create slices through volume data to examine cross-sections
- **`configure_volume_display`** - Toggle volume rendering for direct 3D visualization
- **`generate_flow_streamlines`** - Create streamlines from vector field data for flow visualization
- **`take_viewport_screenshot`** - Capture screenshots of current ParaView viewport
- **`apply_field_coloring`** - Color visualizations by specific data fields
- **`rotate_camera`** - Rotate camera to adjust viewing perspective
- **`reset_camera`** - Reset camera to optimal viewing parameters

## Examples

### Basic Scientific Data Visualization
```
Load /data/simulation_output.vtk with temperature data, create an isosurface at temperature 300, and take a screenshot.
```

### Volume Visualization with Flow Analysis
```
Using /data/fluid_dynamics.bp5, create volume rendering of pressure field and add streamlines to visualize flow patterns.
```

### Multi-Slice Data Exploration
```
Load /data/medical_scan.vti and create three orthogonal slices through the center, color by density field.
```

---

## Credits and Attribution

### Original Inspiration
This project builds upon concepts from the original LLNL ParaView MCP work:

**Original work**: [LLNL ParaView MCP](https://github.com/LLNL/paraview_mcp)  
**Authors**: Shusen Liu, Haichao Miao (LLNL)

### Dependencies
- **ParaView**: [Kitware ParaView](https://www.paraview.org/) - Open-source scientific visualization
- **ADIOS2**: [ORNL ADIOS2](https://adios2.readthedocs.io/) - Adaptable I/O System
- **FastMCP**: [FastMCP Framework](https://github.com/jlowin/fastmcp) - Model Context Protocol implementation

## Documentation

- **Full Documentation**: [Agent Toolkit Website](https://iowarp.github.io/agent-toolkit/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/agent-toolkit/wiki/Contribution)
- **Detailed Setup**: See [USAGE_README.md](./USAGE_README.md) for complete ParaView installation and configuration

## License

BSD-3-Clause with proper attribution to original LLNL work
