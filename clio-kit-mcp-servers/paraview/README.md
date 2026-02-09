# ParaView MCP Server

[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![PyPI version](https://img.shields.io/pypi/v/paraview-mcp.svg)](https://pypi.org/project/paraview-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [CLIO Kit](https://toolkit.iowarp.ai/) - Gnosis Research Center**

ParaView MCP is a Model Context Protocol server that enables LLMs to create scientific 3D visualizations using ParaView through natural language commands. Features autonomous visualization, native ADIOS2/BP5 support, visual feedback, and no programming required.

## Quick Start

```bash
uvx clio-kit mcp-server paraview
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
      "args": ["clio-kit", "mcp-server", "paraview"]
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
      "args": ["clio-kit", "mcp-server", "paraview"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add paraview-mcp -- uvx clio-kit mcp-server paraview
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
      "args": ["clio-kit", "mcp-server", "paraview"]
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
git clone https://github.com/iowarp/clio-kit.git
uv --directory=$CLONE_DIR/clio-kit/clio-kit-mcp-servers/paraview run paraview-mcp --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/clio-kit.git
uv --directory=%CLONE_DIR%\clio-kit\clio-kit-mcp-servers\paraview run paraview-mcp --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/clio-kit.git
uv --directory=$env:CLONE_DIR\clio-kit\clio-kit-mcp-servers\paraview run paraview-mcp --help
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

### Data I/O (5 tools)
- **`load_scientific_data`** - Load VTK, EXODUS, CSV, RAW, BP5/ADIOS2, and other scientific data formats
- **`export_data`** - Export data to VTK, CSV, or other supported formats
- **`get_data_info`** - Get detailed information about loaded datasets
- **`list_arrays`** - List all available data arrays in the current dataset
- **`get_array_range`** - Get value range for a specific data array

### Visualization & Filtering (10 tools)
- **`generate_isosurface`** - Create isosurface visualizations for extracting surfaces of constant value
- **`create_data_slice`** - Create slices through volume data to examine cross-sections
- **`configure_volume_display`** - Toggle volume rendering for direct 3D visualization
- **`generate_flow_streamlines`** - Create streamlines from vector field data for flow visualization
- **`apply_threshold_filter`** - Apply threshold filter to extract data within value range
- **`apply_clip_filter`** - Clip data using planes or other geometric shapes
- **`apply_calculator`** - Create derived fields using mathematical expressions
- **`apply_contour`** - Create contours at specific data values
- **`apply_warp_by_vector`** - Warp geometry using vector field
- **`toggle_object_visibility`** - Show or hide visualization objects in the pipeline

### Rendering & Display (8 tools)
- **`apply_field_coloring`** - Color visualizations by specific data fields
- **`set_color_map`** - Set color map (lookup table) for data visualization
- **`set_color_map_preset`** - Apply preset color maps (Rainbow, Blue to Red, Viridis, etc.)
- **`get_histogram`** - Get histogram data for field values with automatic binning
- **`adjust_volume_opacity`** - Edit volume rendering opacity transfer function
- **`take_viewport_screenshot`** - Capture high-resolution screenshots of current ParaView viewport
- **`set_background_color`** - Set viewport background color
- **`set_representation`** - Change visualization representation (Surface, Wireframe, Points, etc.)

### Camera & View Control (4 tools)
- **`rotate_camera`** - Rotate camera around center of rotation
- **`reset_camera`** - Reset camera to optimal viewing parameters
- **`set_camera_position`** - Set specific camera position and focal point
- **`adjust_camera_zoom`** - Adjust camera zoom level

### ADIOS2/BP5 Support (2 tools)
- **`query_adios2_metadata`** - Query metadata from ADIOS2/BP5 files
- **`convert_bp5_to_vtk`** - Convert BP5/ADIOS2 files to VTK format

**Total: 29 MCP Tools** providing comprehensive ParaView automation through natural language

## Examples

### Basic Scientific Data Visualization
```
Load /data/simulation_output.vtk with temperature data, create an isosurface at temperature 300, 
apply Blue to Red color map preset, and take a high-resolution screenshot.
```

### Volume Visualization with Flow Analysis
```
Using /data/fluid_dynamics.bp5, create volume rendering of pressure field with Rainbow color map,
add streamlines to visualize flow patterns, and adjust opacity for better visibility.
```

### Multi-Slice Data Exploration
```
Load /data/medical_scan.vti, get array info to identify density field, create three orthogonal slices 
through the center, color by density field using Viridis preset, and export slices to VTK format.
```

### Advanced ADIOS2/BP5 Analysis
```
Query metadata from /data/checkpoint.bp5 to list available timesteps and variables, 
convert to VTK format, create histogram of temperature distribution, apply threshold filter 
to extract hot regions (>500K), and visualize with appropriate color mapping.
```

### Interactive Camera Control
```
Load /data/molecule.vtk, create isosurface of electron density, rotate camera 45 degrees around Y axis,
zoom in to focus on binding site, set camera position for optimal viewing angle, and save multiple viewpoints.
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

- **Full Documentation**: [CLIO Kit Website](https://toolkit.iowarp.ai/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/clio-kit/wiki/Contribution)
- **Detailed Setup**: See [USAGE_README.md](./USAGE_README.md) for complete ParaView installation and configuration

## License

BSD-3-Clause with proper attribution to original LLNL work
