# ParaView MCP

ParaView-MCP is an autonomous agent that integrates large language models with ParaView through the Model Context Protocol, enabling users to create scientific visualizations using natural language instead of complex GUI operations.

## Quick Setup

### 1. Install Dependencies

```bash
# Install uv package manager
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc

# Clone repository
git clone path/to/paraview_mcp.git
cd paraview_mcp

# Create virtual environment
uv venv --python 3.10
uv sync
```

### 2. Automated Installation

**One-command setup:**
```bash
uv run automate-setup
```

This will automatically:
- Install system dependencies (Ubuntu/Debian/Fedora/Arch)
- Build ParaView with ADIOS2 support 
- Configure Python integration
- Verify installation

**Time:** ~1-2 hours (mostly automated, depends on internet & CPU)

### 3. Alternative Quick Installs

If you prefer faster setup with existing ParaView:

```bash
# Option A: Conda (recommended for quick testing)
conda install -c conda-forge paraview adios2
uv run configure-paraview

# Option B: System packages (Ubuntu/Debian)
sudo apt install paraview python3-paraview libadios2-dev
uv run configure-paraview
```

## Usage

### Quick Start Commands

```bash
# Start ParaView GUI
uv run paraview-gui

# Start ParaView server (for remote/distributed computing)
uv run paraview-server

# Start MCP server (for Claude integration)
uv run paraview-mcp
```

**Server connection:** After starting `paraview-server`, open `paraview-gui` → File → Connect → localhost:11111

### Running ParaView Server and GUI

#### Option 1: Using Project Scripts (Recommended)
```bash
# Start ParaView server in background
uv run paraview-server

# In another terminal, start ParaView GUI
uv run paraview-gui

# Connect GUI to server: File → Connect → localhost:11111
```

#### Option 2: Using Direct ParaView Installation
```bash
# Find your ParaView installation
find ~ -name "pvserver" -type f | head -1
find ~ -name "paraview" -type f | head -1

# Start ParaView server (replace with your actual path)
/path/to/your/paraview/bin/pvserver --multi-clients --server-port=11111

# In another terminal, start ParaView GUI (replace with your actual path)
/path/to/your/paraview/bin/paraview
```

#### Option 3: Testing the Full Integration
```bash
# Terminal 1: Start ParaView server (replace with your actual path)
/path/to/your/paraview/bin/pvserver --multi-clients --server-port=11111

# Terminal 2: Start ParaView GUI (replace with your actual path)
/path/to/your/paraview/bin/paraview

# Terminal 3: Start MCP server
cd /path/to/paraview-mcp
uv run paraview-mcp

# Now you can use Claude Desktop to interact with ParaView through MCP
```

**Note:** Replace `/path/to/your/paraview/bin/` with your actual ParaView installation path found using the commands above.

### Claude Desktop Configuration

Add to `~/.config/Claude/claude_desktop_config.json` (Linux):

```json
{
  "mcpServers": {
    "ParaView": {
      "command": "uv",
      "args": [
        "run",
        "--directory",
        "/path/to/your/paraview_mcp",
        "paraview-mcp"
      ]
    }
  }
}
```

## Features

- **Native ADIOS2/BP5 support** - Direct reading of scientific simulation data
- **Autonomous visualization** - Natural language to visualization pipeline
- **Visual feedback** - AI can see and refine visualizations
- **No programming required** - Accessible to non-experts

## Troubleshooting

```bash
# Verify installation
uv run test-installation

# Check dependencies
uv run install-deps --check-only

# Reconfigure ParaView
uv run configure-paraview

# Clean rebuild
uv run automate-setup --clean
```

### Common Issues

#### ParaView Server Not Found
If `pvserver` command is not found, locate your ParaView installation:
```bash
find ~ -name "pvserver" -type f 2>/dev/null | head -5
find ~ -name "paraview" -type f 2>/dev/null | head -5
```

#### Library Path Issues
If you get library errors like `libpython3.13.so.1.0: cannot open shared object file`:
```bash
# Try different ParaView installations from the find command above
# Use the second or third path if the first one has library issues
PARAVIEW_PATH=$(find ~ -name "pvserver" -type f 2>/dev/null | head -1)
$(dirname "$PARAVIEW_PATH")/pvserver --multi-clients --server-port=11111
```

#### MCP Server Connection Issues
1. Ensure ParaView server is running first:
   ```bash
   # Should show "Waiting for client..." and "Accepting connection(s)"
   PARAVIEW_PATH=$(find ~ -name "pvserver" -type f 2>/dev/null | head -1)
   $(dirname "$PARAVIEW_PATH")/pvserver --multi-clients --server-port=11111
   ```

2. Then start MCP server:
   ```bash
   uv run paraview-mcp
   # Should show: "Successfully connected to ParaView server"
   ```

#### Testing the Connection
```bash
# Check if ParaView server is listening
netstat -ln | grep :11111

# Test MCP server functionality
python test_mcp_client.py  # If you have created the test client
```

## Commands Reference

**Setup & Installation:**
- `uv run automate-setup` - Complete automated setup
- `uv run install-deps --install` - Install system dependencies
- `uv run build-paraview` - Build ParaView from source
- `uv run configure-paraview` - Configure existing ParaView
- `uv run test-installation` - Comprehensive testing

**Usage:**
- `uv run paraview-gui` - Start ParaView GUI
- `uv run paraview-server` - Start ParaView server (port 11111)
- `uv run paraview-mcp` - Start MCP server for Claude integration

## Attribution

This project builds upon concepts from the original LLNL ParaView MCP work:

Original work: [LLNL ParaView MCP](https://github.com/LLNL/paraview_mcp)  
Authors: Shusen Liu, Haichao Miao (LLNL)

This enhanced implementation adds:
- Complete automation system
- Multi-installation detection  
- Cross-platform support
- Advanced ADIOS2/BP5 integration

## Quick Verification

To verify everything is working correctly:

1. **Find and Start ParaView Server:**
   ```bash
   # Find your ParaView installation
   PARAVIEW_PATH=$(find ~ -name "pvserver" -type f 2>/dev/null | head -1)
   echo "Using ParaView at: $PARAVIEW_PATH"
   
   # Start ParaView server
   $(dirname "$PARAVIEW_PATH")/pvserver --multi-clients --server-port=11111
   # Should show: "Waiting for client..." and "Accepting connection(s): hostname:11111"
   ```

2. **Start ParaView GUI:**
   ```bash
   # Find and start ParaView GUI
   PARAVIEW_GUI=$(find ~ -name "paraview" -type f 2>/dev/null | head -1)
   echo "Using ParaView GUI at: $PARAVIEW_GUI"
   $PARAVIEW_GUI
   # GUI should open successfully
   ```

3. **Start MCP Server:**
   ```bash
   cd /path/to/paraview-mcp
   uv run paraview-mcp
   # Should show: "Successfully connected to ParaView server"
   ```

4. **Test Integration with Agent Toolkit:**
   ```bash
   # From agent-toolkit root directory
   uv run agent-toolkit paraview --help
   # Should display ParaView MCP server help
   ```

## License

BSD-3-Clause with proper attribution to original LLNL work
