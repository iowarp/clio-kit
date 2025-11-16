# Node Hardware MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/node-hardware-mcp.svg)](https://pypi.org/project/node-hardware-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [Agent Toolkit](https://iowarp.github.io/agent-toolkit/) - Gnosis Research Center**

Node Hardware MCP is a Model Context Protocol server that enables LLMs to monitor and analyze system hardware information including CPU specifications, memory usage, disk performance, network interfaces, GPU details, and sensor data for both local and remote nodes via SSH connections, providing c...

## Quick Start

```bash
uvx iowarp-agent-toolkit mcp-server node-hardware
```

## Documentation

- **Full Documentation**: [Agent Toolkit Website](https://iowarp.github.io/agent-toolkit/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/agent-toolkit/wiki/Contribution)

---

## Description

Node Hardware MCP is a Model Context Protocol server that enables LLMs to monitor and analyze system hardware information including CPU specifications, memory usage, disk performance, network interfaces, GPU details, and sensor data for both local and remote nodes via SSH connections, providing comprehensive hardware monitoring and performance analysis capabilities.


## 🛠️ Installation

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- SSH client (for remote node capabilities)

<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file is the recommended approach. You may also install in a specific project by creating `.cursor/mcp.json` in your project folder. See [Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for more info.

```json
{
  "mcpServers": {
    "node-hardware-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "node-hardware"]
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
    "node-hardware-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "node-hardware"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add node-hardware-mcp -- uvx iowarp-agent-toolkit mcp-server node-hardware
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

```json
{
  "mcpServers": {
    "node-hardware-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "node-hardware"]
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
uv --directory=$CLONE_DIR/agent-toolkit/agent-toolkit-mcp-servers/node-hardware run node-hardware-mcp --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=%CLONE_DIR%\agent-toolkit\agent-toolkit-mcp-servers\node-hardware run node-hardware-mcp --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=$env:CLONE_DIR\agent-toolkit\agent-toolkit-mcp-servers\node-hardware run node-hardware-mcp --help
```

</details>

## Capabilities

### `get_cpu_info`
**Description**: Get comprehensive CPU information including specifications, core configuration, frequency analysis, and performance metrics.

**Returns**: dict: Structured CPU information with performance insights and optimization recommendations.

### `get_memory_info`
**Description**: Get comprehensive memory information including capacity, usage patterns, and performance characteristics.

**Returns**: dict: Structured memory information with usage insights and optimization recommendations.

### `get_system_info`
**Description**: Get comprehensive system information including operating system details, platform configuration, and system status.

**Returns**: dict: Structured system information with configuration insights and security recommendations.

### `get_disk_info`
**Description**: Get comprehensive disk information including storage devices, partitions, and I/O performance metrics.

**Returns**: dict: Structured disk information with performance insights and maintenance recommendations.

### `get_network_info`
**Description**: Get comprehensive network information including interfaces, connections, and bandwidth analysis.

**Returns**: dict: Structured network information with performance insights and security recommendations.

### `get_gpu_info`
**Description**: Get comprehensive GPU information including specifications, memory, and compute capabilities.

**Returns**: dict: Structured GPU information with performance insights and optimization recommendations.

### `get_sensor_info`
**Description**: Get sensor information including temperature, fan speeds, and thermal data.

**Returns**: dict: Structured sensor information with thermal insights and health recommendations.

### `get_process_info`
**Description**: Get process information including running processes and resource usage.

**Returns**: dict: Structured process information with resource insights and optimization recommendations.

### `get_performance_info`
**Description**: Get real-time performance metrics including CPU, memory, and disk usage.

**Returns**: dict: Structured performance information with bottleneck analysis and optimization recommendations.

### `get_remote_node_info`
**Description**: Get comprehensive remote node hardware and system information via SSH with advanced filtering and intelligent analysis.

**Parameters**:
- `hostname` (str): Target hostname or IP address for remote collection.
- `username` (Optional[str]): SSH username for remote authentication.
- `port` (int): SSH port number for remote connection.
- `ssh_key` (Optional[str]): Path to SSH private key file for authentication.
- `timeout` (int): SSH connection timeout in seconds.
- `components` (Optional[List[str]]): List of specific components to include in collection.
- `exclude_components` (Optional[List[str]]): List of specific components to exclude from collection.
- `include_performance` (bool): Whether to include real-time performance analysis.
- `include_health` (bool): Whether to include health assessment and predictive maintenance insights.

**Returns**: dict: Comprehensive remote hardware and system analysis, including hardware_data, collection_metadata, performance_analysis, health_assessment, ssh_connection_info, error_information, intelligent_insights, optimization_recommendations, and beautiful_formatting.

### `health_check`
**Description**: Perform comprehensive health check and system diagnostics with advanced capability verification.

**Returns**: dict: Comprehensive health assessment, including server_status, capability_status, system_compatibility, performance_metrics, diagnostic_insights, optimization_recommendations, troubleshooting_guide, predictive_maintenance, security_assessment, and health_summary.
## Examples

### 1. Local Hardware Overview
```
I need a comprehensive overview of my local system's hardware including CPU, memory, disk, and network components.
```

**Tools called:**
- `get_node_info` - Get complete local hardware information with all components
- Components collected: cpu, memory, disk, network, system, summary

### 2. Remote Server Monitoring
```
Monitor the hardware status of a remote server via SSH, focusing on CPU and memory utilization for performance analysis.
```

**Tools called:**
- `get_remote_node_info` - Connect to remote host with SSH authentication
- Components collected: cpu, memory, performance, system

### 3. GPU and Thermal Monitoring
```
Check GPU specifications and thermal sensors on both local and remote systems for machine learning workloads.
```

**Tools called:**
- `get_node_info` - Local GPU and sensor monitoring  
- `get_remote_node_info` - Remote GPU and thermal analysis
- Components collected: gpu, sensors, performance

### 4. System Health Assessment
```
Perform a comprehensive health check of system capabilities and verify all monitoring tools are working correctly.
```

**Tools called:**
- `health_check` - System health verification and diagnostic assessment
- `get_node_info` - Comprehensive local system analysis with health metrics

### 5. Performance Bottleneck Analysis  
```
Identify performance bottlenecks on a production server by analyzing CPU, memory, disk I/O, and running processes.
```

**Tools called:**
- `get_remote_node_info` - Remote performance analysis via SSH
- Components collected: cpu, memory, disk, performance, processes

### 6. Storage and Network Analysis
```
Analyze storage health and network interface performance on multiple systems for infrastructure monitoring.
```

**Tools called:**
- `get_node_info` - Local storage and network analysis
- `get_remote_node_info` - Remote storage and network monitoring  
- Components collected: disk, network, system, summary

