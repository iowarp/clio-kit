# Lmod MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/lmod-mcp.svg)](https://pypi.org/project/lmod-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [CLIO Kit](https://toolkit.iowarp.ai/) - Gnosis Research Center**

Lmod MCP is a comprehensive Model Context Protocol (MCP) server that enables Language Learning Models (LLMs) to manage environment modules using the Lmod system. This server provides advanced module management capabilities, environment configuration tools, and HPC workflow support with seamless i...

## Quick Start

Install the launcher with the [setup guide](../../setup.md) first.

```bash
clio-kit mcp-server lmod
```

## Documentation

- **Full Documentation**: [CLIO Kit Website](https://toolkit.iowarp.ai/)
- **Installation Guide**: See [setup guide](../../setup.md)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md)

---

## Description

Lmod MCP is a comprehensive Model Context Protocol (MCP) server that enables Language Learning Models (LLMs) to manage environment modules using the Lmod system. This server provides advanced module management capabilities, environment configuration tools, and HPC workflow support with seamless integration with AI coding assistants.

**Key Features:**
- **Comprehensive Module Management**: List, search, load, unload, and inspect modules with intelligent dependency handling
- **Advanced Search Capabilities**: Spider search through entire module hierarchy with pattern matching and filtering
- **Environment Collections**: Save and restore complete module configurations for reproducible environments
- **Atomic Operations**: Safe module swapping and dependency-aware loading with conflict resolution
- **HPC Integration**: Optimized for scientific computing workflows with batch job environment management
- **MCP Integration**: Full Model Context Protocol compliance for seamless LLM integration


## 🛠️ Installation

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)
- Lmod and Bash installed; set `LMOD_CMD` to Lmod's `libexec/lmod` executable (or put `lmod` on `PATH`) and `MODULEPATH` to the site modulefiles
- HPC environment with module system access

The server evaluates Lmod's shell integration and retains collection changes
for subsequent calls in the same MCP process. A shell `module` function does
not need to be an executable on `PATH`. These changes do not modify the parent
shell or other MCP servers. Modulefiles execute site-provided shell code and
must come from a trusted installation.

<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file is the recommended approach. You may also install in a specific project by creating `.cursor/mcp.json` in your project folder. See [Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for more info.

```json
{
  "mcpServers": {
    "lmod-mcp": {
      "command": "clio-kit",
      "args": [
        "mcp-server",
        "lmod"
      ]
    }
  }
}
```

</details>

<details>
<summary><b>Install in VS Code</b></summary>

Add this to `.vscode/mcp.json` in your project. See [VS Code MCP docs](https://code.visualstudio.com/docs/copilot/chat/mcp-servers) for more info.

```json
{
  "servers": {
    "lmod-mcp": {
      "type": "stdio",
      "command": "clio-kit",
      "args": [
        "mcp-server",
        "lmod"
      ]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add lmod-mcp -- clio-kit mcp-server lmod
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

```json
{
  "mcpServers": {
    "lmod-mcp": {
      "command": "clio-kit",
      "args": [
        "mcp-server",
        "lmod"
      ]
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
uv --directory=$CLONE_DIR/clio-kit/clio-kit-mcp-servers/lmod run lmod-mcp --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/clio-kit.git
uv --directory=%CLONE_DIR%\clio-kit\clio-kit-mcp-servers\lmod run lmod-mcp --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/clio-kit.git
uv --directory=$env:CLONE_DIR\clio-kit\clio-kit-mcp-servers\lmod run lmod-mcp --help
```

</details>

## Capabilities

### `module_list`
**Description**: List all currently loaded environment modules.
**Hints**: read-only, idempotent
**Tags**: modules, query

### `module_avail`
**Description**: Search for available modules, optionally filtered by name pattern.
**Hints**: read-only, idempotent
**Tags**: modules, query

### `module_show`
**Description**: Display detailed information about a specific module.
**Hints**: read-only, idempotent
**Tags**: modules, query

### `module_spider`
**Description**: Search the entire module tree comprehensively for matching modules.
**Hints**: read-only, idempotent
**Tags**: modules, query

### `module_save`
**Description**: Save currently loaded modules as a named collection.
**Tags**: management, modules

### `module_restore`
**Description**: Restore a previously saved module collection.
**Tags**: management, modules

### `module_savelist`
**Description**: List all saved module collections.
**Hints**: read-only, idempotent
**Tags**: modules, query

### Resources

- `lmod://status` - Current Lmod module system status.
- `lmod://capabilities` - Describe the stateless Lmod contract exposed by this server.

### Prompts

- **setup_environment**: Guided workflow for setting up an HPC software environment.
## Claude Code

```bash
claude mcp add clio-lmod -- clio-kit mcp-server lmod
```

Or install via the CLIO Kit plugin marketplace:

```
/plugin marketplace add iowarp/clio-kit
/plugin install clio-lmod@clio-kit
```
## Claude Desktop

Add to your Claude Desktop config (`claude_desktop_config.json`):

```json
{
  "mcpServers": {
    "clio-lmod": {
      "command": "clio-kit",
      "args": [
        "mcp-server",
        "lmod"
      ]
    }
  }
}
```

## Examples

### Discover a software environment

Use `module_avail` to search visible modules, `module_spider` for hierarchical
search, and `module_show` to inspect a selected module's prerequisites and
settings. Select versions from the site's actual inventory.

### Record and restore a collection

Use `module_list` to inspect the MCP process's environment, `module_save` to
record it, and `module_savelist` to inspect available collections. Restore a
selected collection with `module_restore`, then call `module_list` again to
verify its contents. Collection changes persist for subsequent calls in that
MCP process, but do not change the parent shell or other MCP servers.

The seven-tool surface does not expose module loading, unloading or swapping.
For a JARVIS workload, resolve software with Spack and pass the exact returned
load specifications to `jarvis_run`. Verify the actual workload environment.
