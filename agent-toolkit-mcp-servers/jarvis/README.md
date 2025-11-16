# Jarvis MCP Server

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![PyPI version](https://img.shields.io/pypi/v/jarvis-mcp.svg)](https://pypi.org/project/jarvis-mcp/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)

**Part of [Agent Toolkit](https://iowarp.github.io/agent-toolkit/) - Gnosis Research Center**

Jarvis MCP is a Model Context Protocol server that enables LLMs to manage the full lifecycle of data-centric pipelines using the Jarvis framework, featuring pipeline creation, package management, configuration updates, environment building, and execution capabilities for high-performance computin...

## Quick Start

```bash
uvx iowarp-agent-toolkit mcp-server jarvis
```

## Documentation

- **Full Documentation**: [Agent Toolkit Website](https://iowarp.github.io/agent-toolkit/)
- **Installation Guide**: See [INSTALLATION.md](../../../CLAUDE.md#setup--installation)
- **Contributing**: See [Contribution Guide](https://github.com/iowarp/agent-toolkit/wiki/Contribution)

---

## Description

Jarvis MCP is a Model Context Protocol server that enables LLMs to manage the full lifecycle of data-centric pipelines using the Jarvis framework, featuring pipeline creation, package management, configuration updates, environment building, and execution capabilities for high-performance computing and data science workflows.

## 🛠️ Installation

### Requirements

- Python 3.10 or higher
- [uv](https://docs.astral.sh/uv/) package manager (recommended)


<details>
<summary><b>Install in Cursor</b></summary>

Go to: `Settings` -> `Cursor Settings` -> `MCP` -> `Add new global MCP server`

Pasting the following configuration into your Cursor `~/.cursor/mcp.json` file is the recommended approach. You may also install in a specific project by creating `.cursor/mcp.json` in your project folder. See [Cursor MCP docs](https://docs.cursor.com/context/model-context-protocol) for more info.

```json
{
  "mcpServers": {
    "jarvis-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "jarvis"]
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
    "jarvis-mcp": {
      "type": "stdio",
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "jarvis"]
    }
  }
}
```

</details>

<details>
<summary><b>Install in Claude Code</b></summary>

Run this command. See [Claude Code MCP docs](https://docs.anthropic.com/en/docs/agents-and-tools/claude-code/tutorials#set-up-model-context-protocol-mcp) for more info.

```sh
claude mcp add jarvis-mcp -- uvx iowarp-agent-toolkit mcp-server jarvis
```

</details>

<details>
<summary><b>Install in Claude Desktop</b></summary>

Add this to your Claude Desktop `claude_desktop_config.json` file. See [Claude Desktop MCP docs](https://modelcontextprotocol.io/quickstart/user) for more info.

```json
{
  "mcpServers": {
    "jarvis-mcp": {
      "command": "uvx",
      "args": ["iowarp-agent-toolkit", "mcp-server", "jarvis"]
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
uv --directory=$CLONE_DIR/agent-toolkit/agent-toolkit-mcp-servers/jarvis run jarvis-mcp --help
```

**Windows CMD:**
```cmd
set CLONE_DIR=%cd%
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=%CLONE_DIR%\agent-toolkit\agent-toolkit-mcp-servers\jarvis run jarvis-mcp --help
```

**Windows PowerShell:**
```powershell
$env:CLONE_DIR=$PWD
git clone https://github.com/iowarp/agent-toolkit.git
uv --directory=$env:CLONE_DIR\agent-toolkit\agent-toolkit-mcp-servers\jarvis run jarvis-mcp --help
```

</details>

## Capabilities

### `update_pipeline`
**Description**: Re-apply environment and configuration to every package in a Jarvis pipeline.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline to update

**Returns**: dict: Status and results of the update operation.

### `build_pipeline_env`
**Description**: Build the pipeline execution environment for a given pipeline.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline to build

**Returns**: dict: Status and results of the environment build operation.

### `create_pipeline`
**Description**: Create a new pipeline environment for data-centric workflows.

**Parameters**:
- `pipeline_id` (str): Name/ID for the new pipeline

**Returns**: dict: Status and details of the created pipeline.

### `load_pipeline`
**Description**: Load an existing pipeline environment by ID, or the current one if not specified.

**Parameters**:
- `pipeline_id` (str, optional): ID of the pipeline to load

**Returns**: dict: Status and details of the loaded pipeline.

### `get_pkg_config`
**Description**: Retrieve the configuration of a specific package in a pipeline.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline
- `pkg_id` (str): ID of the package

**Returns**: dict: Current configuration of the package.

### `append_pkg`
**Description**: Add a package to a pipeline for execution or analysis.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline
- `pkg_type` (str): Type of package to add
- `pkg_id` (str, optional): ID for the new package
- `do_configure` (bool, optional): Whether to configure after adding
- `extra_args` (dict, optional): Additional configuration arguments

**Returns**: dict: Status and details of the package addition.

### `configure_pkg`
**Description**: Configure a package in a pipeline with new settings.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline
- `pkg_id` (str): ID of the package
- `extra_args` (dict, optional): Configuration arguments

**Returns**: dict: Status and details of the configuration operation.

### `unlink_pkg`
**Description**: Unlink a package from a pipeline without deleting its files.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline
- `pkg_id` (str): ID of the package to unlink

**Returns**: dict: Status and details of the unlink operation.

### `remove_pkg`
**Description**: Remove a package and its files from a pipeline.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline
- `pkg_id` (str): ID of the package to remove

**Returns**: dict: Status and details of the removal operation.

### `run_pipeline`
**Description**: Execute the pipeline, running all configured steps.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline to run

**Returns**: dict: Status and results of the pipeline execution.

### `destroy_pipeline`
**Description**: Destroy a pipeline and clean up all associated files and resources.

**Parameters**:
- `pipeline_id` (str): ID of the pipeline to destroy

**Returns**: dict: Status and details of the destruction operation.

### `jm_create_config`
**Description**: Initialize manager directories and persist configuration.

**Parameters**:
- `config_dir` (str): Parameter for config_dir
- `private_dir` (str): Parameter for private_dir
- `shared_dir` (Any, optional): Parameter for shared_dir

**Returns**: Returns list

### `jm_load_config`
**Description**: Load manager configuration from saved state.

**Returns**: Returns list

### `jm_save_config`
**Description**: Save current configuration state to disk.

**Returns**: Returns list

### `jm_set_hostfile`
**Description**: Set and save the path to the hostfile for deployments.

**Parameters**:
- `path` (str): Parameter for path

**Returns**: Returns list

### `jm_bootstrap_from`
**Description**: Bootstrap configuration based on a predefined machine template.

**Parameters**:
- `machine` (str): Parameter for machine

**Returns**: Returns list

### `jm_bootstrap_list`
**Description**: List all bootstrap templates available.

**Returns**: Returns list

### `jm_reset`
**Description**: Reset manager to a clean state by destroying all pipelines and config.

**Returns**: Returns list

### `jm_list_pipelines`
**Description**: List all current pipelines under management.

**Returns**: Returns list

### `jm_cd`
**Description**: Set the working pipeline context.

**Parameters**:
- `pipeline_id` (str): Parameter for pipeline_id

**Returns**: Returns list

### `jm_list_repos`
**Description**: List all registered repositories.

**Returns**: Returns list

### `jm_add_repo`
**Description**: Add a repository path to the manager.

**Parameters**:
- `path` (str): Parameter for path
- `force` (bool, optional): Parameter for force (default: False)

**Returns**: Returns list

### `jm_remove_repo`
**Description**: Remove a repository from configuration.

**Parameters**:
- `repo_name` (str): Parameter for repo_name

**Returns**: Returns list

### `jm_promote_repo`
**Description**: Promote a repository to higher priority.

**Parameters**:
- `repo_name` (str): Parameter for repo_name

**Returns**: Returns list

### `jm_get_repo`
**Description**: Get detailed information about a repository.

**Parameters**:
- `repo_name` (str): Parameter for repo_name

**Returns**: Returns list

### `jm_construct_pkg`
**Description**: Generate a new package skeleton by type.

**Parameters**:
- `pkg_type` (str): Parameter for pkg_type

**Returns**: Returns list

### `jm_graph_show`
**Description**: Print the resource graph to the console.

**Returns**: Returns list

### `jm_graph_build`
**Description**: Construct or rebuild the graph with a given sleep delay.

**Parameters**:
- `net_sleep` (float): Parameter for net_sleep

**Returns**: Returns list

### `jm_graph_modify`
**Description**: Modify the current resource graph with a delay between operations.

**Parameters**:
- `net_sleep` (float): Parameter for net_sleep

**Returns**: Returns list
## Examples

### 1. Pipeline Creation and Basic Management
```
I need to set up a new HPC pipeline called "performance_test" for running performance benchmarks. Initialize the Jarvis configuration, create the pipeline, and add the IOR package to it.
```

**Tools called:**
- `jm_create_config` - Initialize Jarvis configuration directories
- `create_pipeline` - Create new pipeline environment  
- `append_pkg` - Add IOR package to the pipeline

This prompt will:
- Initialize Jarvis with proper configuration directories
- Create a new pipeline for performance testing
- Add the IOR I/O benchmarking package with default settings
- Prepare the pipeline for further configuration and execution

### 2. Package Configuration and Environment Setup
```
For the "performance_test" pipeline, configure the IOR package to use 16 processes, then build the pipeline environment and execute it.
```

**Tools called:**
- `get_pkg_config` - Retrieve current IOR package configuration
- `configure_pkg` - Configure IOR package with 16 processes
- `build_pipeline_env` - Build pipeline execution environment
- `run_pipeline` - Execute the configured pipeline

This prompt will:
- Display current package configuration settings
- Update IOR package to use 16 processes for the benchmark
- Build the complete pipeline environment with all dependencies
- Execute the pipeline end-to-end with proper resource allocation

### 3. Repository Management and Package Discovery
```
Show me all available repositories in the system, add a new custom repository at "/path/to/custom/repo", and promote it to higher priority for package resolution.
```

**Tools called:**
- `jm_list_repos` - List all configured repositories
- `jm_add_repo` - Add custom repository to the system
- `jm_promote_repo` - Promote repository to higher priority

This prompt will:
- Display all currently configured repositories with their status
- Add a new repository to expand available packages
- Promote the custom repository for prioritized package discovery
- Update the system configuration with the new repository settings

### 4. Advanced Pipeline Operations and Package Management
```
Create a comprehensive workflow: list all existing pipelines, load pipeline "data_analysis", show configuration of the "pandas" package, update its memory settings to 32GB, and rebuild the environment.
```

**Tools called:**
- `jm_list_pipelines` - List all existing pipelines
- `load_pipeline` - Load the data_analysis pipeline
- `get_pkg_config` - Show pandas package configuration  
- `configure_pkg` - Update memory settings for pandas package
- `update_pipeline` - Re-apply configurations to all packages
- `build_pipeline_env` - Rebuild environment with new settings

This prompt will:
- Show all available pipelines in the system
- Load the specific data analysis pipeline
- Display current pandas package configuration
- Update memory allocation for the pandas package
- Ensure all package configurations are current and consistent
- Rebuild the execution environment with optimized settings

### 5. Resource Monitoring and System Management
```
I need to monitor system resources and manage pipeline contexts. Show me the current resource graph, build a new resource graph with 2.0 second intervals, and set "hpc_workflow" as the current working pipeline.
```

**Tools called:**
- `jm_graph_show` - Display current resource graph frames
- `jm_graph_build` - Build resource graph with custom intervals
- `jm_cd` - Change current pipeline context

This prompt will:
- Display the current system resource graph for monitoring
- Build an updated resource graph with 2.0 second monitoring intervals
- Set the HPC workflow as the active working pipeline
- Provide comprehensive system resource visibility and management

### 6. Package Lifecycle Management
```
For troubleshooting purposes, I need to unlink the "tensorflow" package from "ml_pipeline" without deleting files, then completely remove the "outdated_tool" package, and finally destroy the "test_pipeline" entirely.
```

**Tools called:**
- `unlink_pkg` - Unlink tensorflow package while preserving files
- `remove_pkg` - Completely remove outdated_tool package  
- `destroy_pipeline` - Destroy entire test pipeline and cleanup

This prompt will:
- Safely unlink tensorflow package preserving data and configurations
- Completely remove outdated tools with full cleanup
- Destroy test pipeline and clean up all associated resources
- Provide safe package and pipeline lifecycle management

### 7. Configuration Management and Bootstrapping
```
Initialize a new Jarvis setup by listing available bootstrap templates, bootstrap from "cluster_template", save the configuration, and create a package skeleton for a "mpi_application" type.
```

**Tools called:**
- `jm_bootstrap_list` - List available bootstrap machine templates
- `jm_bootstrap_from` - Bootstrap configuration from cluster template
- `jm_save_config` - Save current configuration state
- `jm_construct_pkg` - Create MPI application package skeleton

This prompt will:
- Show all available system templates for quick setup
- Bootstrap Jarvis configuration from a cluster-optimized template  
- Persist the configuration for future use
- Generate a complete package skeleton for MPI applications
- Provide rapid deployment capabilities for HPC environments

### 8. System Reset and Repository Information
```
I need to perform maintenance on the Jarvis system: get detailed information about the "hpc_tools" repository, then perform a complete system reset to clean state for testing.
```

**Tools called:**
- `jm_get_repo` - Get detailed repository information
- `jm_reset` - Reset entire Jarvis system to clean state

This prompt will:
- Retrieve comprehensive information about the specified repository
- Perform complete system reset destroying all pipelines and configurations
- Provide clean slate for testing and development
- Ensure system maintenance and troubleshooting capabilities

### 9. Advanced Resource Graph Management
```
Set up comprehensive resource monitoring by modifying the resource graph with 1.5 second intervals, then display the updated graph frames for analysis.
```

**Tools called:**
- `jm_graph_modify` - Modify resource graph with custom intervals
- `jm_graph_show` - Display current resource graph frames

This prompt will:
- Modify the existing resource graph with optimized monitoring intervals
- Display the updated resource graph for comprehensive system analysis
- Provide advanced resource monitoring and performance tracking
- Enable real-time system resource visualization and management

### 10. Complete Pipeline Lifecycle with Configuration Management
```
Load my existing configuration, set hostfile to "/etc/hosts.cluster", remove the deprecated "old_repo" repository, and create a comprehensive workflow pipeline.
```

**Tools called:**
- `jm_load_config` - Load existing Jarvis configuration
- `jm_set_hostfile` - Configure hostfile for deployments
- `jm_remove_repo` - Remove deprecated repository
- `create_pipeline` - Create new comprehensive workflow pipeline

This prompt will:
- Load existing system configuration and settings
- Configure cluster hostfile for distributed deployments
- Remove outdated repository to clean system state
- Create a new pipeline ready for comprehensive workflow management
- Provide complete system configuration and lifecycle management

### More Examples
##### 1. **Initialize Jarvis**

The first step is to initialize Jarvis. This prepares the system for interaction.

```bash
# Command to initialize Jarvis
Query: Initialize jarvis with configur, private and shared dir as " . /jarvis—pipelines'
```

**Output Screenshot**

![alt text](<./docs/assets/Screenshot 2025-05-15 160800.png>)

---

##### 2. **Create Pipeline (`ior_test`) and append package**

Create a new pipeline named `ior_test` and append package to it. This will be used for testing purposes.

```bash
# Command to create a pipeline
Query: create a pipeline called ior_test and append package ior to it
```

**Output Screenshot**

![alt text](<./docs/assets/Screenshot 2025-05-15 162219.png>)

---

##### 3. **Change Configuration of Added Package**

You can also see and modify the configuration of the package you've added to the pipeline.

```bash
# Command to change the configuration
Query: show the configuration of ior in ior_test
```

**Output Screenshot**

![alt text](<./docs/assets/Screenshot 2025-05-15 162322.png>)

```bash
# Command to change the configuration
Query: update the nprocs to 8 for package ior in pipeline ior_test
```

![alt text](<./docs/assets/Screenshot 2025-05-15 162545.png>)
---

##### 4. **Build Environment for `ior_test` Pipeline**

After configuring the pipeline, you can build the environment for `ior_test`.

```bash
# Command to build the environment
Query: Build environment for pipeline ior_test 
```

**Output Screenshot**

![alt text](<./docs/assets/Screenshot 2025-05-15 162922.png>)

---

##### 5. **Run the Pipeline (`ior_test`)**

Finally, you can run the pipeline to see everything in action.

```bash
# Command to run the pipeline
Query: select the pipeline ior_test and run it
```

**Output Screenshot**

![alt text](<./docs/assets/Screenshot 2025-05-15 163023.png>)

---

or **write below** to create pipeline, append package to it and run it:
```bash
Query: create a pipeline called ior_test_2. Add package ior with nprocs set to 16. After adding, set the pipeline ior_test_2 as current and build environment for it and run it.
```
![alt text](<./docs/assets/Screenshot 2025-05-15 163759.png>)

---