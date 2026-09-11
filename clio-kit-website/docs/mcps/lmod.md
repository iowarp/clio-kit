---
title: Lmod MCP
description: "Lmod MCP - Environment Module Management for LLMs with comprehensive module operations"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Lmod"
  icon="📦"
  category="System Management"
  description="Lmod MCP - Environment Module Management for LLMs with comprehensive module operations"
  version="3.0.1"
  actions={["module_list", "module_avail", "module_show", "module_spider", "module_save", "module_restore", "module_savelist"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={["lmod", "environment-modules", "module-management", "hpc", "scientific-computing", "supercomputing", "cluster-computing", "module-system"]}
  license="BSD-3-Clause"
  tools={[{"name": "module_list", "description": "List all currently loaded environment modules.", "function_name": "module_list"}, {"name": "module_avail", "description": "Search for available modules, optionally filtered by name pattern.", "function_name": "module_avail"}, {"name": "module_show", "description": "Display detailed information about a specific module.", "function_name": "module_show"}, {"name": "module_spider", "description": "Search the entire module tree comprehensively for matching modules.", "function_name": "module_spider"}, {"name": "module_save", "description": "Save currently loaded modules as a named collection.", "function_name": "module_save"}, {"name": "module_restore", "description": "Restore a previously saved module collection.", "function_name": "module_restore"}, {"name": "module_savelist", "description": "List all saved module collections.", "function_name": "module_savelist"}]}
>

{/* clio-kit:usage:start */}

### Discover the available environment

Use `module_list` to inspect modules loaded in the MCP process. Search visible
modules with `module_avail`, use `module_spider` for hierarchical discovery,
and inspect prerequisites and settings with `module_show`.

The user tool surface does not load, unload or swap modules. A child process
cannot establish the environment of your shell or a later workload. For a
JARVIS execution, resolve software through Spack and pass exact load specs to
`jarvis_run`.

### Saved collections

`module_save` writes a named collection; `module_savelist` lists collections.
`module_restore` acts in its child process and does not establish the environment
of subsequent tools. Treat a collection as an environment record and verify the
actual workload environment separately.

Lmod and its configured module tree must be present on the host. An MCP
connection alone does not confirm either prerequisite.

### Native backend configuration

Set `LMOD_CMD` to the site Lmod executable and `MODULEPATH` to trusted modulefiles. The server uses Bash to evaluate Lmod and retains collection changes for subsequent calls in the same MCP process. It does not change your parent shell or other MCP servers.

{/* clio-kit:usage:end */}

</MCPDetail>
