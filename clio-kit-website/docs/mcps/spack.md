---
title: Spack MCP
description: "Structured Spack discovery and installation tools for scientific agents"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Spack"
  icon="🔧"
  category="Utilities"
  description="Structured Spack discovery and installation tools for scientific agents"
  version="2.3.1"
  actions={["spack_find", "spack_locate", "spack_search", "spack_info", "spack_install"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={[]}
  license="BSD-3-Clause"
  tools={[{"name": "spack_find", "description": "List installed Spack packages matching an optional constraint. No matches is a successful result with count=0 and packages=[].", "function_name": "spack_find"}, {"name": "spack_locate", "description": "Resolve one unique installed Spack spec. Copy spack_locate.output.load_spec unchanged into one element of jarvis_run.input.spack_specs; do not derive or pass an executable path from the returned prefix. An absent package returns the structured not_installed error, whose detail now says whether a recipe is available to install (call spack_install) or exists in no registered repo at all.", "function_name": "spack_locate"}, {"name": "spack_search", "description": "Search recipe AVAILABILITY across every registered Spack repo -- broader than spack_find/spack_locate, which only see what is already installed. Answers 'does a recipe exist', 'in which repo', and 'is it already installed' in one call. No matches is a successful result with count=0.", "function_name": "spack_search"}, {"name": "spack_info", "description": "Describe one recipe: versions, variants, and description. Probes `spack info` first; if that subcommand is unavailable or fails on this deployment, falls back to statically parsing the recipe's package.py and marks the result degraded=true with degraded_reason explaining why -- never silently. A package absent from every registered repo returns the structured recipe_not_found error.", "function_name": "spack_info"}, {"name": "spack_install", "description": "Install one Spack spec with explicit reusable or fresh concretization. Runs synchronously (streaming/task augmentation is deferred to the kit tasks-semantics slice, SEP-2663) with a configurable timeout; captures the full build log to disk and returns its path plus a bounded tail, and the install prefix on success. A missing recipe, a failed build, and a timeout are distinct typed errors (recipe_not_found / build_failure / timed_out), each naming the recovery affordance (searched repos / log tail / log path).", "function_name": "spack_install"}]}
>

{/* clio-kit:usage:start */}

### Discover before installing

Use `spack_find` to check installed packages. Empty matches are a successful
result. Use `spack_search` for recipe availability and `spack_info` for versions
and variants. Install a missing package with `spack_install` only when the work
requires it and choose reuse versus fresh concretization explicitly.

### Hand software to an execution

Use `spack_locate` to resolve the exact installed identity. Pass its unchanged
`output.load_spec` into JARVIS `jarvis_run`'s top-level `spack_specs` list.
A prefix or inferred executable path is not a substitute for that identity.
Spack must be installed and configured on the MCP host.

{/* clio-kit:usage:end */}

</MCPDetail>
