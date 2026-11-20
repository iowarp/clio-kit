---
title: Spack MCP
description: "Structured Spack discovery and installation tools for scientific agents"
---

import MCPDetail from '@site/src/components/MCPDetail';

<MCPDetail
  name="Spack"
  icon="🛠️"
  category="Utilities"
  description="Structured Spack discovery and installation tools for scientific agents"
  version="3.0.0"
  actions={["spack_find", "spack_locate", "spack_search", "spack_info", "spack_prepare_site", "spack_site_query"]}
  platforms={["claude", "cursor", "vscode"]}
  keywords={[]}
  license="BSD-3-Clause"
  tools={[{"name": "spack_find", "description": "List installed Spack packages matching an optional constraint. No matches is a successful result with count=0 and packages=[].", "function_name": "spack_find"}, {"name": "spack_locate", "description": "Resolve one unique installed Spack spec. Copy spack_locate.output.load_spec unchanged into one element of jarvis_run.input.spack_specs; do not derive or pass an executable path from the returned prefix. An absent package returns the structured not_installed error, whose detail now says whether a recipe is available to install (through a jarvis install job) or exists in no registered repo at all.", "function_name": "spack_locate"}, {"name": "spack_search", "description": "Search recipe AVAILABILITY across every registered Spack repo -- broader than spack_find/spack_locate, which only see what is already installed. Answers 'does a recipe exist', 'in which repo', and 'is it already installed' in one call. No matches is a successful result with count=0.", "function_name": "spack_search"}, {"name": "spack_info", "description": "Describe one recipe: versions, variants, and description. Probes `spack info` first; if that subcommand is unavailable or fails on this deployment, falls back to statically parsing the recipe's package.py and marks the result degraded=true with degraded_reason explaining why -- never silently. A package absent from every registered repo returns the structured recipe_not_found error.", "function_name": "spack_info"}, {"name": "spack_prepare_site", "description": "Set up a CLIO Spack on a clio-relay site that reuses what the site already has. It shallow-clones the Spack release the site runs (pinned by commit) into <site_home>/spack, points upstreams.yaml at the site install tree, runs compiler find, registers build tools with external find --not-buildable, sets packages:mpi:buildable:false, records module-only MPIs as candidate externals and writes packages:all:compiler with full toolchains (cc, cxx, fc) first, the upstream MPI's compiler ahead, plus build_jobs. When several full toolchains qualify, report.compiler.choice_needed asks the person once. MPI is never externalized from PATH. Config lives in the checkout and SPACK_DISABLE_LOCAL_CONFIG keeps ~/.spack out. Idempotent; it never installs, builds or installs jarvis-cd. The first call returns the relay exec call (with a spack plan that a guarded relay site asks the person to approve) and the approval text; the second, with relay_result, returns the report, including the JARVIS_SPACK_ROOT and JARVIS_SPACK_CONFIG to export. needs_approval means retry the same exec once the person answers; declined means stop.", "function_name": "spack_prepare_site"}, {"name": "spack_site_query", "description": "Research a clio-relay site's CLIO Spack (prepared by spack_prepare_site): find (installed and upstream packages, --json), info (one recipe), spec (spack spec -I -l: what is installed, upstream, external or still to build) or modules (module -t avail). Read-only, short login-node work; Spack runs as env SPACK_DISABLE_LOCAL_CONFIG=1 <spack> <verb>, the shape clio-relay's guard accepts. The first call returns the relay exec call; the second, with relay_result, returns the parsed answer.", "function_name": "spack_site_query"}]}
>

{/* clio-kit:usage:start */}

### Discover before installing

Use `spack_find` to check installed packages. Empty matches are a successful
result. Use `spack_search` for recipe availability and `spack_info` for versions
and variants. Install a missing package through a JARVIS install job
(`jarvis ppl install +submit`); `spack_install` is an admin tool for a local
Spack.

### Remote HPC sites

`spack_prepare_site` sets up a CLIO Spack on a clio-relay site that reuses the
site's Spack as an upstream, and `spack_site_query` researches it (find, info,
spec, modules). Both reach the site only through clio-relay `exec`, do light
login-node work only, and never install or build. On a guarded relay site the
research runs as `env SPACK_DISABLE_LOCAL_CONFIG=1 .clio-relay/spack/bin/spack
find|info|spec`, and preparation carries a `spack` plan the person approves;
`needs_approval` means send the same exec again after they answer, and
`declined` means stop. Preparation writes `packages:all:compiler` with full
toolchains (cc, cxx, fc) first. When `report.compiler.choice_needed` is set,
ask the person once and prepare again with `compiler=`, which is also the
lever when a build fails in a compiler check.

### Hand software to an execution

Use `spack_locate` to resolve the exact installed identity. Pass its unchanged
`output.load_spec` into JARVIS `jarvis_run`'s top-level `spack_specs` list.
A prefix or inferred executable path is not a substitute for that identity.
Spack must be installed and configured on the MCP host.

{/* clio-kit:usage:end */}

</MCPDetail>
