---
title: Authoring plugins, MCPs, skills and hooks
---

# Authoring components

A plugin packages any combination of skills, MCP configuration, agents,
commands and hooks. CLIO Kit maintains components in this repository and can
index plugins or whole marketplaces maintained elsewhere.

Install the launcher from the checkout before using the commands below:

```bash
uv tool install --force --reinstall ".[verification]"
```

## Create a plugin

```bash
clio-kit plugin init my-plugin
clio-kit plugin validate my-plugin
claude plugin validate my-plugin --strict
```

The scaffold contains `.claude-plugin/plugin.json`, a sample skill and evaluation
scenarios. Edit the manifest's name, description and version, then add the
components you need. The last command checks Claude Code's native rules; it
requires that client. Portable skills can also be installed independently.

## Add a skill

Place each skill in `my-plugin/skills/<name>/SKILL.md`. A minimal example:

```markdown
---
name: inspecting-simulation-output
description: 'Use when inspecting an unfamiliar simulation dataset. Triggers on "inspect simulation output". Not for plotting; use summarizing-and-plotting-results.'
---

Discover the available scientific I/O tools. Inspect the file metadata and
dataset shapes first, then read a bounded slice. Report the slice bounds and
units; do not treat a sample as a whole-dataset summary.
```

Add `evals.md` beside it with concrete prompts, expected results and failure
cases, then validate:

```bash
clio-kit skill validate my-plugin/skills/inspecting-simulation-output
clio-kit plugin validate my-plugin
```

For skills maintained here, use `skills/clio-<bundle>-skills/skills/<name>/`.
Configure required MCP servers separately and test the procedure with actual
tool results. Skill validation alone does not establish agent performance.

## Add an MCP server

For an external server, add `.mcp.json` at the plugin root. For example, a plugin
using the installed CLIO Kit HDF5 server can declare:

```json
{
  "mcpServers": {
    "hdf5": {"command": "clio-kit", "args": ["mcp-server", "hdf5"]}
  }
}
```

For your own server, use its real executable and arguments and document its
installation requirements. Its implementation may use any language; the client
must be able to run or connect to it. Verify initialization, tool discovery and
a representative tool call in the target client.

To ship a maintained server with CLIO Kit, add `mcp-servers/<name>/`, a
`clio-server.toml`, the runtime's lock file and its tests. Register it in
`mcp-server-versions.toml` and generate its manifests. Python, Node and Go are
supported; see the
[hosted-server requirements](https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md#contributing-a-server-in-another-language).

## Add a hook

```bash
clio-kit plugin init my-hook-plugin --hook
clio-kit plugin validate my-hook-plugin
claude plugin validate my-hook-plugin --strict
```

This adds `hooks/hooks.json` and a Python SessionStart handler that prints
context without modifying files. Edit the handler for the intended behavior;
the scaffold requires `python3` on the user's PATH.

Maintained and community plugins use the same hook validation. A hook can be
packaged with other components or in its own plugin. Adding a community entry
only lists the plugin; the supported client installs and executes its hooks.

Hooks are host-specific executable behavior. Our native manifests target Claude
Code; portable skills do not make these hooks portable to every agent. Validate
with the target client's rules, review commands before installation, and test
multiple enabled plugins together for duplicated actions or conflicting effects.
Do not rely on an execution order between independent plugins.

CLIO Kit checks configuration without running handlers. Its acceptance test
exercises SessionStart, successful tool completion and a blocked tool call in
the real Claude Code runtime:

```bash
uv run --frozen python scripts/verify_plugin_hooks.py
```

This establishes those command-hook paths, not every event or handler type.
See the [hook reference](https://github.com/iowarp/clio-kit/blob/main/community/README.md#hooks)
for supported configuration forms and isolated client trials.

## Index an external contribution

```bash
clio-kit plugin submit my-plugin --repo owner/name
```

This prints an entry for `community/entries/<name>.toml`. Review it and open a
pull request, or use `--open-pr` for the submission command to do so. For an
entire marketplace, pass its directory and `--kind marketplace` instead.
Implementations stay upstream. Source types, pinning and local installation
trials are documented in the
[community guide](https://github.com/iowarp/clio-kit/blob/main/community/README.md).
