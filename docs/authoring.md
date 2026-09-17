---
title: Authoring plugins, MCPs, skills and hooks
---

# Authoring components

MCPs, skills, agents and hooks are individual components. A plugin bundles the
components a workflow needs, optionally including commands; it does not need
every component type. Contribute components separately or compose a plugin.
CLIO Kit maintains components here and indexes packages or whole marketplaces
maintained elsewhere. See [catalogue types](plugins.md#catalogue-types-and-installation-packages)
for how components, collections and workflow plugins are displayed.

For maintained plugins that reuse existing components, see
[Choosing and composing plugins](plugins.md). Folder-based packages need no
central inventory entry. Optional generated task plugins use `[workflows.*]`;
the six primary bundles keep their separate coverage rule.

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

For a maintained example combining MCP dependencies, a task skill, an evidence
reviewer and a hook, see [Scientific dataset report](plugins.md#scientific-dataset-report)
and `plugins/clio-dataset-report/`. External plugins can package the same
component types in their own repository and use the indexing route below.

## Add a component folder to CLIO Kit

Add a package under the directory that describes its purpose:

| Directory | Typical contents inside the package |
| --- | --- |
| `plugins/my-plugin/` | Any combination, including `.mcp.json` or plugin dependencies |
| `skills/my-skills/` | `skills/<skill-name>/SKILL.md` and evaluation scenarios |
| `agents/my-agents/` | `agents/<agent-name>.md` |
| `hooks/my-hooks/` | `hooks/hooks.json` and its handlers |

Each package needs `.claude-plugin/plugin.json` with a name matching its folder,
a description and a semantic version. Dependencies are optional. These are
installable packages, not loose files placed directly at the repository root.
The manifest is a Claude Code distribution requirement, not a claim that an
individual skill, agent or hook is a workflow plugin.
Use the scaffold, then remove components you do not need:

```bash
clio-kit plugin init plugins/my-plugin --agent --hook
# Edit the components and evaluation scenarios.
clio-kit plugin validate plugins/my-plugin
claude plugin validate plugins/my-plugin --strict
```

Add the package folder to your PR. CI discovers it and updates both catalogues
for validation; contributors do not run generator scripts or hand-edit the index.
Website `npm start` and `npm run build` also sync automatically. Handwritten
manifests are preserved and discovery does not execute component commands.

After a merge/push to a supported branch, the catalogue sync workflow commits
the generated indices so native clients can discover the new package. Until
that workflow succeeds, the Git-hosted native index may still reflect the
previous state. This uses the existing bot's repository write permission; branch
rules must allow its generated commit. Fork PRs only run read-only validation.

For native client testing before opening the PR, use the optional CLI sync:

```bash
clio-kit marketplace sync --root .
claude plugin marketplace add "$PWD"
claude plugin install my-plugin@clio-kit
```

Existing users refresh their marketplace and update the installed plugin after
changes merge. Increase the package version when changing its installed contents.
Deleting a package folder removes its catalogue entry on the next automatic sync; it does
not uninstall existing client copies.

Package names must be unique across the four directories and external entries.
Portable skill names must also be unique, because `clio-kit skill install` selects
by skill name. Dependencies may name other local packages or generated CLIO
components; missing names, cycles and external dependencies are rejected.
Linked files are not supported. `clio-` names remain reserved for maintained
packages; validate those explicitly with `--maintained`.

Skills and their resources join the wheel automatically. Agents, hooks and MCP
configuration travel through native plugin installation; they do not become
portable merely by being indexed. Bundled MCP configuration runs its declared
command or connects to its URL. Hosting a server in `mcp-servers/` still uses the
runtime descriptor, lock, registration and tests described below.

Run the complete folder-contribution acceptance check with:

```bash
uv run --frozen python scripts/verify_local_components.py --output /tmp/clio-local-components
```

It creates disposable packages, checks real native skill/agent/hook loading and
MCP calls, then exercises updates, removal and portable wheel installation.
It uses scripted model responses, so it validates integration rather than model
reasoning. No public PR is opened.

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
