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
uv tool install --force --reinstall --editable ".[verification]"
```

## Create a plugin

```bash
clio-kit plugin init my-plugin
clio-kit plugin validate my-plugin
claude plugin validate my-plugin --strict
```

The scaffold contains `.claude-plugin/plugin.json`, a sample skill named
`<plugin>-workflow` and evaluation scenarios. Edit the manifest's description,
author and version, then add the components you need. With `--mcp-command` the
wrapped server is registered under the plugin's name. The last command checks Claude Code's native rules; it
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
CI runs `clio-kit marketplace sync` before the root test suite. To run that
suite locally (`uv run --frozen pytest tests -q`), sync first with the command
below; otherwise the catalogue tests report a stale index.
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
Linked files are not supported; `plugin validate` reports each one. `clio-` names
remain reserved for maintained packages; scaffold and validate those explicitly
with `--maintained` (`clio-kit plugin init plugins/clio-my-task --maintained`).
Skills in a `clio-` package must also declare `metadata.bundle` and
`metadata.eval-status`, and each needs an evaluation case; see
[Add a skill](#add-a-skill).

Release builds automatically package skills and their resources into individual
artifacts, and packages into separate native payloads. The launcher carries their
catalogue and hashes. Agents, hooks and MCP configuration travel through native plugin installation; they do not become
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
description: 'Use when inspecting an unfamiliar simulation dataset. Triggers on "inspect simulation output". Not for plotting; use results-summary.'
metadata:
  bundle: my-plugin
  servers: clio-hdf5
  provenance: designed
  eval-status: scenarios-recorded
---

Discover the available scientific I/O tools. Inspect the file metadata and
dataset shapes first, then read a bounded slice. Report the slice bounds and
units; do not treat a sample as a whole-dataset summary.
```

`metadata` values are strings. `bundle` names the package or bundle that lists
the skill, `servers` the MCP plugins it expects (`none` if it calls no tools).
`eval-status` records how far the skill has been checked and must be one of,
weakest first: `untested`, `scenarios-recorded`, `trigger-checked`,
`smoke-checked`, `eval-run`.

Add a nonempty `evals.md` beside it with concrete prompts, expected results and
failure cases:

```markdown
# Evals - inspecting-simulation-output

## S1 - unfamiliar HDF5 file

Setup: Prompt: "Inspect simulation output in run.h5 and tell me what it holds."

Expected:

- Lists the datasets with their shapes and units before reading any values.
- Reads a bounded slice and states its bounds.

Without the skill: reads a whole dataset, or reports a sample as the whole file.
```

Then validate:

```bash
clio-kit skill validate my-plugin/skills/inspecting-simulation-output
clio-kit plugin validate my-plugin
```

For skills maintained here, use `skills/clio-<bundle>-skills/skills/<name>/`,
set `metadata.bundle: clio-<bundle>`, and validate with `--maintained`, which
makes `metadata.bundle` and `metadata.eval-status` blocking:

```bash
clio-kit skill validate --maintained skills/clio-<bundle>-skills/skills/<name>
```

A skill added to a maintained bundle also needs an evaluation case in
`evals/codex_cases.py` and a version bump in `mcp-server-versions.toml`
(`[bundles.clio-<bundle>].version`; the skills plugin's manifest is generated
from it, so editing the manifest is reverted by the next sync). The
[contributor guide](https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md#contributing-a-skill)
lists every step.
Use concise capability names such as `dataset-explore`, `data-clean` and
`slurm-script`, matching the folder and frontmatter. Imported adaptations retain
their upstream provenance and distinct names.
Configure required MCP servers separately and test the procedure with actual
tool results. Skill validation alone does not establish agent performance.

For a live Codex comparison using an existing sign-in, run a selected case in a
fresh output directory:

```bash
uv run --frozen python evals/codex_eval.py --skill data-clean --output /tmp/clio-skill-eval
```

The runner compares the same task and tools with and without the installed
skill, recording model usage, tool calls and independent artifact checks. It
uses subscription quota; `--model` selects the model and `--list` previews tasks
without a model call. Omitting `--skill` requests all cases. Read the resulting
artifacts before judging scientific quality: passing execution checks alone
does not establish usefulness, and a single pair does not establish token savings.
The default run stops scheduling at 250,000 observed uncached input plus output
tokens (`--max-uncached-tokens`); already-running cases can exceed that limit.
Authentication, quota and sandbox failures also stop further cases. Native Claude
agents and hooks require their separate supported-client acceptance checks.

For the native scientific planner, use an existing Claude sign-in:

```bash
uv run --frozen python evals/claude_planner_eval.py --output /tmp/clio-planner-eval
```

This installs the agent in temporary profiles and repeats missing-reader,
preview-only and supported-reader scenarios. The agent has read-only tools.
The planner selects Claude Sonnet 5 with medium reasoning effort for scientific
planning, using the client's [native agent settings](https://code.claude.com/docs/en/sub-agents#supported-frontmatter-fields).
The evaluation reads those settings from the installed agent; `--model` allows
comparison with another model without changing the plugin.
Inspect each `answer.md`: successful execution does not establish a correct plan.
Use `--fixture /path/to/saved/project` to also repeat a trusted earlier fixture;
`--case` and `--repeat` limit model usage. Reports remain in the output directory.

Select a workflow's predefined evaluation case with `--plugin`:

```bash
uv run --frozen python evals/codex_eval.py --plugin clio-dataset-report --modes skill --output /tmp/clio-plugin-eval
uv run --frozen python evals/codex_report.py /tmp/clio-plugin-eval
```

The runner keeps the installer-generated Codex project configuration and records
any components excluded by the portable route, such as Claude-format agents and
hooks. `--list` previews coverage without needing an output directory. For a
separate explicit-invocation smoke test, use `--modes skill --invoke-skill`;
automatic skill selection remains a different test.

Use `--mcp-config /path/to/backends.toml` for site-specific settings in a Codex
`[mcp_servers.<name>]` table: stdio `command`, `args`, `env` and timeouts are
supported. The same overrides apply to both comparison arms. Keep credentials
out of that file; temporary configuration may be retained as local evidence.
Temporary authentication links are removed after each run, including failures.
Incomplete model runs return a nonzero exit code.

Reports distinguish task-outcome checks from skill/MCP activation. To recheck
trusted saved fixtures after correcting a checker, add `--recheck-artifacts` to
the report command. This can execute generated fixture code and writes a separate
`rechecked-summary`, preserving the original results. Drafts and prerequisite
checks still do not establish operational scientific workflows.

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
`mcp-server-versions.toml` and generate its manifests. Include its icon in
`[icons]` and any executable or environment checks in `[prerequisites.<name>]`;
these feed the website and `clio-kit doctor`, including partial installations.
The contributor guide's
[Adding a New MCP Server](https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md#adding-a-new-mcp-server)
lists every file to touch and ends with a green root suite.
Python, Node and Go are supported; see the
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

Hooks are host-specific executable behavior. The native marketplace uses Claude
Code conventions; other clients use the explicit adapters described below. Validate
with the target client's rules, review commands before installation, and test
multiple enabled plugins together for duplicated actions or conflicting effects.
Do not rely on an execution order between independent plugins.

CLIO Kit checks configuration without running handlers. Its acceptance test
exercises SessionStart, successful tool completion and a blocked tool call in
the real Claude Code runtime:

```bash
uv run --frozen python scripts/verify_plugin_hooks.py --output /tmp/clio-hook-acceptance
```

Use a new output directory for each run.

This establishes those command-hook paths, not every event or handler type.
See the [hook reference](https://github.com/iowarp/clio-kit/blob/main/community/README.md#hooks)
for supported configuration forms and isolated client trials.

## Client-specific hooks

A package can ship a hook for each supported host alongside the shared skills,
MCP configuration and agent instructions:

| File | Host |
| --- | --- |
| `hooks/hooks.json` | Claude Code event configuration |
| `hooks/codex.json` | Codex event configuration (command handlers) |
| `hooks/opencode.js` | OpenCode plugin with a default exported async function |

The project installer selects only the requested host's adapter. JSON hooks may
reference `${CLAUDE_PLUGIN_ROOT}` or `${CLIO_PLUGIN_ROOT}`; OpenCode adapters
receive a `CLIO_PLUGIN_ROOT` constant pointing to the retained package. Handlers
must stay inside the package and shared helpers should be reused. Installation
validates declarations without executing them; client execution is a separate
trust decision. Codex requires `/hooks` review. OpenCode `--pure` disables these
plugins. Missing adapters produce an installation error, unless the user opts
into `--components-only`.

Dataset Report demonstrates all three adapters reusing the same verifier. They
watch each host's native write/edit operations, not arbitrary shell writes. The
Codex adapter extracts file paths from an apply_patch payload; OpenCode adds the
verifier's feedback to the completed tool output. This feedback is evidence
review, not a sandbox or a guarantee that every file mutation was observed.

Keep native custom agent options and slash commands in their supported client.
The shared agent adapter accepts only Read/Glob/Grep declarations; model overrides
remain host-specific. Test the installed role and permissions in the actual client.

To exercise all external routes with the project installers and real MCP queries:

```bash
uv run --frozen python scripts/verify_client_marketplace.py --output /tmp/clio-client-marketplace
```

This uses temporary Git/registry fixtures and makes no public submissions.
It verifies configuration and actual stdio calls, not model-driven skill use.

## Index an external contribution

```bash
clio-kit plugin submit my-plugin --repo owner/name
```

This prints the TOML entry for `community/entries/<name>.toml` on standard
output and its guidance on standard error, so `> <name>.toml` (or
`--output <name>.toml`) saves exactly the entry. The command refuses a manifest
that still carries the scaffold's placeholder description or author. Review the
entry and open a pull request, or use `--open-pr` for the submission command to
do so. For an
entire marketplace, pass its directory and `--kind marketplace` instead.
Implementations stay upstream. Source types, pinning and local installation
trials are documented in the
[community guide](https://github.com/iowarp/clio-kit/blob/main/community/README.md).
