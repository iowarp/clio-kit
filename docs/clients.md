---
title: Set up your agent
---

# Install CLIO Kit for your agent

Use this guide after choosing a component in the [catalogue](/catalogue).
Start with the launcher, choose **one** client route, then verify an actual tool
call. Skills provide instructions; MCP servers provide tools. Installing one does
not automatically configure the other.

## 1. Install the launcher

For the current source checkout:

```bash
git clone https://github.com/iowarp/clio-kit.git
cd clio-kit
uv tool install --force --reinstall --editable ".[verification]"
clio-kit mcp-servers
```

Requires Git, uv and Python 3.10 or newer. Keep the checkout available. If the
command is not found, run `uv tool update-shell` and reopen your terminal/client.
First server starts install isolated dependencies. HPC applications may need
additional site software. The [installation guide](installation.md) covers
released launchers, selective downloads, offline preparation and updates.

All project-install commands below run **from the checkout**. Replace
`/path/to/project` with your working project, or add `--root /path/to/clio-kit`
when running elsewhere. Add `--dry-run` to preview changes first.

## 2. Choose your client

| Client | Project configuration written by Kit | Skills installed to |
| --- | --- | --- |
| Codex CLI / IDE | `.codex/config.toml` | `.agents/skills` |
| Claude Code (portable route) | `.mcp.json` | `.claude/skills` |
| OpenCode | `opencode.json` | `.opencode/skills` |
| Cursor | `.cursor/mcp.json` | `.cursor/skills` |
| Antigravity | `.agents/mcp_config.json` | `.agents/skills` |
| VS Code / GitHub Copilot | `.vscode/mcp.json` | `.github/skills` |

### Component support

One catalogue and contribution route serve all six project adapters. Indexed
GitHub, Git URL, Git subdirectory and npm packages, including entries imported
from another marketplace, use the same `clio-kit plugin install` command.
Only the selected package and its dependencies are fetched.

| Component | Codex | Claude Code | OpenCode | Cursor / Antigravity / VS Code |
| --- | --- | --- | --- | --- |
| Skills and stdio MCPs | Yes | Yes | Yes | Yes |
| Scientific planner/reviewer | Read-only agent export and explicit invocation | Native agent | Read-only subagent | Explicitly omit |
| Hooks | `hooks/codex.json` | `hooks/hooks.json` | `hooks/opencode.js` | Explicitly omit |
| Claude slash commands | Unsupported | Native command | Unsupported | Unsupported |

The installer refuses unsupported components before changing the project.
`--components-only` deliberately installs only skills and MCPs. This is project
installation, not a claim that native plugin manifests are interchangeable.
The agent adapter accepts only the Read/Glob/Grep subset: OpenCode receives
explicit permissions; Codex receives a read-only sandbox and disables the
package's MCPs for that role. Other inherited client tools remain subject to the
client's policy. Claude model/effort overrides are not copied to other hosts.

Codex receives named agent definitions with complete, disabled MCP transport
entries. Native reviewer delegation has been exercised in Codex CLI. If your
client cannot discover the exported agent, invoke its installed instructions explicitly:

```bash
clio-kit plugin run-agent scientific-evidence-reviewer --client codex --project /path/to/project --prompt "Review these results: ..."
```

This starts a separate Codex session with the installed instructions, read-only
sandbox and package MCPs disabled. It does not silently fall back to a general
agent. Reading files still requires a working Codex sandbox on the machine.
OpenCode's free model endpoint may reject custom subagents; use a provider/model
that supports them and check the actual task result.

For a complete supported workflow:

```bash
clio-kit plugin install clio-dataset-report --client codex --project /path/to/project
```

Replace `codex` with `claude-code` or `opencode`. In Codex, trust the project and
review new or changed hooks through `/hooks`; installation does not grant hook
trust. OpenCode must run with plugins enabled (not `--pure`). Older clients may
need an upgrade. See the [authoring contract](authoring.md#client-specific-hooks).


### Codex CLI and IDE extension

```bash
clio-kit plugin install clio-scientific-io --client codex --project /path/to/project
cd /path/to/project
codex
```

Open/trust the project and start a fresh session. Inspect MCP status in the
client, and explicitly request a skill such as `$dataset-explore` for a first
check. The CLI and IDE extension share Codex configuration conventions.

The installer allows up to five minutes for a server's first locked dependency
build. For unattended `codex exec` runs, mark the server needed by the task as
required so the run fails if it cannot initialize. For a trusted server in a
disposable test project, authorize only the intended tools:

```bash
codex exec -c 'mcp_servers.clio-hdf5.required=true' \
  -c 'mcp_servers.clio-hdf5.enabled_tools=["open_file","list_keys","get_shape","close_file"]' \
  -c 'mcp_servers.clio-hdf5.default_tools_approval_mode="approve"' \
  'Inspect the names and shapes in sample.h5 using MCP, then close the file.'
```

These overrides apply only to this invocation. Use the actual configured server
name. Interactive sessions retain the client's normal approval policy. See the
[Codex MCP configuration](https://developers.openai.com/codex/config-reference).

For **one MCP and one skill** instead of a workflow:

```bash
codex mcp add clio-hdf5 -- clio-kit mcp-server hdf5
clio-kit skill install dataset-explore --target /path/to/project/.agents/skills
codex mcp list
```

The `codex mcp add` route writes the user's MCP configuration; Kit's project
installer writes project-local settings. Avoid registering the same server twice.
A skill may need additional tools; check its declared requirements.
Official references: [MCP configuration](https://developers.openai.com/codex/mcp),
[skills](https://developers.openai.com/codex/skills).

### Claude Code

For skills and MCP configuration in a project:

```bash
clio-kit plugin install clio-scientific-io --client claude-code --project /path/to/project
cd /path/to/project
claude
```

For the **full native plugin route**, including supported agents and hooks,
register the CLIO Kit checkout after installing the launcher:

```bash
claude plugin marketplace add /path/to/clio-kit
claude plugin install clio-scientific-io@clio-kit
claude mcp list
```

Restart/reload an existing session. Check that the servers connect, not merely
that the plugin is enabled. The dataset-report task plugin also includes a
reviewer dependency and verification hook; read its [workflow guide](plugins.md#scientific-dataset-report)
before using it. For a single server, use `claude mcp add --scope project
clio-compression -- clio-kit mcp-server compression` from the target project.
Official reference: [Claude Code plugins](https://code.claude.com/docs/en/plugins).

### OpenCode

```bash
clio-kit plugin install clio-scientific-io --client opencode --project /path/to/project
cd /path/to/project
opencode
```

Review `opencode.json`, then verify the local MCP connections in OpenCode. Its
MCP entries use `type: "local"` and a command array. The installer refuses an
existing `opencode.jsonc` rather than stripping comments; merge manually using
the catalogue's OpenCode configuration in that case.
For a skill alone, use `clio-kit skill install dataset-explore --target
/path/to/project/.opencode/skills`.
Official references: [MCP servers](https://opencode.ai/docs/mcp-servers/),
[skills](https://opencode.ai/docs/skills/).

### Cursor

```bash
clio-kit plugin install clio-scientific-io --client cursor --project /path/to/project
```

Open that folder in Cursor, review its MCP settings and approve/trust the intended
servers. Reload the window/session and check connection status before asking
Agent to use a skill. Project MCP configuration uses `.cursor/mcp.json` with a
`mcpServers` object. For a skill alone, install to `.cursor/skills`.
Kit's project route does not install a native Cursor plugin or translate Claude
hooks. Official references: [MCP](https://cursor.com/docs/mcp),
[skills](https://cursor.com/docs/skills).

### Antigravity

```bash
clio-kit plugin install clio-scientific-io --client antigravity --project /path/to/project
```

Open the project in Antigravity and check MCP settings. The adapter writes
`.agents/mcp_config.json` and `.agents/skills`. If your client version uses a
managed/global MCP configuration, merge the generated server entries into its
MCP settings rather than assuming the workspace file was loaded.
For the CLI, initialize the intended project with `agy --new-project` and reopen
it with `agy --project <name-or-id>`. A default CLI project may not discover the
files in your current shell directory. Kit does not translate native Claude hooks
into Antigravity plugins.
Official references: [MCP](https://antigravity.google/docs/mcp),
[skills](https://antigravity.google/docs/skills).

### VS Code / GitHub Copilot

```bash
clio-kit plugin install clio-scientific-io --client vscode --project /path/to/project
```

Open that project, trust it where appropriate, and use **MCP: List Servers** from
the command palette to inspect/start its servers. Confirm that the tools are
available in the agent chat. `.vscode/mcp.json` uses a `servers` object and
`type: "stdio"`; skills go to `.github/skills`.
The Codex VS Code extension is a different client: use the Codex route above.
Official reference: [VS Code MCP](https://code.visualstudio.com/docs/agent-customization/mcp-servers).

### Claude Desktop

Install the launcher on the same computer as Desktop. Open its developer MCP
configuration and merge an entry such as:

```json
{
  "mcpServers": {
    "clio-compression": {
      "command": "clio-kit",
      "args": ["mcp-server", "compression"]
    }
  }
}
```

Restart Desktop and check its MCP connection/tool list. If the GUI cannot resolve
`clio-kit`, use the executable's absolute path. This local MCP configuration does
not install skills or Claude Code plugins into Desktop. Do not copy Claude Code
plugin commands into Desktop's chat and assume they installed anything.
For client-managed paths, see the [local MCP setup guide](https://modelcontextprotocol.io/docs/develop/connect-local-servers).

### Clio Coder

Kit includes a portable collection of adapted Clio Coder skills. Its native
library route is separate from the six-client project installer:

```bash
clio-coder library install /path/to/clio-kit/skills/clio-coder-skills --project
clio-coder library pin clio-coder-skills --project
```

Run these from the intended project, then start a fresh Clio Coder session. Read
[Clio Coder integration](marketplace.md#clio-coder-integration) for provenance,
names, compatibility and complete native plugin packages. Configure local MCPs
through Clio Coder's own MCP settings; Kit has no `--client clio-coder` adapter.
See [Clio Coder's guides](https://coder.iowarp.ai/docs.html).

### Another MCP or Agent Skills client

Use `clio-kit mcp-server <name>` as the stdio process command. For a skill, run
`clio-kit skill install <skill-name> --target <client-skill-directory>`. Use the
host's documented configuration schema and verify discovery. There is no general
conversion of agent definitions, commands or hooks between hosts.

## 3. Check actual behavior

```bash
clio-kit doctor --server compression --connect
```

Then follow [HDF5 exploration in Codex](tutorials/codex-dataset.md) or
[CSV analysis in Claude Code](tutorials/claude-analysis.md). Inspect the skill
invocation, MCP calls and saved outputs, and compare with the supplied inputs.

Kit's six project adapters were exercised in fresh directories for this guide.
That confirms generated files, not live behavior in every listed GUI. The [tutorials](/tutorials) also exercise live Codex and Claude Code sessions,
including individual tools, portable skills, native plugins and a contributed package.

## Updates, removal and troubleshooting

- **Update a checkout:** pull the reviewed changes, update its editable launcher,
  and rerun the selected skill/project installation. Review conflicts before
  `--replace`. [Release installations](installation.md) have their own artifact path.
- **Remove a native Claude plugin:** `claude plugin uninstall <name>@clio-kit`.
  Shared dependencies may remain; inspect the client's installed components.
- **Update an indexed package:** repeat `plugin install` with `--update --replace`
  after reviewing the publisher. Otherwise the project reuses its recorded revision
  and verifies its content. Existing payloads also support offline reinstall.
- **Remove a project installation:** `clio-kit plugin uninstall <name> --client
  <client> --project /path/to/project`. Add `--dry-run` to preview. Receipts preserve
  shared components and unrelated settings; edited components stop removal.
  Publisher payloads and locks remain for other clients and offline reuse.
  Installations made before receipts were introduced still require manual removal.
- **Command not found:** the agent process needs the launcher on its PATH. Remote
  workspaces/containers need the launcher where the subprocess actually runs.
- **Server enabled but disconnected:** inspect startup logs and `doctor --connect`;
  first builds need network access, and scientific backends have system prerequisites.
- **Skill installed but not used:** reopen the intended project/session, verify the
  discovery path, explicitly invoke it, and inspect whether the file was read.
- **Unsupported native components:** choose the host's native route, or deliberately
  use `--components-only` and perform the missing review/hook steps yourself.
