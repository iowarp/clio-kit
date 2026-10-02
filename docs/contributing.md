---
title: Contribute to CLIO Kit
---

# Choose a contribution route

Contribute a reusable procedure, a tool, or a complete workflow. Start with the
[terminal plugin tutorial](tutorials/contribute-plugin.md) if this is your first
package. The detailed contracts are in [authoring](authoring.md) and the repository's
[contributor guide](https://github.com/iowarp/clio-kit/blob/main/CONTRIBUTING.md).

## What are you adding?

| Contribution | Repository location | Include |
| --- | --- | --- |
| Skill package | `skills/<package>/` | Plugin manifest; `skills/<name>/SKILL.md`; resources and evaluation scenarios |
| Agent package | `agents/<package>/` | Plugin manifest; `agents/<name>.md` with the host's frontmatter |
| Hook package | `hooks/<package>/` | Plugin manifest; hook configuration and handler files |
| Workflow plugin | `plugins/<name>/` | Manifest and the components/dependencies needed for one job |
| Maintained MCP implementation | `mcp-servers/<name>/` | Runtime descriptor, dependency lock, source, tests and registry metadata |
| External plugin or marketplace | `community/entries/<name>.toml` | Publisher/source information pointing to the external repository or package |

A skill, MCP, agent or hook is a component. A plugin combines components for a
workflow; it need not contain every type. Native packaging wrappers do not turn
individual servers into elevated workflow plugins.

## Add a component or plugin here

1. Scaffold a package with `clio-kit plugin init plugins/my-plugin`. Add
   `--agent` or `--hook` only when you need those components. For a package of
   one component type, use its corresponding parent directory from the table.
2. Give it a meaningful description, implement its content, and define a bounded
   usage scenario with an expected result. Keep credentials, local sessions and
   generated evaluation output out of the package.
3. Run `clio-kit plugin validate <directory>`. For Claude packages also run
   `claude plugin validate <directory> --strict`.
4. Install and exercise it in a temporary project. Inspect actual tool calls,
   outputs and hook effects. A manifest check alone does not prove behavior.
5. Open a repository PR with the package and a concise account of the checks.

Valid folders are discovered automatically. Website start/build and CI update
catalogues; contributors do not manually edit generated JSON or run generator
commands. Until a build/sync has happened, a client's native marketplace snapshot
may still show the previous entries. The optional `clio-kit marketplace sync
--root .` refreshes it for local native testing.

See [package layouts](authoring.md#add-a-component-folder-to-clio-kit),
[skill rules](authoring.md#add-a-skill), and [workflow composition](plugins.md).
Preserve imported skill provenance; adaptations use the repository's importer.

## Add a maintained MCP in Python, Node or Go

Follow [the MCP authoring contract](authoring.md#add-an-mcp-server). Register the
server in `mcp-server-versions.toml` and keep its dependencies isolated from the
launcher. Provide the runtime descriptor and the appropriate lock/build/start
commands; a source folder alone is not a launchable server.

Exercise real stdio negotiation, tool discovery, and at least one representative
operation on known input. Run the server's own suite in its locked environment.
Document required native packages/services and which checks need them. Supporting
a runtime is separate from having every backend installed on a contributor's machine.

## Keep your code in your own repository

Follow the [community contribution tutorial](tutorials/contribute-plugin.md#8-prepare-a-community-entry)
to prepare an entry, open a PR and install an accepted contribution.

Create and validate the plugin there, then render the entry:

```bash
clio-kit plugin submit /path/to/my-plugin --repo your-org/your-repo
```

Review the generated TOML and put it in `community/entries/` in a PR. A whole
marketplace uses `--kind marketplace`; it remains controlled by its publisher.
`--open-pr` is an optional public submission action, requiring GitHub access.

Accepted source types include GitHub repositories, Git subdirectories, npm
packages and supported URL sources. Use the exact field schemas and pinning rules
in the [community guide](https://github.com/iowarp/clio-kit/blob/main/community/README.md).
Do not infer an external package's capabilities from its name. Indexing an entry
is not code certification or a promise about future upstream releases.

## Agents and hooks need host-specific checks

Kit's maintained native agent/hook packages target Claude Code. Other clients may
have their own formats; portable skill installation does not translate those
formats. Review hook commands without executing them first, then test in an
isolated supported client and verify what changed. See [hook authoring](authoring.md#add-a-hook).

## What to put in the PR

State the problem, intended users, installation route, required backends, and the
smallest repeatable usage example. Report the client/version, inputs, expected and
observed outputs, tests run, and anything skipped. Use the
[repository verification commands](https://github.com/iowarp/clio-kit/blob/main/AGENTS.md)
for your change type. Do not describe a successful connection as a successful
scientific workflow.
