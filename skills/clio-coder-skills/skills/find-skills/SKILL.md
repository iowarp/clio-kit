---
name: find-skills
description: 'Use when this workflow is requested: Finds published skills and prepares
  their Library installation for the operator when the user asks whether a capability
  exists as a skill. Not for authoring a new skill; use skill-craft. Triggers on "find
  a skill", "is there a skill for this".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/meta/find-skills
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Find Skills

Find reusable skills through Clio-Coder's Library. Keep the package's origin,
format, installation scope and actual recipe availability distinct. Library
management is operator authority in Clio-Coder; the agent discovers, inspects
and recommends without trying to write protected active roots.

## Procedure

1. **Search the Library first.** Run `clio-coder library search <query>` and
   inspect `clio-coder library recipes --kind skill --json` for actual recipes.
   In a Clio session, `context(scope="library", kind="skill", query="...")`
   provides the same discovery facts. A catalog hint describes a package's
   contents; it does not prove a skill is installed or usable. Check copy scope,
   trust, enablement, integrity and resource diagnostics before reporting that
   the capability is available. An installed but disabled package needs a
   different action from a missing package.

2. **Research external candidates when needed.** Browse https://skills.sh and
   GitHub read-only. Record source URL, description and evidence of quality.
   Distinguish a complete portable `plugin.json` package, a supported Claude
   Code or Codex plugin, and a bare `SKILL.md` directory. A GitHub listing alone
   does not prove its payload is directly installable. Do not execute a
   candidate's setup instructions while researching it.

3. **Prepare the exact operator action.** Use a typed reference such as
   `clio-coder library install skill:context-handoff --project` for a catalog
   package. Use `library install <package-path-or-github-tree-url>` for a
   portable package, and `library import <path-or-github-tree-url> --dry-run`
   to review a supported foreign plugin's projection and omitted features.
   A bare skill needs a portable package envelope; use the authoring template
   under `library/_authoring/templates/skill/` or the package authoring guide.
   Do not promise that every skill URL works as an install source.

4. **Use the operator's Library controls.** Point to `/library` or Alt+L for
   review and installation. Show the requested scope explicitly: `--project`
   uses `<cwd>/.clio-coder/plugins/<package-name>/`; `--user` uses
   `<configDir>/plugins/<package-name>/` and is the CLI default. Every kind
   uses that complete-package store. Loose `.clio-coder/skills/` files are a
   separate operator-managed discovery surface. Honor authorization already
   provided, while explaining that the Clio operator must apply the mutation
   through the UI or CLI; do not retry blocked install commands as shell tricks.

5. **Verify after the operator applies it.** Read
   `clio-coder library recipes --kind skill --json` and
   `clio-coder library inspect <kind>:<package-name> --json`. Confirm the actual
   runtime name, whole-package owner, selected scope and recipe availability.
   Report disabled, foreign, shadowed or invalid states plainly; an installed
   directory alone is not success. Use `/skill <runtime-name>` for activation;
   `/skills` opens the category browser. Do not invent an audit status.

## Host boundaries

For a Clio-Coder installation, keep management in its Library. External
installers such as `npx skills add` target another host's roots and do not
update Clio's package state. If the user explicitly wants a Claude Code or
Codex installation instead, follow that host's documented marketplace or
skill-installation route and name the destination clearly. The Clio-Coder
library is portable; installing it for one host does not install it for all.
