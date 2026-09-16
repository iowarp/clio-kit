---
name: clio-kit-worktree-create
description: 'Use when this workflow is requested: Stands up one or more git worktrees
  for parallel work, each on its own branch with gitignored config copied in, dependencies
  installed, and a verified health check. Not for merging finished worktrees; use
  worktree-merge. Not for closeout; use branch-closeout. Triggers on "create a git
  worktree", "set up worktrees for these branches".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/git/worktree-create
  upstream-name: worktree-create
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Worktree Create

Stand up isolated git worktrees, each on its own branch, ready for immediate development and validation: gitignored configuration copied, dependencies installed, and health checks verified. Everything is detected from the repository; nothing about the tech stack is assumed.

See [worktree setup](references/worktree-setup.md) for package manager detection, configuration files, and health checks.

## Arguments

Arguments are passed in the user invocation message. Interpret them structurally from the prompt:

```text
/skill:clio-kit-worktree-create [--base ref] [--root path] [--setup auto|none] <branch...>
```

### Examples
- `/skill:clio-kit-worktree-create feat/user-profiles feat/billing-portal`
- `/skill:clio-kit-worktree-create --base main --root .worktrees feat/api-v2`
- `/skill:clio-kit-worktree-create --setup none chore/refactor-docs`

### Positional Arguments
- `<branch...>`: One or more branch names to create and check out.
  - Required. If omitted, prompt the user for the list of branches; never guess or invent branch names.
  - Validation: Each branch must be a valid ref name (`git check-ref-format --branch <branch>`).

### Options
- `--base <ref>`: Base git reference to branch from.
  - Default: detected canonical default branch (e.g. `main` or `master`) for clean branches, or `HEAD` if the user requested carrying current commits.
  - Validation: Must resolve via `git rev-parse --verify <ref>`.
- `--root <path>`: Directory path under which worktrees will be created.
  - Default: `worktrees/` relative to the repository root.
  - Path safety: Must remain within the repository boundary; reject paths containing directory traversal (`..`).
- `--setup <auto|none>`: Setup behavior for newly created worktrees.
  - `auto` (default): Detects and copies gitignored config, installs dependencies, runs codegen/build if needed, and executes health checks.
  - `none`: Checks out the worktree only without copying files or executing commands.

### Unknown Arguments and Path Validation
- Unknown flags must be rejected with an error.
- Ref names and filesystem paths must be validated before execution. Never interpolate untrusted user strings into shell commands without safe quoting.

## Step 1 — Validate Arguments and Ignore Root

> [!IMPORTANT]
> Execute steps directly; do not declare task boards or write artifact files. Run discrete git commands sequentially; never nest commands inside `$(...)` command substitutions or subshells.

1. Validate branch names and `--base <ref>`.
2. Ensure `--root <path>` is gitignored in the main repository. If not already ignored, add it to `.gitignore` and notify the user.
3. Derive safe filesystem paths under `--root` for each branch:
   - Handle slashes in branch names safely (e.g. `feat/auth` maps to safe directory `<root>/feat-auth` or `<root>/feat/auth` without escaping `<root>`).

## Step 2 — Detect Project Setup Once

If `--setup auto` is active, detect environment requirements from the repository per [worktree setup](references/worktree-setup.md):
- **Install command**: detected from manifests and lockfiles (`package-lock.json`, `pnpm-lock.yaml`, `Cargo.lock`, `uv.lock`, `poetry.lock`, etc.).
- **Gitignored config**: candidate files (`.env*`, local config) verified with `git check-ignore <file>`. Never copy tracked files. If `.worktreeinclude` exists, use it as the source of truth.
- **Health-check command**: prefer existing CI commands (`.github/workflows/*`), Makefile targets, or manifest scripts.
- **Base port**: assign unique ports (`base_port + index`) if health checks start network daemons.

## Step 3 — Stand Up Each Worktree Sequentially

For each `<branch>` in order:
1. Create the worktree:
   ```bash
   git worktree add <safe-worktree-path> -b <branch> <base>
   ```
2. If `--setup auto`:
   - Copy verified gitignored config files into `<safe-worktree-path>`.
   - Run detected package manager install command inside the worktree directory.
   - Run any necessary build/codegen steps.
   - Execute the detected health check. Stop any background servers started during verification.
3. Record status for the worktree: path, branch, dependencies installed, health status (PASS/FAIL), and any errors.

A failure in one worktree does not abort setup for subsequent branches; continue setup and report the failure.

## Step 4 — Report

Output a structured summary table:
- Branch name and filesystem path
- Dependency installation status
- Health check result (PASS/FAIL with error output if failed)
- Port assignment (if applicable)

Remind the user:
- Start development in the ready worktrees.
- When finished, merge using `worktree-merge` and close out using `branch-closeout`. Never remove worktrees via `rm -rf`; always use `git worktree remove`. In canonical-main-only repositories, maintainer worktree branches stay local.

## Red Flags

- Assuming package managers or test runners instead of detecting them from the repository.
- Copying files into the worktree without verifying `git check-ignore`.
- Duplicating tracked files across worktrees.
- Escaping the worktree root via unsanitized branch name path traversal.
- Reporting a worktree ready when its health check failed.
- Using `rm -rf` on a registered worktree.
