---
name: ship
description: 'Use when this workflow is requested: Ships finished work: writes one
  reviewed atomic commit, keeps maintainer branches local, or pushes a contributor
  branch to their fork and opens a PR only on explicit intent. Not for producing the
  change; use fix-issue. Triggers on "ship this", "commit this".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/git/ship
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Ship

Move finished, verified work out safely. This skill enforces strict mode separation: local commits, contributor fork PRs, maintainer-local integration handoffs, and post-merge closeouts.

See [remote and branch policy](references/remote-and-branch-policy.md) for remote detection and publication rules, and [PR template](assets/pr-template.md) for standard pull request layout.

## Arguments

Arguments are passed in the user invocation message. Interpret them structurally from the prompt:

```text
/skill:ship commit|pr|closeout [--base branch] [--fork remote] [--issue number] [paths...]
```

### Examples
- `/skill:ship commit src/cli/run.ts tests/run.test.ts`
- `/skill:ship pr --base main --fork fork --issue 105`
- `/skill:ship commit --issue 42`
- `/skill:ship closeout --base main`

### Positional Arguments
- `mode`: The operational mode. Must be one of:
  - `commit`: Review and stage changes, then create one atomic conventional commit. Never pushes; never opens a PR.
  - `pr`: Validate clean tree, push topic branch to contributor fork remote, and open a pull request against canonical base.
  - `closeout`: Clean up local worktrees and branches after a PR is verified merged; delegates to `branch-closeout`.
- `paths...`: Optional explicit file paths to stage for `commit`. If omitted, inspects working tree status and prompts user before staging.

### Options
- `--base <branch>`: Target base branch on the canonical remote (e.g. `main`, `master`).
  - Default: detected default branch from `gh repo view` or remote tracking.
  - Validation: Must be a valid git ref name (`git check-ref-format --branch <branch>`).
- `--fork <remote>`: Remote name pointing to the contributor's personal fork.
  - Default: detected fork remote from Git URLs.
  - Refusal Guard: Never assumed to be `origin`. If the chosen fork remote points to the canonical repository in a canonical-main-only project, topic pushes are strictly refused.
- `--issue <number>`: Issue number or identifier to reference in commit footer (`fixes #N`) and PR description.

### Unknown Arguments and Validation
- Reject unknown options with an error. Validate ref names and file paths before executing git commands. Never interpolate unvalidated user strings into raw shell commands.

## Step 1 — Project Conventions Win

Inspect project instructions (`CLIO-CODER.md`, `AGENTS.md`, `CONTRIBUTING.md`, `CLAUDE.md`) for commit formats, PR templates, and branching policy. Detected repository policy overrides default assumptions.

## Step 2 — Commit Mode (`commit`)

Review the entire working tree before staging:
1. Run `git status --porcelain` and `git diff HEAD`.
2. Exclude secret-shaped files (`.env*`, credentials, keys), generated build outputs, and unrelated files.
3. Stage explicit paths only (using `paths...` or prompted list). Never run `git add -A` blindly.
4. Compose one atomic conventional commit:
   `<type>(<scope>): <imperative summary>`
   Include `fixes #N` or `refs #N` footer when `--issue <number>` is provided.
5. If hooks or commit signing fail, report the exact error and halt. Use `--no-verify` only upon explicit user instruction.
6. If the selected mode is `commit`, report `git log -1 --stat` and finish here.

## Step 3 — Classify Remotes and Gate Publication

For `pr` mode, classify remotes per [remote and branch policy](references/remote-and-branch-policy.md):
1. Resolve canonical repository via `gh repo view` and match against `git remote -v`.
2. Resolve `--fork <remote>`.
3. Evaluate publication gates:
   - **On base branch directly**: STOP. Work must be on a topic branch.
   - **Uncommitted changes remain**: STOP. Stash or commit before pushing.
   - **Canonical remote is main-only**:
     - Maintainer clones keeping branches local: Report commit complete and ready for local integration (e.g. fast-forward or `worktree-merge`). Refuse to push topic branch to canonical remote.
     - Contributor clones: Require push to fork remote `<fork>`, never canonical.
   - **Open PR exists**: STOP and print the existing PR URL.
   - **Merged PR exists**: Report merged status; suggest `closeout` mode.

## Step 4 — Contributor Fork Push and PR (`pr`)

1. Push topic branch to the verified fork remote using an explicit refspec:
   ```bash
   git push -u <fork> refs/heads/<topic>:refs/heads/<topic>
   ```
2. Open the pull request against the canonical repository:
   ```bash
   gh pr create --repo <canonical-owner/repo> --base <base> --head <fork-owner>:<topic> --title "..." --body-file "$pr_body_file"
   ```
3. Use repository `.github/PULL_REQUEST_TEMPLATE.md` or [standard PR template](assets/pr-template.md). Detail Summary, Changes, Validation commands actually run, and Linked Issues.
4. Report PR URL, number, and review instructions.

## Step 5 — Closeout Mode (`closeout`)

When invoked as `ship closeout` (or when user requests cleanup of merged work):
1. Ensure the user explicitly requested closeout.
2. Delegate to `/skill:branch-closeout` or execute the closeout protocol:
   - Verify merge/squash landing evidence on canonical base branch.
   - Inspect worktree for uncommitted or untracked changes.
   - Remove worktree with `git worktree remove` (never `rm -rf`).
   - Safely delete local branch (`git branch -d`).
   - Audit remaining worktrees, local branches, stashes, and remote heads.

## Red Flags

- Blurring commit and PR phases: running `git push` or `gh pr create` when user only asked to commit.
- Assuming a remote named `origin` is a fork or canonical without checking its URL.
- Pushing a topic branch to a canonical-main-only repository.
- Staging files with `git add -A` without inspecting untracked files.
- Claiming validation checks passed when they were never run.
- Automatically deleting branches or worktrees without verifying merge evidence.
