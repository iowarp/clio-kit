---
name: worktree-merge
description: 'Use when this workflow is requested: Integrates finished worktree branches
  through one throwaway integration branch, testing after each merge and running the
  full suite before the main line moves, with exact rollback on failure. Not for creating
  worktrees; use worktree-create. Not for closeout; use branch-closeout. Triggers
  on "merge my worktrees", "integrate these worktree branches".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/git/worktree-merge
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Worktree Merge

Integrate finished branches through a disposable integration branch. The target branch moves only after every branch integrates cleanly and the project's full validation suite passes; nothing reaches the main line broken.

See [merge strategies](references/merge-strategies.md) for strategy mechanics (`merge`, `squash`, `ff`), rollback details, and protected base guards.

## Arguments

Arguments are passed in the user invocation message. Interpret them structurally from the prompt:

```text
/skill:worktree-merge [--into branch] [--strategy merge|squash|ff] [--cleanup ask|keep] <branch...>
```

### Examples
- `/skill:worktree-merge feat-a feat-b`
- `/skill:worktree-merge --into main --strategy squash feat/payments feat/invoicing`
- `/skill:worktree-merge --strategy ff --cleanup keep feat/docs-update`

### Positional Arguments
- `<branch...>`: Two or more branch names to integrate in order (or at least one when integrating into a distinct target branch).
  - Required. If fewer than the necessary branches are specified, prompt the user for the branch list; never guess.
  - Validation: Verify each ref exists via `git rev-parse --verify <branch>`.

### Options
- `--into <branch>`: Target base branch receiving the merges.
  - Default: the current branch at invocation time (`<original>`).
  - Validation: Must be a verified local branch.
- `--strategy <merge|squash|ff>`: Git integration strategy.
  - `merge` (default): Uses `git merge --no-ff <branch>` to preserve commit topology.
  - `squash`: Uses `git merge --squash <branch>`, followed by an atomic conventional commit.
  - `ff`: Uses `git merge --ff-only <branch>`. Fails if the branch has diverged.
- `--cleanup <ask|keep>`: Post-integration scaffolding cleanup policy.
  - `ask` (default): Prompt the user via `ask_user` before removing worktrees or deleting branches.
  - `keep`: Retain all source worktrees and branches after integration.

### Unknown Arguments and Ref Validation
- Unknown flags must be rejected with an error. Validate all branch names before creating integration branches. Never interpolate unvalidated user input into shell strings without safe quoting.

## Step 1 — Preconditions and Base Guards

1. Verify working directory is the repository root (fail if currently inside a worktree directory).
2. Record current branch as `<into>` (or use `--into <branch>`).
3. Verify no uncommitted changes exist on `<into>`. If dirty, STOP: commit or stash first.
4. **Base Guard**: Check if `<into>` is a protected or canonical default branch. In canonical-main-only repositories (such as `CONTRIBUTING.md`), maintainer integration is strictly local; never push integration or topic branches to the canonical remote.
5. Verify every branch in `<branch...>` with `git rev-parse --verify <branch>`. Any missing branch aborts the run before anything is created.

## Step 2 — Detect Validation Commands Once

Detect the project's test and validation suite per repository manifests and CI workflows (`.github/workflows/*`, Makefile, npm scripts):
- **Fast test command**: run immediately after each branch merge to localize failures.
- **Full validation suite**: run after all branches have merged (unit tests, typecheck, lint).

## Step 3 — Stand Up Disposable Integration Branch

Create a disposable integration branch off `<into>`:
```bash
git checkout -b integration-<first-branch> <into>
```
All integration and intermediate test runs occur exclusively on this branch.

## Step 4 — Merge Each Branch in Order

For each branch in `<branch...>`:
1. Apply the specified `--strategy`:
   - **`merge`**: `git merge --no-ff <branch>`
   - **`squash`**: `git merge --squash <branch> && git commit -m "feat: squash <branch>"`
   - **`ff`**: `git merge --ff-only <branch>`
2. **Conflict**: STOP immediately. Name conflicting branch and files, report manual resolution steps, and do not attempt automatic resolution.
3. **Run fast test command**:
   - If tests fail: STOP immediately. Localize the failure to this branch and execute exact rollback:
     ```bash
     git checkout <into> && git branch -D integration-<first-branch>
     ```
   - Report the breaking branch, test failures, and that `<into>` remains untouched.

## Step 5 — Full Gate, Land, and Rollback Safety

1. Once all branches are merged, run the full validation suite (tests, typecheck, lint).
2. If any check fails: report failures and execute rollback as in Step 4.
3. If all checks pass:
   - Land integration branch into `<into>`:
     ```bash
     git checkout <into>
     git merge --no-ff integration-<first-branch>
     ```
   - Delete the disposable integration branch: `git branch -d integration-<first-branch>`.

## Step 6 — Cleanup Scaffolding

If `--cleanup ask` is active, prompt the user via `ask_user` whether to clean up the merged branches and worktrees:
- On approval, delegate to or execute `branch-closeout`:
  - Locate registered worktree paths via `git worktree list --porcelain`.
  - Inspect tracked and untracked worktree state.
  - Remove worktrees with `git worktree remove` (never `rm -rf`).
  - Delete local source branches with `git branch -d`.
  - Audit and report surviving worktrees, local branches, and remote heads.
- If `--cleanup keep` is active, leave worktrees and branches intact.

## Step 7 — Report

Deliver a comprehensive completion report:
- Strategy used (`merge`, `squash`, or `ff`)
- Per-branch fast test outcomes
- Full gate results
- Merge commit SHA on `<into>`
- Cleanup status and inventory of remaining local refs

## Red Flags

- Merging directly into `<into>` without a disposable integration branch.
- Auto-resolving conflicts without user intervention.
- Skipping the fast test between merges, losing failure localization.
- Deleting worktrees or branches when `--cleanup keep` was passed or without confirmation.
- Pushing integration or topic branches to a canonical-main-only remote.
