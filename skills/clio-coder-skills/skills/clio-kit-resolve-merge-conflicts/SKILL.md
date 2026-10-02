---
name: clio-kit-resolve-merge-conflicts
description: 'Use when this workflow is requested: Resolves a merge, rebase, or cherry-pick
  stopped on conflicts by reconstructing both sides'' intent, validating with the
  project''s own checks, and completing the operation. Not for integrating many branches;
  use worktree-merge. Triggers on "merge conflict", "fix these conflicts".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/git/resolve-merge-conflicts
  upstream-name: resolve-merge-conflicts
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Resolve Merge Conflicts

Resolve conflicts stopped during a merge, rebase, or cherry-pick from intent, not from picking whichever side makes markers disappear. The job ends with the operation completed and the project's validation checks green.

See [conflict matrix](references/conflict-matrix.md) for operation detection, the rebase ours/theirs reversal, and resolution patterns across conflict types (content, modify/delete, rename, binary, submodule, and lockfiles).

## Arguments

Arguments are passed in the user invocation message. Interpret them structurally from the prompt:

```text
/skill:clio-kit-resolve-merge-conflicts [--checks auto|command] [--continue|--no-continue]
```

### Examples
- `/skill:clio-kit-resolve-merge-conflicts`
- `/skill:clio-kit-resolve-merge-conflicts --checks "npm test"`
- `/skill:clio-kit-resolve-merge-conflicts --no-continue`

### Options
- `--checks <auto|command>`: Validation checks execution mode.
  - `auto` (default): Detects and runs test, typecheck, and lint commands from the repository (CI workflows, Makefile, package manifests).
  - `<command>`: Custom shell command to run for verification (e.g. `npm test`, `cargo check`).
- `--continue|--no-continue`: Whether to complete the in-flight operation after resolution and validation.
  - `--continue` (default): Runs the operation's continue command (`git merge --continue`, `git rebase --continue`, or `git cherry-pick --continue`).
  - `--no-continue`: Leaves resolved files staged in the index and stops without executing the continue command.

### Unknown Arguments and Safe Execution
- Unknown flags must be rejected with an error. Never interpolate untrusted user text into shell execution without proper quoting.

## Step 1 — Map the State and Operation

> [!IMPORTANT]
> Execute steps directly; do not declare task boards or write artifact files. Run discrete git commands sequentially; never nest commands inside `$(...)` command substitutions or subshells.

Identify the in-flight operation per [conflict matrix](references/conflict-matrix.md):
- **Merge**: `.git/MERGE_HEAD` exists. `HEAD` is ours (target branch), `MERGE_HEAD` is theirs (incoming branch).
- **Rebase**: `.git/rebase-merge` or `.git/rebase-apply` exists.
  > [!WARNING]
  > In `git rebase`, `--ours` and `--theirs` are **reversed**!
  > - `HEAD` ("ours") is the upstream branch being rebased onto.
  > - `REBASE_HEAD` ("theirs") is your branch commit being replayed.
- **Cherry-pick**: `.git/CHERRY_PICK_HEAD` exists. `HEAD` is target branch, `CHERRY_PICK_HEAD` is commit being cherry-picked.

Query all unmerged paths:
```bash
git diff --name-only --diff-filter=U
```

## Step 2 — Reconstruct Both Intents

For each conflicting file, determine why each side modified the code:
- Check commit logs:
  ```bash
  # In merge:
  git log --oneline HEAD..MERGE_HEAD -- <file>
  git log --oneline MERGE_HEAD..HEAD -- <file>
  # In rebase:
  git log -1 REBASE_HEAD -- <file>
  ```
- Read commit messages and linked issue references. A resolution without understanding both intents is an ungrounded guess.

## Step 3 — Resolve Each Conflict by Type

Handle each file according to its conflict class:
- **Content conflict markers**: Preserve both changes if compatible (e.g. non-overlapping functions or independent parameters). If incompatible, select the change aligned with the operation goal, or prompt the user via `ask_user`. Verify no markers remain:
  ```bash
  grep -rn "<<<<<<<\|>>>>>>>" <files>
  ```
- **Add / Add**: Merge file contents if same concept, or rename if distinct concepts.
- **Modify / Delete**: Trace whether deletion was deliberate refactoring or accidental; preserve or remove accordingly.
- **Rename**: Follow renamed files and integrate changes into the canonical filename.
- **Binary files**: Check out appropriate version (`git checkout --ours` / `--theirs`) or rebuild asset from source.
- **Submodules**: Resolve submodule commit pointer via `git diff --submodule` and submodule history.
- **Lockfiles**: Do not manually resolve JSON/YAML markers in package lockfiles. Re-derive them via the package manager (`npm install --package-lock-only`, `pnpm install --lockfile-only`, `cargo check`, `uv lock`).

Stage resolved files: `git add <files>`.

## Step 4 — Validate

Run the validation suite determined by `--checks`:
- If `auto`: execute detected typecheck, tests, and linter.
- If `<command>`: execute the specified validation command.
- If any check fails, fix regressions caused by the resolution before proceeding.

## Step 5 — Complete Operation or Multi-Stop Loop

If `--no-continue` was specified:
- Report that all conflicts are resolved and files are staged; stop here.

If `--continue` is active (default):
- Execute the operation-specific continue command:
  - **Merge**: `git merge --continue` (or `git commit`)
  - **Rebase**: `git rebase --continue`
  - **Cherry-pick**: `git cherry-pick --continue`
- **Multi-stop rebases**: If git stops on a subsequent commit conflict, loop back to Step 1 and repeat until the rebase completes.
- **Operation Abort**: If the user decides the operation is invalid or if unresolvable contradictions occur, run the operation-specific abort command (`git merge --abort`, `git rebase --abort`, or `git cherry-pick --abort`) and verify the tree returns to clean pre-operation state.

## Step 6 — Report

Summarize:
- Completed operation type (merge, rebase, cherry-pick)
- Each conflicting file, conflict type, and how intent was resolved
- Validation checks run and real outputs
- Current branch status and clean working tree verification

## Red Flags

- Taking `--ours` or `--theirs` wholesale without reading both sides.
- Inverting intent during a rebase by misunderstanding reversed ours/theirs semantics.
- Hand-editing conflict markers in binary files or complex lockfiles.
- Leaving `<<<<<<<` or `>>>>>>>` markers in any committed file.
- Declaring done without running validation checks.
- Executing an abort without user instruction.
