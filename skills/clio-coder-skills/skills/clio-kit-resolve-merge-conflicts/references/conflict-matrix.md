# Conflict Matrix and Resolution Reference

This reference documents operation detection, ours/theirs orientations, conflict types, and operation-specific resolution workflows for `resolve-merge-conflicts`.

## 1. Operation State Detection

Determine which git operation is currently paused on conflicts:

| Indicator | Operation | Current (`HEAD` / "ours") | Incoming ("theirs") | Abort Command | Continue Command |
|---|---|---|---|---|---|
| `.git/MERGE_HEAD` exists | **Merge** | Target branch receiving merge | Source branch being merged in | `git merge --abort` | `git merge --continue` (or `git commit`) |
| `.git/rebase-merge` or `.git/rebase-apply` exists | **Rebase** | **Upstream branch** being rebased onto | **Your topic commit** being replayed | `git rebase --abort` | `git rebase --continue` |
| `.git/CHERRY_PICK_HEAD` exists | **Cherry-pick** | Target branch receiving commit | Commit being cherry-picked | `git cherry-pick --abort` | `git cherry-pick --continue` |
| `.git/REVERT_HEAD` exists | **Revert** | Current branch | Commit being reverted | `git revert --abort` | `git revert --continue` |

### The Rebase "Ours / Theirs" Reversal Trap

> [!WARNING]
> In `git rebase`, `--ours` and `--theirs` are **reversed** compared to `git merge`!
> - `--ours` refers to the upstream target branch you are rebasing onto.
> - `--theirs` refers to the original topic branch commit that is being replayed.
> Failing to account for this inversion causes accidental reversion of your own changes.

To inspect intent without relying on relative terms:
```bash
# During merge:
git log --oneline HEAD...MERGE_HEAD

# During rebase:
# Check the commit currently being applied:
git log -1 REBASE_HEAD 2>/dev/null || cat .git/rebase-merge/stopped-sha
```

## 2. Conflict Types and Handling

### A. Content Conflicts (Conflict Markers)
Files contain `<<<<<<<`, `=======`, `>>>>>>>` markers.
1. Read commit history for both sides to understand why each line changed.
2. If both changes are compatible (e.g. adjacent methods, distinct properties), preserve both.
3. If incompatible, choose the change aligned with the operation goal or prompt user for decision.
4. Verify markers are gone: `grep -rn "<<<<<<<\|>>>>>>>" <files>` must return zero matches.

### B. Add / Add Conflicts
Both branches added a file at the same path with different contents.
1. Inspect both versions:
   ```bash
   git show :2:path/to/file  # ours
   git show :3:path/to/file  # theirs
   ```
2. Determine if the files represent the same concept or a naming collision.
3. If distinct concepts, rename one file and adjust imports. If same concept, merge contents and stage with `git add path/to/file`.

### C. Modify / Delete Conflicts
One branch modified a file while the other deleted it.
- Git status reports: `CONFLICT (modify/delete): file deleted in <side> and modified in <other>`.
- To keep the modified file:
  ```bash
  git add path/to/file
  ```
- To delete the file:
  ```bash
  git rm path/to/file
  ```
- Always inspect the commit that deleted the file (`git log -n 5 -- <file>`) to confirm whether the deletion was intentional refactoring or an accidental deletion.

### D. Rename Conflicts
Git detects: `CONFLICT (rename/rename)`, `CONFLICT (rename/delete)`, or `CONFLICT (rename/modify)`.
1. Inspect rename targets using `git status` and `git log`.
2. Consolidate modifications into the intended canonical filename.
3. Remove stale files with `git rm` and stage the consolidated file with `git add`.

### E. Binary File Conflicts
Binary files (images, compiled archives, model checkpoints) cannot contain text conflict markers.
1. Determine which binary is required:
   ```bash
   # Select current branch version
   git checkout --ours -- path/to/binary
   # Or select incoming version
   git checkout --theirs -- path/to/binary
   ```
2. Or regenerate the binary from source assets if applicable.
3. Stage with `git add path/to/binary`.

### F. Submodule Conflicts
Submodule conflict occurs when both sides updated the submodule pointer to different commit SHAs.
1. Inspect the two submodule pointers:
   ```bash
   git diff --submodule
   ```
2. Enter submodule directory and check which commit is the ancestor or latest intended state:
   ```bash
   cd path/to/submodule && git merge-base --is-ancestor <sha1> <sha2>
   ```
3. Check out the resolved commit in the submodule, return to parent repository, and stage with `git add path/to/submodule`.

### G. Generated Files and Lockfiles
Never attempt manual string edits on complex lockfiles (`package-lock.json`, `pnpm-lock.yaml`, `yarn.lock`, `Cargo.lock`, `uv.lock`, `poetry.lock`).
1. Check out the base version or let package manager re-derive resolutions:
   - Node (npm): `npm install --package-lock-only`
   - Node (pnpm): `pnpm install --lockfile-only`
   - Rust: `cargo check` (updates Cargo.lock)
   - Python (uv): `uv lock`
2. Stage the regenerated lockfile with `git add`.

## 3. Multi-Stop Rebase Workflow

When rebasing a series of commits, conflicts may occur at multiple commits:
1. **Loop**:
   - Inspect stopped commit and conflicting files.
   - Reconstruct intent and resolve each conflict.
   - Run validation checks (or fast tests).
   - Stage all resolved files: `git add <files>`.
   - Run `git rebase --continue`.
2. **Next State**:
   - If another conflict arises, repeat the loop.
   - If the commit becomes empty (because changes were already incorporated upstream), use `git rebase --skip` only after verifying changes exist upstream.
3. **Completion**:
   - Once all commits are replayed, run the full validation suite.

## 4. Operation Abort Handling

If conflicts cannot be resolved, or if the user requests aborting:
- **Merge**: `git merge --abort`
- **Rebase**: `git rebase --abort`
- **Cherry-pick**: `git cherry-pick --abort`
- Verify working tree is clean with `git status`.
