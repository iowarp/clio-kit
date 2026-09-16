# Branch and Worktree Closeout Checklist

This reference provides the verification steps, safety invariants, and reporting requirements for `branch-closeout`.

## 1. Landing Evidence Verification

Never delete a branch or remove a worktree based on matching commit subjects alone. Require deterministic proof of integration on the canonical base branch:

### Merge Commits and Fast-Forwards
```bash
git fetch --prune <canonical>
git merge-base --is-ancestor <branch> <canonical>/<base>
```
If the command exits 0, the branch tip is fully contained in the base branch history.

### Squash Merges and Cherry-Picks
A squash merge generates a new commit with a different tree SHA and patch identity; the ancestry test will fail. Prove integration via PR evidence:
```bash
gh pr view <branch> --repo <canonical> --json state,mergedAt,mergeCommit -q '{state: .state, merged: .mergedAt, commit: .mergeCommit.oid}'
```
Verify that the `mergeCommit.oid` exists in `<canonical>/<base>`:
```bash
git branch -r --contains <mergeCommit.oid> | grep -E "(^|\s)<canonical>/<base>$"
```

## 2. Worktree State Inspection

Before unregistering a worktree, ensure no valuable untracked or ignored artifacts are discarded:
1. Locate the registered worktree path:
   ```bash
   git worktree list --porcelain
   ```
2. Check for uncommitted changes:
   ```bash
   git -C <worktree-path> status --porcelain
   ```
3. Check for untracked and ignored files:
   ```bash
   git -C <worktree-path> status --ignored --porcelain
   ```
4. If non-rebuildable files (e.g. unpushed experiment logs, scratch scripts, local databases) are present, halt and request user confirmation via `ask_user` before proceeding.

## 3. Worktree Removal

- Always remove registered worktrees using Git:
  ```bash
  git worktree remove <worktree-path>
  ```
- **Never use raw filesystem deletion (`rm -rf`)** on registered worktrees, which leaves corrupted records in `.git/worktrees/`.
- Use `--force` only when the user has explicitly authorized discarding uncommitted changes or untracked artifacts.

## 4. Local Branch Deletion

- Delete the local branch safely:
  ```bash
  git branch -d <branch>
  ```
- If git warns that the branch is not fully merged (common after squash merges), do not blindly force-delete. Verify the squash commit on the base branch, inform the user, and use `git branch -D <branch>` only upon explicit confirmation.

## 5. Contributor Fork Branch Cleanup

- If `--delete-fork-branch` was requested:
  - Verify that `<fork>` is indeed the contributor's fork remote and NOT the canonical repository.
  - Delete the remote topic branch:
    ```bash
    git push --delete <fork> refs/heads/<branch>
    ```
  - **Invariants**:
    - Never delete branches from the canonical remote.
    - Never delete canonical base branches (`main`, `master`).
    - Never delete tags or stashes implicitly.

## 6. Post-Closeout Inventory Report

Always audit and report surviving repository state:
1. **Registered worktrees**: `git worktree list`
2. **Local branches**: `git branch`
3. **Stashes**: `git stash list`
4. **Local-only tags**: tags that do not exist on the canonical remote
5. **Canonical remote heads**: `git ls-remote --heads <canonical>`
   - In a canonical-main-only repository, the expected canonical head set is strictly `refs/heads/main`. Any extra head is flagged as an anomaly.
