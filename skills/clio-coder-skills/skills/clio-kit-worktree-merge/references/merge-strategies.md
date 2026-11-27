# Merge Strategies and Integration Architecture

This reference outlines integration strategies, throwaway branch lifecycles, and protected base guards for `worktree-merge`.

## 1. Supported Integration Strategies

| Strategy | Flag / Command | When to Use | Ancestry Behavior |
|---|---|---|---|
| `merge` (default) | `git merge --no-ff <branch>` | Multi-commit features where history and branch topology should be explicitly preserved. | Direct ancestor; `git merge-base --is-ancestor` passes. |
| `squash` | `git merge --squash <branch>` | Features with exploratory commits, fixups, or noisy history that should land as one atomic commit. | Not a direct ancestor; commit SHA and patch change. Closeout requires PR/commit evidence. |
| `ff` | `git merge --ff-only <branch>` | Clean, linear history where the feature branch was rebased onto `<into>` and no merge commit is desired. | Direct ancestor; `<into>` moves pointer to branch tip. Fails if diverged. |

## 2. Disposable Integration Branch Lifecycle

Never merge directly into the target base branch (`<into>`, e.g. `main` or `release-candidate`) while testing individual branches. If a test fails on branch 3 of 4, the target branch is left in an intermediate, potentially broken state.

### Execution Pattern
1. Create disposable branch off `<into>`:
   ```bash
   git checkout -b integration-<first-branch> <into>
   ```
2. Integrate each branch sequentially using the specified strategy:
   - **`merge`**: `git merge --no-ff <branch>`
   - **`squash`**: `git merge --squash <branch> && git commit -m "feat: squash <branch>"`
   - **`ff`**: `git merge --ff-only <branch>`
3. Run fast tests immediately after each branch is merged. If tests fail:
   - Abort integration and roll back:
     ```bash
     git checkout <into>
     git branch -D integration-<first-branch>
     ```
   - Report the exact branch that broke the build.
4. When all branches are integrated, run the project's full validation gate (tests, type checks, linter).
5. Only after all checks pass, land the integration branch into `<into>`:
   - If `merge` strategy was used: `git checkout <into> && git merge --no-ff integration-<first-branch>`
   - If `squash` strategy was used: `git checkout <into> && git merge --ff-only integration-<first-branch>`
   - If `ff` strategy was used: `git checkout <into> && git merge --ff-only integration-<first-branch>`
6. Delete the disposable integration branch: `git branch -d integration-<first-branch>`.

## 3. Protected Base Guards

- If `<into>` is the canonical default branch (`main`/`master`) or a release branch:
  - Check repository instructions (`CONTRIBUTING.md`). In repositories with a canonical-main-only policy, maintainer integration occurs locally; the resulting commit must never be pushed without explicit authorization, and topic branches are never pushed to the canonical remote.
  - Verify that `<into>` matches the intended target before landing.
  - If `<into>` has moved upstream during the merge, fetch and re-verify before finalizing.

## 4. Integration Proof Before Closeout

Before offering to delete source branches or remove worktrees:
- For `merge` and `ff`: Verify `git merge-base --is-ancestor <branch> <into>`.
- For `squash`: Verify the squashed commit containing the changes is present on `<into>` (`git log --grep="<branch>" <into>` or diff comparison).
- Once proved, delegate cleanup to `branch-closeout` or prompt the user.
