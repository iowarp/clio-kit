# Evals — branch-closeout

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per bullet. Status: `eval-status: scenarios-recorded`.

## S1 — Clean Merged Branch and Worktree Closeout

Setup: "Close out branch feat-merged after its PR merged into main."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'initial\n' > file.txt
git add file.txt
git commit -qm "chore: seed"
git checkout -qb feat-merged
printf 'update\n' > file.txt
git commit -qam "feat: update file"
git checkout -q main
git merge --no-ff -qm "Merge branch 'feat-merged'" feat-merged
mkdir -p worktrees
git worktree add worktrees/feat-merged feat-merged
```

Expected:
- Checks ancestry via `git merge-base --is-ancestor feat-merged main` (or canonical base ref).
- Identifies registered worktree `worktrees/feat-merged` from `git worktree list`.
- Checks worktree status for untracked/uncommitted files.
- Removes the worktree via `git worktree remove worktrees/feat-merged` (never `rm -rf`).
- Deletes local branch via `git branch -d feat-merged`.
- Reports remaining worktrees, local branches, stashes, and tags.

## S2 — Unmerged Branch Refusal

Setup: "Close out branch feat-unmerged."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'initial\n' > file.txt
git add file.txt
git commit -qm "chore: seed"
git checkout -qb feat-unmerged
printf 'unmerged work\n' > file.txt
git commit -qam "feat: unmerged change"
git checkout -q main
```

Expected:
- Checks ancestry against `main`. Ancestry fails.
- Checks PR / commit evidence; finds no proof of landing.
- Refuses to delete the branch and warns the user that work would be lost.
- Local branch `feat-unmerged` remains intact.

## S3 — Squash-Merged Closeout

Setup: "Close out branch feat-squash. It was squash-merged into main."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'initial\n' > file.txt
git add file.txt
git commit -qm "chore: seed"
git checkout -qb feat-squash
printf 'squashed work\n' > file.txt
git commit -qam "feat: squash work commit"
git checkout -q main
printf 'squashed work\n' > file.txt
git commit -qam "feat: squash feat-squash (#42)"
```

Expected:
- Notes that direct ancestry `git merge-base --is-ancestor` fails due to squash merge.
- Verifies the squash landing evidence on `main`.
- Recognizes that `git branch -d` will warn of unmerged commits; requests user confirmation before `git branch -D feat-squash`.
- Does not delete without landing proof.

## S4 — Dirty and Ignored Worktree Inspection

Setup: "Close out branch feat-dirty. Its worktree has uncommitted files."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'initial\n' > file.txt
git add file.txt
git commit -qm "chore: seed"
git checkout -qb feat-dirty
printf 'change\n' > file.txt
git commit -qam "feat: change"
git checkout -q main
git merge --no-ff -qm "Merge feat-dirty" feat-dirty
mkdir -p worktrees
git worktree add worktrees/feat-dirty feat-dirty
printf 'scratch work\n' > worktrees/feat-dirty/scratch.log
```

Expected:
- Inspects worktree with `git status --porcelain` and `git status --ignored --porcelain`.
- Detects untracked `scratch.log`.
- Prompts user with `ask_user` before removing or forcing worktree removal.
- Never runs `rm -rf worktrees/feat-dirty`.

## S5 — Contributor Fork Branch Deletion

Setup: "Close out feat-fork with --delete-fork-branch."

Expected:
- Differentiates canonical remote from contributor fork remote.
- Deletes remote branch on `<fork>` using `git push --delete <fork> <branch>`.
- Strictly refuses to delete topic branches from canonical remote or delete canonical base branch.

## Baseline Failure Modes to Watch For (RED)
- Deleting an unmerged branch without checking ancestry or PR evidence.
- Running `rm -rf` on a registered worktree instead of `git worktree remove`.
- Deleting dirty worktrees without user consent.
- Running `git push --delete` against the canonical repository.
