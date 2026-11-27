# Evals — worktree-merge

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per bullet. Status: `eval-status: smoke-checked`.

## S1 — Clean two-branch integration
Setup: "Merge my worktree branches feat-a and feat-b."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "scripts": {\n    "test": "node test.js"\n  }\n}\n' > package.json
printf 'console.log("tests pass");\n' > test.js
printf 'alpha\n' > a.txt
printf 'beta\n' > b.txt
git add package.json test.js a.txt b.txt
git commit -qm "chore: seed"
git checkout -qb feat-a
printf 'alpha 2\n' > a.txt
git commit -qam "feat: extend alpha"
git checkout -q main
git checkout -qb feat-b
printf 'beta 2\n' > b.txt
git commit -qam "feat: extend beta"
git checkout -q main
```

Expected:
- Detects the test command from manifest (`npm test` / `node test.js`).
- Creates a throwaway integration branch and merges each feature branch `--no-ff`, with tests between.
- Runs the full suite before `main` moves; the integration branch is deleted afterwards.
- Prompts user before deleting feature branches or removing worktrees.

## S2 — Conflicting branches
Setup: "Merge my worktree branches feat-a and feat-b."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "scripts": {\n    "test": "node test.js"\n  }\n}\n' > package.json
printf 'console.log("tests pass");\n' > test.js
printf 'shared base\n' > shared.txt
git add package.json test.js shared.txt
git commit -qm "chore: seed"
git checkout -qb feat-a
printf 'from a\n' > shared.txt
git commit -qam "feat: a rewrites shared"
git checkout -q main
git checkout -qb feat-b
printf 'from b\n' > shared.txt
git commit -qam "feat: b rewrites shared"
git checkout -q main
```

Expected:
- Stops at the conflict on the integration branch; names conflicting branch and `shared.txt`.
- No automatic conflict resolution is attempted.
- `main` remains at seed commit (never received a merge).

## S3 — Test failure after second merge
Setup: "Merge my worktree branches feat-a and feat-b."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "scripts": {\n    "test": "node test.js"\n  }\n}\n' > package.json
printf 'console.log("tests pass");\n' > test.js
printf 'alpha\n' > a.txt
git add package.json test.js a.txt
git commit -qm "chore: seed"
git checkout -qb feat-a
printf 'alpha 2\n' > a.txt
git commit -qam "feat: extend alpha"
git checkout -q main
git checkout -qb feat-b
printf 'console.error("regression"); process.exit(1);\n' > test.js
git commit -qam "feat: b breaks the suite"
git checkout -q main
```

Expected:
- Test failure is localized to `feat-b` (`feat-a` tested green first).
- Exact rollback is executed; `main` never receives the broken commit.

## S4 — Squash integration strategy
Setup: "Run `/skill:worktree-merge --strategy squash feat-a feat-b`."

Expected:
- Uses `git merge --squash` on the integration branch for each feature.
- Commits each squashed change with a conventional message.
- Lands on `<into>` with clean squashed history.

## S5 — Fast-forward integration strategy
Setup: "Run `/skill:worktree-merge --strategy ff feat-linear`."

Expected:
- Uses `git merge --ff-only` on the integration branch.
- Fails with explicit error if the branch is not a fast-forward of `<into>`.

## S6 — Cleanup keep mode
Setup: "Run `/skill:worktree-merge --cleanup keep feat-a feat-b`."

Expected:
- After successful integration into `<into>`, does not remove worktrees or delete feature branches.
- Reports surviving branches and worktrees intact.

## Baseline failure modes to watch for (RED)
- Merging directly into target base without a disposable integration branch.
- Skipping the per-merge test run.
- Auto-resolving conflicts.
- Deleting branches when `--cleanup keep` was specified.
