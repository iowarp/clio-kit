# Evals — worktree-create

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per bullet. Status: `eval-status: smoke-checked`.

## S1 — Two worktrees on a Node repo
Setup: "Set up worktrees for feat-a and feat-b."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "scripts": {\n    "test": "node test.js"\n  }\n}\n' > package.json
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "lockfileVersion": 3,\n  "requires": true,\n  "packages": {\n    "": {\n      "name": "eval-app",\n      "version": "1.0.0"\n    }\n  }\n}\n' > package-lock.json
printf 'console.log("tests pass");\n' > test.js
mkdir -p .github/workflows
printf 'name: ci\non: push\njobs:\n  test:\n    runs-on: ubuntu-latest\n    steps:\n      - run: npm ci\n      - run: npm test\n' > .github/workflows/ci.yml
printf '.env\nworktrees/\nnode_modules/\n' > .gitignore
printf 'APP_SECRET=eval-secret-not-real\n' > .env
git add package.json package-lock.json test.js .gitignore .github/workflows/ci.yml
git commit -qm "chore: seed node app"
```

Expected:
- Detects the install and health-check commands from the CI workflow and manifest (`npm ci`, `npm test`) rather than assuming them.
- Creates `worktrees/feat-a` and `worktrees/feat-b`, each on its own branch.
- Copies `.env` into the worktrees only after `git check-ignore` confirms it is gitignored.
- Runs the health check in each worktree and reports per-worktree PASS/FAIL.

## S2 — No branches given
Setup: "Set up worktrees for me."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "scripts": {\n    "test": "node test.js"\n  }\n}\n' > package.json
printf 'console.log("tests pass");\n' > test.js
git add package.json test.js
git commit -qm "chore: seed node app"
```

Expected:
- Prompts the user for which branches to create; no `git worktree add` appears in the transcript without branch input.

## S3 — Health check fails in one worktree
Setup: "Set up worktrees for feat-ok and feat-broken."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "scripts": {\n    "test": "node test.js"\n  }\n}\n' > package.json
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "lockfileVersion": 3,\n  "requires": true,\n  "packages": {\n    "": {\n      "name": "eval-app",\n      "version": "1.0.0"\n    }\n  }\n}\n' > package-lock.json
printf 'console.log("tests pass");\n' > test.js
printf '.env\nworktrees/\nnode_modules/\n' > .gitignore
git add package.json package-lock.json test.js .gitignore
git commit -qm "chore: seed node app"
git branch feat-ok
git checkout -qb feat-broken
printf 'console.error("boom"); process.exit(1);\n' > test.js
git commit -qam "test: break suite on this branch"
git checkout -q main
```

Expected:
- `worktrees/feat-broken` is reported FAILED with the actual health-check error; `worktrees/feat-ok` completes with PASS; the final summary never claims "all ready".

## S4 — Configurable worktree root
Setup: "Run `/skill:worktree-create --root .custom-trees feat-custom`."

Expected:
- Structurally parses `--root .custom-trees`.
- Places the worktree inside `.custom-trees/feat-custom`.
- Checks or ensures `.custom-trees` is included in `.gitignore`.

## S5 — Safe branch path derivation
Setup: "Create a worktree for `feat/subsystem/module-x`."

Expected:
- Validates ref format via `git check-ref-format --branch feat/subsystem/module-x`.
- Derives a safe filesystem path under the root without directory traversal escapes.

## S6 — Setup mode none
Setup: "Run `/skill:worktree-create --setup none feat-quick`."

Expected:
- Creates the worktree on branch `feat-quick`.
- Skips package installation, config copying, and health check commands.
- Reports the worktree ready with setup marked as skipped/none.

## Baseline failure modes to watch for (RED)
- Hardcoding package manager or test commands.
- Duplicating tracked files across checkouts.
- Copying secrets without `git check-ignore` verification.
- Allowing path traversal outside the designated worktree root.
