# Evals — resolve-merge-conflicts

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per bullet. Status: `eval-status: smoke-checked`.

## S1 — Compatible two-sided conflict
Setup: the merge of feat-rename stopped on a conflict in util.js. Resolve it and finish the merge.

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'function getUser(id) {\n  return db.find(id);\n}\n\nmodule.exports = { getUser };\n' > util.js
git add util.js
git commit -qm "chore: seed util"
git checkout -qb feat-rename
printf 'function findUser(id) {\n  return db.find(id);\n}\n\nmodule.exports = { findUser };\n' > util.js
git commit -qam "refactor: rename getUser to findUser"
git checkout -q main
printf 'function getUser(id) {\n  return db.find(id);\n}\n\nfunction getUserName(id) {\n  return getUser(id).name;\n}\n\nmodule.exports = { getUser, getUserName };\n' > util.js
git commit -qam "feat: add getUserName call site"
git merge feat-rename > merge-output.txt 2>&1 || true
```

Expected:
- Reads both sides' history (log/show on main and feat-rename) before editing.
- Resolution preserves both intents: the function is named `findUser` and `getUserName` calls `findUser`.
- The merge is committed and no conflict markers remain in `util.js`.

## S2 — Incompatible conflict
Setup: both sides changed the same default to different values.

Expected:
- Identifies the incompatibility. Picks per the merge's stated goal or prompts the user; trade-off recorded in the report.

## S3 — Mid-rebase, multiple stops
Setup: a rebase is stopped on its first conflict with more commits still to replay; resolve and finish the whole rebase.

Expected:
- Correctly treats `HEAD` as upstream target and `REBASE_HEAD` as incoming replayed commit (recognizes reversed ours/theirs).
- Continues the rebase with `git rebase --continue` across multiple stops until complete.
- Runs validation checks at completion.

## S4 — Binary file conflict
Setup: merge conflict on a binary asset (e.g. `logo.png`).

Expected:
- Detects binary conflict (no text conflict markers present).
- Determines intended version and resolves using `git checkout --ours` or `git checkout --theirs`.
- Stages binary asset with `git add logo.png`.

## S5 — Modify/delete conflict
Setup: one branch modified `helper.js`, the other deleted it.

Expected:
- Checks git status for `CONFLICT (modify/delete)`.
- Inspects the commit that deleted the file to determine if deletion was intentional.
- Resolves by either staging the file (`git add helper.js`) or completing the deletion (`git rm helper.js`).

## S6 — Operation-specific abort behavior
Setup: user requests aborting an in-flight rebase or merge.

Expected:
- For merge: invokes `git merge --abort`.
- For rebase: invokes `git rebase --abort`.
- Verifies working tree returns to clean pre-operation state.

## S7 — No-continue flag
Setup: user runs `/skill:resolve-merge-conflicts --no-continue`.

Expected:
- Resolves all conflict markers and stages resolved files with `git add`.
- Does not run `git merge --continue` or `git rebase --continue`.
- Reports files staged and ready for user inspection.

## Baseline failure modes to watch for (RED)
- Running `git checkout --theirs` on everything to make markers go away.
- Inverting intent during a rebase due to ours/theirs reversal.
- Leaving conflict markers in files.
- Running generic abort when resolution is achievable.

## Smoke record (2026-08-13)
One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning (30B local, llamacpp on mini), full-auto sandbox. PASS. Both sides' history read; both intents preserved; merge committed, no markers.
