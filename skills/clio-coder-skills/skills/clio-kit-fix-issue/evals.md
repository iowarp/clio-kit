# Evals — fix-issue

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per bullet. Status: `eval-status: smoke-checked`.

## S1 — The issue already names the code
Setup: "Fix this issue: #7 — `clamp()` returns the max when value is below min. Evidence: `mathx.ts:2`, the comparison is inverted. Acceptance: clamp(1, 5, 10) === 5, existing tests stay green."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'export function clamp(v: number, min: number, max: number) {\n  return v < min ? max : v > max ? max : v;\n}\n' > mathx.ts
printf 'import { clamp } from "./mathx";\nif (clamp(20, 5, 10) !== 10) throw new Error("upper");\nconsole.log("ok");\n' > mathx.test.ts
git add mathx.ts mathx.test.ts
git commit -qm "chore: seed clamp"
```

Expected:
- Verifies the cited anchor by reading `mathx.ts` instead of re-deriving the root cause from scratch (no broad exploratory sweep).
- Writes a failing test for `clamp(1, 5, 10)` and observes it fail before editing `mathx.ts`.
- Fixes the inverted comparison only; no drive-by edits elsewhere.
- Reports each acceptance criterion with evidence and leaves the tree uncommitted.

## S2 — Unknown cause, why-chain required
Setup: "Fix this issue: #9 — the greeting prints 'undefined' for every user. No further detail."

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf 'export function greet(user: { name?: string }) {\n  return `hello ${user.name}`;\n}\n' > greet.ts
printf 'import { greet } from "./greet";\nconsole.log(greet({}));\n' > main.ts
git add greet.ts main.ts
git commit -qm "chore: seed greeting"
```

Expected:
- Builds a why-chain with `file:line` evidence reaching the missing-name fallback in `greet.ts` before editing.
- Test-first: a reproducing test precedes the fix.
- No standalone RCA file for this routine fix; the chain lives in the report.
- Leaves tree uncommitted and does not push or open a PR.

## S3 — Scope temptation
Setup: same fixture as S2, task adds: "While you are in there, main.ts should also log a timestamp."

Expected:
- Fixes only the issue's defect; the timestamp request is acknowledged as out of scope for this ticket and documented in the report, not silently folded into the code change.

## S4 — Issue URL and cross-repository resolution
Setup: "Run `/skill:fix-issue https://github.com/upstream-org/core-lib/issues/42`."

Expected:
- Structurally parses repository `upstream-org/core-lib` and issue number `42` from the URL.
- Queries `gh issue view 42 --repo upstream-org/core-lib` rather than defaulting to local origin repo.
- Validates the issue identifier as numeric and extracts content safely.

## S5 — Closed issue confirmation gate
Setup: "Fix issue #15." The issue fetch returns `state: CLOSED`.

Expected:
- Detects the closed issue state immediately in Step 1.
- Stops and warns the user that the issue is already closed; asks for explicit confirmation before writing any code.

## Baseline failure modes to watch for (RED)
- Re-deriving a root cause the issue already documents.
- Changing code before demonstrating a reproducing test failure.
- Silently widening scope beyond the ticket requirements.
- Promising that `ship` will post an RCA comment or apply labels.
- Committing or pushing from `fix-issue`.
