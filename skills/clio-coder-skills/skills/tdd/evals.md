# Evals — tdd

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`).

## S1 — small feature, test-first
Setup: add a parseDuration function (strings like "1h30m" to seconds)
test-first.

Fixture:
```bash
printf '{\n  "name": "eval-app",\n  "version": "1.0.0",\n  "private": true,\n  "type": "commonjs",\n  "scripts": {\n    "test": "node --test"\n  }\n}\n' > package.json
```

Expected:
- Seams proposed and confirmed before any test.
- Each cycle: one failing test observed red, then minimal implementation.
- Expected values are independent literals, not recomputed formulas.
- Final report names covered and deliberately-uncovered seams.

## S2 — bug fix
Setup: seeded bug with a reproducing input.
Expected:
- First move is a failing test reproducing the bug at a public seam.
- Fix follows; test goes green; suite stays green.

## S3 — refactor temptation
Setup: mid-loop, adjacent ugly code invites cleanup.
Expected:
- Refactor deferred out of the red-green loop; noted for a separate pass.

## Baseline failure modes to watch for (RED)
- Implementation written first, tests back-filled.
- A batch of tests written up front (horizontal slicing).
- Tests mocking the module under test's internals.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS on re-run under working exec: red observed, then green. The earlier exec-gated run produced a fabricated pass table, which motivated the no-fabricated-verification rule now in the body.

## Empirical Battletest (2026-09-03)

Tested with `ornith1.5-35b-moe` via mini server (`http://192.168.86.141:8080`) on S1 parseDuration fixture:
- Baseline (No skill): 17 turns, 75.49s, excessive `tasks` churn (9 task calls), incomplete seam articulation.
- Skill V1: 24 turns, 234.42s; horizontal slicing (11 tests upfront) led to iterative thrashing and test arithmetic bugs.
- Hardening applied (v0.4.0): Narrowed tool surface to `read`, `grep`, `ls`, `bash`, `write`, `edit` (dropped `git` and `ask_user` to eliminate modal risk and commit churn), added structured `## Arguments` specification, enforced strict vertical slices (one test case per cycle), banned upfront test batching and bash command substitutions `$(...)`, and added deterministic headless seam confirmation.
- Skill V2 (Hardened): 22 turns, 223.82s, 0 task churn, 0 modal warnings. Executed 4 flawless red-to-green cycles sequentially:
  1. `45s -> 45` (observed red `MODULE_NOT_FOUND`, then minimal code green)
  2. `2h -> 7200` (observed red assertion failure, updated code green)
  3. `1h30m -> 5400` (observed red, updated code green)
  4. `1h30x -> null` (observed red, updated code green)
  5. Final `npm test` gate passed with 4 passing tests, zero regressions.


Follow-up (2026-09-03, same session): re-read of the v0.4.0 run showed one
blocked `tasks` plan call (`skill_surface`) and otherwise clean sequential
cycles. Added "the two steps below are the plan; do not open a task list"
and replaced the "Unknown Arguments and Validation" wording with a plain
"remaining text is the spec" rule. No re-run; the change is prose only.
