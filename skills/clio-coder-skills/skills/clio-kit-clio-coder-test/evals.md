# Evals — clio-coder-test

Retrieval + application scenarios. Run a subagent WITHOUT the skill to capture
the gap (it cites the dead unit/integration/e2e taxonomy), then WITH it.

## T1 — pick the layer
Prompt: "I changed pure logic in `src/domains/dispatch/validation.ts`. What do I
run and why?"
Expected:
- Locates the closest contract coverage and starts with
  `npm run test:file -- tests/contracts/dispatch-admission.test.ts`, adding
  other dispatch files only when the changed path reaches them.
- Runs `npm run lint` if imports changed.
- Explains contracts import `src` via tsx, so no build is needed.
- Does NOT suggest `test:unit` / `test:e2e` (those don't exist).

## T2 — CLI change needs a build
Prompt: "I edited `src/cli/run.ts`. How do I verify end-to-end?"
Expected:
- Builds (or relies on `npm run dev` watch), then starts with
  `npm run test:file -- tests/extended-smoke/cli-core.test.ts`.
- Explains smoke spawns `dist/cli/index.js`, so it only sees built code.

## T3 — hot reload
Prompt: "How do I keep testing without rebuilding every time?"
Expected:
- Fast loop (contracts/boundaries, tsx, no build) for logic/contracts.
- `npm run dev` (`scripts/build.ts --watch`) keeps `dist/` fresh for smoke.
- States there is no in-process code reload of a running session; restart for
  interactive testing. Distinguishes this from config hot-reload (classify.ts).

## T4 — boundary violation
Prompt: "`npm run lint` says a domain imports another domain's extension.ts.
Quickest fix?"
Expected:
- Route through the target domain's `index.ts` contract (rule3). Does NOT
  suggest a `biome-ignore` or exclude.

## Baseline failure modes to watch for (RED)
- Cites `test:unit`/`test:integration`/`test:e2e`, the deleted shared spawn
  helper, or the deleted PTY harness.
- Claims smoke tests run against source (they run against `dist/`).
- Invents a hot-reload feature that reloads a running session's code.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT CLEANLY RUN: scenario id is T1; driver's --scenario S1 exited 2; re-run did not land before the time-box.

## T5 — current application gates
Prompt: "The web API changed. Which checks belong to this release?"
Expected:
- Builds the root CLI for ACP fixtures, then runs
  `pnpm --filter @iowarp/clio-coder-web verify` with temporary state outside the checkout.
- Names app type, API/worker boundary, test, client build and headless-browser checks.
- Explains root `ci:release` includes web tests and the installed package audit.
- Does not run the retired trace viewer or reference-only Workbench gates.

RED: the prior 0.3.0 instructions prescribe both obsolete application lanes.
GREEN: the current instructions name only the unified app lane and retain the
independent CLI/TUI/headless checks. This is a source/command reconciliation,
not a claim of a new live-model skill evaluation.
