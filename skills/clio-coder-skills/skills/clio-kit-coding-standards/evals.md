# Evals — coding-standards

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`); skill is marked provisional.

## S1 — new TypeScript module
Prompt: "add a module that parses and stores webhook payloads."
Expected:
- Payload parsed at the boundary into a domain type; `unknown` never
  travels inward.
- Expected failures are tagged error values in return types, not throws.
- No `any`, `!`, or bare `as` casts; state modeled as a tagged union.

## S2 — host-repo conflict
Setup: repo's own instruction file mandates exceptions for error flow.
Expected:
- Host standard wins; skill's preference contained at the boundary, not
  imposed.

## S3 — test style
Prompt: "test the new module."
Expected:
- Real seams; no vi.mock/jest.mock; assertions on observable outcomes.

## Baseline failure modes to watch for (RED)
- Thrown errors for expected failures in new code.
- Boolean-flag APIs and `Partial<T>` inputs.
- Module mocks and spy-sequence assertions.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS (smoke). Skill loaded and produced its standards plan; judge emitted no parseable bullets (judge truncation, not a skill failure).

## Battletest record (2026-09-03)

S1 fixture: strict `tsconfig.json` (`strict`, `noUncheckedIndexedAccess`,
`exactOptionalPropertyTypes`), empty `src/`, `typescript` on PATH. Task:
add `src/webhook.ts` (parse unknown payload to a domain value, in-memory
store, report invalid payloads to the caller), then run `npx tsc --noEmit`.
`ornith1.5-35b-moe` on mini (llamacpp), `clio-coder run --autonomy
full-auto --json`, headless.

| run | wall | turns | in / out tokens | outcome |
|---|---|---|---|---|
| baseline (no skill) | 163s | 13 | 4.1k / 9.8k | typechecks; `{ok: boolean}` result with string errors; 5 `as` casts after manual checks |
| v0.2.0 | 693s | 7 | 4.6k / 42.6k | **no file written.** `allowed-tools: [read, grep]` narrowed the surface so `write` and `dispatch` were blocked (`skill_surface`); the model ended with a design-only report |
| v0.3.0 | 304s | 14 | 19.4k / 18.0k | typechecks; `_tag` discriminated `Result` and per-field tagged parse errors; 0 throws, 0 `any`, 0 `!`, 0 casts. Lost 5 turns cleaning a smoke script (`rm -f`, `find -delete`, `mv` to `/tmp` all refused) |

Root cause of the v0.2.0 failure: `src/core/skill-activation.ts` enforces
`allowed-tools` as a hard admission block, so a reference skill that names
only read tools makes the coding task impossible whenever it is the only
loaded skill. v0.3.0 declares no tool surface (same pattern as the meta
reference skills), adds an `## Arguments` contract, a four-line apply
checklist, and a scratch/cleanup rule (`.clio-coder/scratch/`, plain `rm`)
so the safety net stops eating turns.
