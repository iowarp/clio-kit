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
