---
name: clio-kit-tdd
description: 'Use when this workflow is requested: Builds a feature or fix test-first,
  red to green at pre-agreed public seams, one vertical slice at a time. Not for designing
  benchmark criteria; use experiment-protocol. Triggers on "test-driven development",
  "write the test first".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/coding/tdd
  upstream-name: tdd
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Test-Driven Development

The red → green loop, run so it produces tests worth keeping. A good test
verifies behavior through a public interface and reads like a
specification: "user can checkout with valid cart" names a capability. The
implementation can change entirely; the test should not.

## Arguments

Arguments are passed in the user invocation message or via `/skill clio-kit-tdd`:

```text
/skill clio-kit-tdd [--runner command] [--file path] [--test-file path] <task description>
```

### Examples
- `/skill clio-kit-tdd implement parseDuration in parse-duration.js`
- `/skill clio-kit-tdd --runner "node --test" reproduce and fix token expiration bug`
- `/skill clio-kit-tdd --test-file tests/cart.test.ts checkout cart calculation`

### Options
- `--runner <command>`: The test runner command to execute (e.g., `node --test`, `npm test`, `pytest`, `cargo test`). If omitted, inspects `package.json`, project configuration, or existing test files.
- `--file <path>`: The target implementation source file to create or update.
- `--test-file <path>`: The target test file to create or update.

### Remaining text
- Everything after the options is the feature specification or bug
  description. If it names the seam already, that is the seam; do not ask
  again.

The two steps below are the plan; do not open a task list for them.

## Step 1 — Agree the seams

A seam is the public boundary you test at, observing behavior without
reaching inside (e.g. exported functions, class methods, or CLI interfaces). Before writing any test:

1. Read the project's instruction file and inspect existing tests/runner configuration (`package.json`, `Makefile`, etc.) so naming, test runner, and test conventions match the host project.
2. Formulate the public seam under test:
   - Target function or module name
   - Input arguments and expected return types
   - Edge case and error behaviors
3. **Headless / Autonomous Fallback**: If running headlessly or if seams are specified in the prompt or clearly evident from module exports, state the agreed seam explicitly in your response (e.g. `Seam agreed: parseDuration(str) -> number | null`) and proceed immediately to Step 2 without waiting for an interactive prompt. When interacting with an operator, confirm the proposed seam before writing code.

No test is written at an unconfirmed or unstated seam. Agreeing seams up front keeps the effort focused on critical public paths rather than internal details.

## Step 2 — The loop (Strict Vertical Slices)

Execute one vertical slice per cycle: exactly one test behavior → minimal implementation → verify.

### Cycle Rules:
1. **Red**:
   - Write or append **EXACTLY ONE** test case (`test(...)` or `it(...)`) for the thinnest unverified slice of behavior.
   - Double-check expected literal values and arithmetic beforehand to avoid tautological or mathematically flawed assertions.
   - Run the test suite directly via `bash` (e.g. `node --test test/parse-duration.test.js`).
   - Observe it fail for the expected reason (e.g. function not defined, or assertion difference).
   - If the test passes immediately on the first run, the test verified nothing: fix the test before proceeding.
2. **Green**:
   - Write or edit **ONLY** enough implementation code to make that failing test pass.
   - Do not write speculative helpers, future error checks, or unrequested features.
   - Run the test runner again. Confirm that the test now passes.
3. **Repeat**:
   - Move to the next slice of behavior (e.g. next format, edge case, or invalid input), adding one test case at a time.
   - Keep all previously written tests passing (no regressions).

### Shell Execution Constraints:
- Never use command substitution `$(...)` or backticks `` ` `` in `bash` commands; execute commands in discrete, direct steps.
- Avoid complex nested shell pipelines (e.g. `cmd 2>&1 | head -40; echo EXIT: ${PIPESTATUS[0]}`). Run the test runner directly:
  ```bash
  node --test <test-file>
  ```
  or
  ```bash
  npm test
  ```
- If the test command cannot execute at all (runner missing, syntax error in test setup, execution blocked), STOP and report the exact failure. Never fabricate test output or assume a test passed without running it.

### Batching and Git Rules:
- **No Horizontal Slicing**: Do NOT write a large batch of tests (e.g. 5–10 test cases) upfront before writing any implementation. Writing multiple tests at once breaks the red-green feedback loop and creates compound debugging failures on smaller models.
- **No In-Loop Commits**: Do not run `git commit` or `git add` between cycles. TDD is complete when the suite passes green; repository shipping is handled separately by `ship`.

## Anti-patterns (reject the test, not the code)

- **Horizontal slicing**: Writing a full suite of tests before any implementation exists.
- **Implementation-coupled**: Mocks internal collaborators, tests private functions, or asserts through side channels. Tell: the test breaks on refactoring while behavior is unchanged.
- **Tautological**: The assertion recomputes the expected value the same way the code does (`expect(add(a,b)).toBe(a+b)`), so it passes by construction. Expected values must come from independent literals or specification examples.
- **Mock-everything**: Heavy mocking instead of testing real boundaries. When mocking strategy is in question, consult `references/mocking.md`. For worked examples, consult `references/tests.md`.

## Done when

Every agreed seam has its behaviors covered by tests that were each observed red before green, the full suite passes, and no test trips the anti-patterns above. Output a concise summary naming:
1. Public seams covered.
2. Behaviors verified.
3. Any edge cases or seams deliberately left untested.

## Red flags

- Writing a batch of tests upfront instead of one vertical slice per cycle.
- An implementation written before the test it claims to satisfy.
- A test passing green on its initial run without an observed red failure.
- Changing test assertions to match incorrect code outputs instead of fixing the code.
- Staging or committing git changes during the TDD loop.
- Using bash command substitutions `$(...)` that trigger approval modals.
