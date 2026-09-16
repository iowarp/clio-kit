# Evals - scientific-modernization

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - rewrite an established genomics tool in Rust

Setup: a maintained genomics parser has a large user base, a reference CLI,
and representative public datasets. The new repository has no validation
contract. Prompt: "Rewrite this parser in Rust and make it the new default."

Expected:

- Checks the upstream contribution path, maintainers, license, and downstream
  compatibility obligations before choosing a fork or successor.
- Writes an acceptance contract before implementation, using the released
  parser and fixed datasets as an independent parity oracle.
- Specifies file, API, CLI, error, ordering, missing-value, and numerical
  compatibility rather than comparing only happy-path output.
- Splits the migration into independently validated stages with retained raw
  comparison artifacts and rollback points.
- Calls the result a prototype unless ownership, release, migration, and
  maintenance are settled.

## S2 - port a numerical solver to GPU

Setup: a CPU solver is trusted, but floating-point reduction order will change
on the GPU. Prompt: "Port the solver to CUDA and prove it is correct and
faster."

Expected:

- Uses the CPU implementation, analytical cases, conserved quantities, or
  fixed simulated truth as an oracle independent of the CUDA code.
- Defines units, shapes, absolute and relative tolerances, nondeterminism, and
  platform variation before the first GPU result.
- Separates correctness parity from a pre-registered performance experiment
  and invokes experiment-protocol for the latter.
- Revalidates the full oracle matrix after each optimization rather than
  inheriting correctness from an earlier version.
- Includes large inputs, fresh environments, failures, and numerical edge
  cases in the last-mile plan.

## S3 - modernize packaging in place

Setup: a scientific Python package has a legacy build, multiple supported
platforms, and an active upstream. Prompt: "Replace the packaging system and
release it without changing scientific behavior."

Expected:

- Prefers an upstreamable bounded change over creating a replacement package.
- Captures fresh-install, upgrade, uninstall, import, CLI, and documented
  workflow behavior from released artifacts before editing the build.
- Uses existing scientific outputs as a no-regression oracle even though the
  requested change appears packaging-only.
- Delivers the migration in reversible stages and records platform-specific
  evidence.
- Names release ownership, deprecation impact, and rollback instructions.

## S4 - anti-triggers

Setup: one request asks why a solver emits NaNs; another asks to benchmark two
MPI collectives without migrating software.

Expected:

- Routes the NaN diagnosis to scientific-debugging.
- Routes the isolated performance comparison to experiment-protocol.
- Does not manufacture a modernization or stewardship project for either.

## Baseline failure modes to watch for (RED)

- Starts a big-bang rewrite immediately and treats self-authored tests as proof.
- Declares parity from a few hand-picked examples with no tolerance semantics.
- Mixes correctness and performance so a speedup excuses changed answers.
- Discovers edge cases only in one final comparison after the rewrite.
- Replaces an active project without early maintainer coordination.
- Calls an unowned fork production-ready because installation succeeds.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT COMPLETED: loop guard at 75 tool calls. The run also surfaced two harness containment findings (write-tool workspace escape; cross-arm workspace visibility) — harness issues, not skill issues.

## Battletest record (2026-09-03)

Same structural gap as the other three research skills: no `## Arguments`,
no `tasks`-refusal, no shell-rules paragraph, no no-operator statement. Added
all four, plus an explicit budget-awareness line (state which stage you
reached and report the work as an incomplete prototype rather than
fabricate completion) directly answering the 2026-08-13 record's loop-guard
finding above.

This mission's plan called for a small legacy-C fixture and a live A/B pass
on `mini`; that track was killed mid-run this session for taking materially
longer than the wall-clock budget available (this is a six-stage,
`model-size: large` skill — the prior smoke record already didn't complete
on a 30B model, and a fresh fixture never got built before the track was
stopped). **This pass is a static hardening pass only** — the four additions
above were applied and `npm run skills:pin && npm run skills:check && npm
run lint` verified green, but no fresh live run against this skill's
hardened body exists yet. Treat 0.4.0 as unverified beyond the structural
fix; it needs the small legacy-C fixture (a single-file numeric routine with
a captured reference output as the oracle) and a real baseline/hardened A/B
pass before its `eval-status` can move past `scenarios-recorded`.

**Still weak**: everything — no fixture built this pass, no run executed
against 0.4.0, no cross-model confirmation. This is the one skill in the
category that did not get real battletest evidence this session; say so
plainly rather than implying otherwise from the version bump.
