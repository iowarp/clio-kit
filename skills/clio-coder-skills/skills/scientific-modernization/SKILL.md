---
name: scientific-modernization
description: 'Use when this workflow is requested: Modernizes, ports, rewrites, packages,
  or replaces established scientific software in independently validated stages against
  an external scientific oracle, settling compatibility and upstream stewardship before
  calling it complete. Not for an isolated benchmark; use experiment-protocol. Not
  for diagnosing wrong results; use scientific-debugging. Triggers on "modernize this
  scientific code", "rewrite this scientific software in Rust".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/research/scientific-modernization
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Scientific Modernization

Modernization preserves scientific behavior and stewardship; it is not source
translation. Faster code, a clean build, and passing self-authored unit tests
do not establish scientific equivalence. Work the stages below in order; each
has an explicit exit condition.

## Arguments

```text
/skill scientific-modernization <what to modernize, port, rewrite, or replace, and why>
```

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered. Stage 1's "the user has seen them" exit condition means,
headlessly: state the four bullets in your reply and proceed, never stall
waiting for acknowledgment. Every other stage gate below works the same way
— state the decision and its reasoning, then continue.

The six stages are the plan; do not open a task list for them — `tasks` sits
outside this skill's tool surface and any call to it is refused.

Shell rules for every `bash` call: one command per call, plain and direct.
Never use `$(...)` or backticks; they trigger an approval gate that ends a
headless run.

This is a long, multi-stage process on a small model or a time-boxed run. If
you are approaching your tool-call or time budget before reaching Stage 6,
stop at the current stage, state exactly which stage you reached and why you
stopped, and report the work as an incomplete prototype — never fabricate
completion of stages you did not actually reach.

## Stage 1 — Decide whether this work should exist

Identify the upstream project: active maintainers, license, release cadence,
supported users, contribution path. Prefer improving the original project when
coordination is viable.

Record the decision in writing before any implementation:

- path chosen: upstream contribution, maintained successor, or explicitly
  scoped fork;
- the accountable maintainer or organization by name;
- the compatibility surface users already rely on;
- release, deprecation, and migration intent.

Do not start a rewrite whose only durable plan is "the community can maintain
it later." Exit: the four bullets above are written down and the user has seen
them.

## Stage 2 — Establish an independent scientific oracle

Write the acceptance contract in `.clio-coder/validation.yaml` (preferred) or
`VALIDATION.md` before changing behavior. Pick at least one oracle that does
not depend on the new implementation agreeing with itself:

- exact outputs from a trusted reference implementation;
- parity against the established tool over a representative corpus;
- known statistical behavior with pre-registered bounds;
- simulated data whose correct answer is fixed in advance;
- conserved quantities, analytical solutions, or domain invariants.

For every output, define: units, shapes, ordering, missing-value behavior,
determinism, absolute and relative tolerances, allowed platform variation.
Include adversarial and historically troublesome inputs.

If no credible oracle exists, stop and report that limitation. Implementation
velocity cannot repair an undefined truth condition. Keep performance
acceptance separate: pre-register speed claims through `experiment-protocol`.
Exit: the contract file exists and names its oracle(s).

## Stage 3 — Freeze the compatibility envelope

Inventory observable behavior before migrating anything:

- CLI and API contracts, file formats, schemas, defaults, error behavior;
- packaging, fresh-install, upgrade, and uninstall paths;
- supported platforms, compilers, runtimes, accelerators, schedulers;
- resource scaling, reproducibility controls, provenance;
- undocumented conventions captured by downstream tests and real workflows.

Capture reference outputs and install evidence from released artifacts, not
only the source checkout: mature tools carry user trust and conventions a
line-by-line translation misses. Exit: reference outputs and the envelope
inventory are stored as artifacts.

## Stage 4 — Deliver in independently valid stages

Split the work into the smallest stages that can each be checked against the
oracle. Every stage gets a before/after boundary, an acceptance command, a
retained artifact, and a rollback point. Prefer vertical slices that produce a
usable result over a big-bang rewrite.

Per stage:

1. Capture the reference result on the pinned corpus and environment.
2. Make one bounded change.
3. Run compatibility and scientific-oracle checks.
4. Preserve raw outputs, discrepancies, and provenance.
5. Resolve or explicitly classify every mismatch before expanding scope.

An agent's confidence is not evidence. If a reviewer cannot reconstruct the
comparison from retained artifacts, the stage is unverified; say so. Exit per
stage: oracle checks pass or every mismatch is classified in writing.

## Stage 5 — Budget explicitly for the last mile

Initial implementation is faster than convergence. Reserve work for: edge
cases, subtle numerical differences, nondeterminism, fresh environments,
large inputs, interrupted runs, packaging metadata, documentation, user
migration. Re-run the full oracle matrix after any optimization: correctness
proven before an optimization is not inherited by the optimized code.

Use `scientific-debugging` when a mismatch needs causal diagnosis. Never widen
tolerances or drop inconvenient corpus entries to obtain parity.

## Stage 6 — Ship only with evidence and stewardship

A completion claim must include all of:

- the oracle and compatibility matrix that passed, plus retained artifacts;
- unresolved differences and their user-visible consequences;
- fresh-install and documented-workflow results;
- performance results, if claimed, under their registered protocol;
- upstream status, or the named owner and maintenance plan;
- migration, rollback, release, and deprecation instructions.

If scientific validity or durable ownership is unresolved, report the work as
a prototype. Do not call it a replacement, successor, or production release.

## Red Flags

- A rewrite begins before maintainers or downstream users are consulted.
- Tests derived only from the new implementation.
- "The outputs look close" replacing declared tolerance semantics.
- Performance wins reported before scientific parity.
- One final comparison substituting for staged validation.
- A passing happy path hiding fresh-install, scale, or edge-case failures.
- A fork shipping without an accountable owner and maintenance horizon.
