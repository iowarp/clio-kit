---
name: clio-kit-tech-spec
description: 'Use when this workflow is requested: Writes a typed call-stack architecture
  handoff: code-shaped contracts plus execution flows, implementation-ready for another
  engineer. User-invoked only. Not for weighing approaches or deciding the design;
  use architecture first. Triggers on "write a tech spec", "typed call-stack handoff".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/planning/tech-spec
  upstream-name: tech-spec
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Tech Spec

A tech spec is a typed call-stack architecture handoff: contracts as
TypeScript pseudocode plus end-to-end execution flows. Prose explains why;
types and call stacks define what changes. Design only — never implement,
and save a file only when the user asks; otherwise return the spec inline.

## Arguments

```text
/skill clio-kit-tech-spec <the change, in a few sentences, or a path to read first>
```

- The text is the design problem: what's changing and why. A doc or file
  path named in the request (a PRD, an architecture decision, a module) is
  context to read, not more arguments — see "Load local context" below.
- Nothing is required beyond some text; a blank invocation falls straight
  to Path B's first question rather than inventing a change to spec.
- **Output defaults to inline.** Write a file only when the request says
  so explicitly — "save it", "write it to `<path>`", "put it in `docs/`".
  Absent that, the finished spec is the reply itself: no file, in this run
  or a prior one in the same session, gets created for it. This holds
  regardless of which path below runs or how long the spec is — length is
  never itself a reason to write a file.
- Disabled for model self-invocation and requires the `clio-kit-tdd` skill be
  installed to reference in the TDD Test Plan section; both are frontmatter
  facts, not something to explain to the user unless asked.

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered, and a
second try will not resolve differently. In Path B (below), that means: state the
question, your recommendation grounded in the codebase and any docs read
(or the most defensible engineering default when nothing grounds it), and
the reasoning; adopt the recommendation; mark it `assumed — confirm`; move
to the next question. Run every question this way, end to end, not just
the first — the interview is the plan to execute, not an outline to
abbreviate because no one answered the opening question. Never invent a
fact or a codebase detail to back an assumption; anything genuinely
unknown becomes an Open Question in the spec, not a plausible guess. This
degrades the interview only — it never licenses writing a file that
wasn't asked for.

The steps below are the plan; do not open a task list for them. `tasks`
sits outside this skill's tool surface and any call to it is refused.
`bash` is also outside this skill's tool surface — verify what you wrote
with `grep`, `read`, and `find`, never `bash`.

## Choose the path

- **Path A — convert context to spec**: the conversation, docs, or codebase
  already describe the change.
- **Path B — interview first**: not enough problem, constraints, or
  acceptance criteria exist. Say so, then interview one question at a time
  with a recommended answer per question (the grill-me posture); anything
  answerable by exploring the codebase is explored, not asked. When context
  suffices, run Path A. Never invent requirements to skip the interview.
  See Arguments above for how a headless run carries every question
  through instead of stalling on the first one.

## Path A

1. **Load local context.** Inspect existing code and docs for vocabulary,
   module layout, error style, adapters, test conventions. Done when the
   spec introduces no pattern, library, or schema style without checking
   local precedent.
2. **Extract the design problem.** Current state, problem, callers, goals,
   non-goals, constraints, invariants, affected systems, entrypoints,
   risks. Unknowns become open questions, never plausible filler. Done when
   every claimed constraint traces to conversation, code, docs, or an open
   question.
3. **Explore alternatives.** 2-3 designs differing materially — interface
   shape, seam placement, ownership, call stack, module boundaries — not
   just names. Sketch each: domain types, interfaces, failure types, seams,
   entrypoint-to-effect call stack, test strategy, trade-offs. Compare on
   caller burden, module depth, locality of change, seam placement,
   testability through real seams, complexity. Done when the
   recommendation is chosen after the comparison, not before.
4. **Specify the contracts.** For the recommended design, every new,
   changed, or deleted type, signature, interface, error union, adapter,
   DTO, and API gets a concrete sketch — or an explicit reason no new
   contract is needed. Name what each layer may know and what must not
   leak across each seam.
5. **Specify call stacks.** Every affected behavior gets an entrypoint →
   side-effect → response trace with the type/data flow (`raw input →
   parser → domain input → service → adapter → typed result → projection →
   output`), current-vs-proposed when changing behavior, and failure /
   retry / idempotency flow where reachable.
6. **Map files.** Files to add, change, delete; test files; config or
   migration files. Each maps to the contract or call-stack step it owns.
7. **Write the TDD plan.** Vertical red-green slices per the `clio-kit-tdd` skill:
   one failing behavior test, minimal implementation, repeat. Cover happy
   paths, failure paths, parser rejection, invariants, changed seams — or
   state why a path is deliberately untested.

## Spec outline

Use this shape (omit sections that truly do not apply, but never omit
contracts, seams, call stacks, or the test plan for being hard):

```md
# <Title>
## Summary · Context / Current State · Goals · Non-Goals · Invariants ·
## Design Constraints
## Alternatives Considered (Option 1..3)
## Recommendation · Proposed Design
## Domain Model and Types · Types, Interfaces, and APIs
## Seams, Boundaries, Adapters, and Implementations
## Call Stacks and Data Flow (current / proposed / failure / retry)
## Files to Add / Change / Delete
## TDD Test Plan
## Risks and Open Questions
```

## Done when

The spec follows the outline, every boundary has a typed contract or a
stated reason it needs none, every behavior has a call stack, unknowns are
open questions rather than invented design, nothing was implemented, and
no file was written unless the request asked for one.

## Red flags

- Prose where a type signature would be more precise.
- A recommendation written before the alternatives.
- Speculative seams no invariant, boundary, or test earns.
- The same rule restated in three sections.
- "While I'm here" implementation.
- Writing the spec to a file when nothing in the request asked for one —
  the default output is always the inline reply.
- A Path B question left unanswered instead of run as the assumed-confirm
  monologue, or an interview skipped straight into Path A without ever
  asking the first question.
- Opening a task list for the steps above; `tasks` is refused. Reaching for
  `bash` to grep or verify the spec; `bash` is not in this skill's tool
  surface and the call is refused — use `grep`/`read`/`find`.
