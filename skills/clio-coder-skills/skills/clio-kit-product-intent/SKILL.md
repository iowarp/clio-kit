---
name: clio-kit-product-intent
description: 'Use when this workflow is requested: Writes a problem-first product
  document for a greenfield effort: interviews for the problem, evidence, and a falsifiable
  hypothesis, with zero engineering decisions. Not for engineering decisions; use
  architecture. Not for turning a locked idea into milestone build prompts; use prd.
  Triggers on "why are we building this", "write the product thesis".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/planning/product-intent
  upstream-name: product-intent
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Product Intent

A PRD is intent: the problem and the hypothesis about solving it, in a form a
team can challenge before building and judge after shipping. Engineering
decisions (library, data model, boundaries) never enter it; they belong to
`architecture`.

## Arguments

```text
/skill clio-kit-product-intent <idea or problem, in a few sentences>
```

- The text is the raw idea, problem statement, or "just write it" request
  that starts Step 0. Reference docs (interviews, tickets, analytics,
  competitor notes) named or pathed in the request are evidence to read
  first, not more arguments.
- Nothing is required beyond some text; a blank invocation gets Step 0's own
  "What do you want to build? A few sentences." question.

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered, every
time; calling it again will not produce a different result. Treat the
first refusal exactly like the user saying "just write it" (see "If
the user declines the interview" below), and apply that treatment from
wherever it happened onward — Step 0's evidence check included, not just
the five clusters: state the question, your best evidence-grounded answer
(or, absent evidence, the most defensible product default) and the
reasoning, mark it `assumed — confirm`, and move to the next step. Never
invent evidence to back an assumption; one with nothing behind it stays an
open question, not a fact.

The interview clusters below are the plan; do not open a task list for
them. `tasks` sits outside this skill's tool surface and any call to it is
refused.

Two hard guards, checked before writing anything:

1. **Intent-framed.** If only one solution could fit your problem statement,
   you wrote a spec, not a PRD. "Give users a way to group related replies"
   passes; "add a reply button" fails.
2. **Never invent requirements.** Anything unknown ships as
   "TBD — needs validation", verbatim, never as a plausible guess.

## Step 0 — Ground in evidence

If the user passed reference docs (interviews, support tickets, analytics,
competitor notes), read them first and treat them as evidence. If none were
passed, ask whether any exist before interviewing.

## The interview

Ask in clusters via `ask_user`, and GATE: post the cluster, stop, wait for
answers. Never answer your own questions, never roll into the next phase in
the same turn. Thin answers get reflected back and dug into.

If the user declines the interview ("just write it"): honor it, name what
you will have to leave TBD, ask only the two or three highest-leverage
questions, and mark everything else "TBD — needs validation". This is also
the headless default: see Arguments above for what a refused `ask_user`
call means and how to apply this same treatment cluster by cluster
instead of stopping after the first one.

1. **Initiate.** Input given → restate and confirm. Blank → "What do you
   want to build? A few sentences." GATE.
2. **Foundation.** Who has the problem (a specific role, not "users")? What
   is the observable pain? How do they cope today (workaround, competitor,
   tolerating)? Why now? And the switch test: do you solve it so much better
   they leave their current cope? No switch case → no product yet; say so.
   GATE.
3. **Users.** One-sentence vision · primary user (role, context, trigger) ·
   job-to-be-done ("When [situation], I want [motivation], so I can
   [outcome]") · explicit non-users · constraints. GATE.
4. **Hypothesis.** Co-write the falsifiable bet. The WRONG condition is the
   most-skipped line and the one that makes it a hypothesis:

   ```
   We believe [change] will cause [users] to [do Y], resulting in [outcome].
   RIGHT if [leading signal] within [timeframe].
   WRONG if [counter-signal / a guardrail moves].
   ```

   No hypothesis ships without a WRONG condition. GATE.
5. **MVP and doors.** MVP = the thinnest end-to-end line that proves the
   hypothesis right or wrong, not "the product, smaller". Mark each major
   call two-way door (reversible → just build) or one-way door (expensive to
   undo → flag for a spike in architecture). GATE before generating.

Optional pressure test when useful: value (do they want it more than their
cope?), usability, feasibility, viability. Most ideas over-argue feasibility
and under-argue value.

## Generate

Write to `docs/<kebab-slug>.prd.md` (slug from the core idea; repo root if
no `docs/`; a user-supplied path always wins). Never hardcode `PRD.md` — a
second PRD must not overwrite the first. Sections, scannable, product only:

1. **Problem statement** — who, what pain, cost of not solving it.
2. **Evidence** — quote/data/observation, or "Assumption — validate via
   [method]".
3. **Thesis** — why this, why now, why it beats the current cope.
4. **Hypothesis** — the RIGHT/WRONG block, verbatim.
5. **Target user & JTBD** — plus non-users.
6. **MVP** — the thinnest proving line.
7. **Success metrics** — outcome-shaped: metric · target · how measured.
   "Engagement" is not a metric.
8. **Non-goals.**
9. **Open questions** — named checkboxes, never hidden.

## Done when

The file exists; the summary to the user leads with thesis and hypothesis,
counts evidenced-vs-assumed claims and open questions; and the next step is
offered: `architecture` for the engineering decisions this PRD
deliberately left open. Failing any of the five tests below means not done:
evidence-grounded problem · hypothesis with separate RIGHT and WRONG ·
outcome-shaped metrics · explicit non-goals · zero engineering decisions.

## Red flags

- A stack, library, database, or framework name anywhere in the document —
  "React + Postgres" appearing at all is an instant fail; that decision
  belongs to `architecture`, not here.
- A hypothesis with a RIGHT condition and no WRONG condition, or a WRONG
  condition that is just the RIGHT one negated instead of a real
  counter-signal.
- The literal filename `PRD.md`, or anything outside `docs/`, instead of
  `docs/<kebab-slug>.prd.md`.
- Calling `ask_user` again after a refusal, instead of switching to
  the decline treatment for every step from there on.
- Opening a task list for the interview clusters; `tasks` is refused.
- Reaching Generate without ever attempting Step 0 or the first cluster —
  the decline/headless treatment is a fallback for a gate that ran and came
  back empty, not a license to skip the loop from the start.
- Claims in the document that trace to neither the seeded evidence nor a
  marked assumption — an invented fact reads as confident and is the
  hardest failure to catch after the fact.
- Reaching for `bash` to grep or count-check the written PRD: `bash` is not
  in this skill's tool surface and the call is refused. Verify with `grep`
  and `read` instead.
