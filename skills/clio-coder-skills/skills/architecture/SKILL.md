---
name: architecture
description: 'Use when this workflow is requested: Decides the engineering approach
  for an intent in an interactive session: investigates, proposes two or three genuinely
  different approaches with trade-offs, recommends, lets the user decide, and writes
  the decision doc. Not a task-by-task plan; use cut-it. Not a multi-perspective debate;
  use design-council. Not product intent; use product-intent. Not a typed implementation
  handoff; use tech-spec. Triggers on "how should we build this", "pick the stack".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/planning/architecture
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Architecture

The intent says what and why; this session decides how: approach, stack,
data shape, boundaries, and what to de-risk first. The deliverable is a
high-level decision doc — the moment you start listing file edits or task
steps, you have gone too deep; pull back up to the decisions.

This is interactive. Run the loop out loud with the user, via `ask_user`:

```
investigate → 2-3 options with trade-offs → recommend + reasoning →
ask → wait for their call → go deeper
```

Never converge silently. Optimize for the user's goals, familiarity (a
stack they know beats a "better" one they don't), leanness (decide only
what is needed to move), and reversibility (spend deliberation on the
expensive calls only).

## Arguments

```text
/skill architecture <the intent — a PRD path, a brief, or a few sentences>
```

- The text is what to decide the engineering approach for. A PRD path,
  brief, or reference doc named or pathed in the request is Step 0's
  grounding to read first, not more arguments.
- Nothing is required beyond some text; a blank invocation falls straight
  to Step 0's own question — what are we building, and is there a written
  intent to read — rather than inventing a project to architect.

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered, every
time; calling it again will not produce a different result. From wherever
the first refusal lands — Step 0's "ask whether reference docs
exist", Step 1's greenfield/brownfield call, or any decision in Step 2 —
switch immediately to the treatment already described above (state the
options, the recommendation, the reasoning, adopt it, mark it `assumed —
confirm`) and keep running every remaining step through Step 3 in the same
turn. Never stop at the first unanswered gate and never go silent; a
one-shot document produced without ever attempting the loop is also wrong
in the other direction — always investigate and state the loop out loud
first, degrading only once a gate actually goes unanswered. Never invent a
fact, evidence, or a codebase detail to back an assumption; anything
genuinely uncertain becomes an open question or a spike, never a plausible
guess.

The steps below (Step 0 through Step 3) are the plan; do not open a task
list for them. `tasks` sits outside this skill's tool surface and any call
to it is refused. `bash` is also outside this skill's tool surface —
investigate with `read`, `grep`, `find`, `git`, and `code_nav` instead;
reaching for `bash` to explore or verify is refused.

## Step 0 — Ground

Read the intent (PRD path, brief, or the user's words). Read any reference
docs passed alongside; if none were passed, ask whether any exist — the
context needed is usually already written down. To see what exists in the
repo before reading it, use the `find` tool (e.g. pattern `**/*`) or `ls`,
never `bash find`/`bash ls` — `bash` is refused outright, see Arguments.

## Step 1 — Greenfield or brownfield

Infer from the input and workspace; genuinely unclear → ask. It changes the
exploration:

- **Greenfield**: explore the solution space — approaches, current best
  practice (web_fetch when it materially helps), first principles.
- **Brownfield**: read the relevant existing surfaces first — where this
  plugs in, what it reuses, what it must not break. High-level read, not a
  file audit.

## Step 2 — Explore, decision by decision

For each area, surface options, recommend with reasoning, ask, record the
user's call. Skip what does not apply and say that you skipped it:

- **Approaches**: 2-3 genuinely different directions with trade-offs.
- **Stack & libraries**: what and why (fit, maturity, familiarity), with
  alternatives named.
- **Data model**: main entities, relationships, storage — shape level, not
  columns and migrations.
- **Boundaries & contracts**: auth posture, secrets, external services,
  the API/integration boundaries crossed. Flag, never gloss.
- **Missing pieces**: what has to exist that does not yet.
- **Spikes**: any uncertain or one-way-door call gets a spike, not a guess:

  ```
  Question:      <what we are unsure about>
  Spike:         <smallest thing to build/test> over <timebox>
  Decision rule: <X> if <signal> / <Y> if <counter-signal>
  ```

  Reversible cheap calls: just decide and move on.

## Step 3 — Write the decision doc

Only after the calls are made. There are exactly three valid locations, in
order of default preference — pick one, never a fourth:

1. `docs/architecture-<slug>.md` — the default, no exceptions needed.
2. An `## Architecture` section folded into the named PRD, only when the
   user said they prefer one document.
3. A tracker page, only when the user names one and an integration exists.

No other filename or location is ever correct. `final_report.md`,
`REPORT.md`, `summary.md`, and anything similar are hard misses, not
stylistic variants — if none of the three above fit, use option 1. Shape:

```markdown
# Architecture — <intent name>

## Problem & goals        (one paragraph, the lens for every decision)
## Approaches considered  (the 2-3 directions with trade-offs)
## Recommended approach   (the shape of the solution; brownfield: where it
                           plugs in and what it reuses)
## Key decisions          (stack+why, data model shape, boundaries,
                           other eng-lead calls; skipped areas noted)
## Missing pieces
## Spikes & experiments   (each with its decision rule)
## Open questions         (deliberately deferred, with what settles each)
```

## Done when

The doc exists at a confirmed location, every section is either filled or
explicitly marked skipped, every one-way-door call has a spike or a user
decision, and the user has been offered the next moves without a forced
pipeline: slice into tickets (`backlog`), slice into a local
sprint (`cut-it`), spike a flagged risk now, or keep refining here.

## Red flags

- A document produced in one shot with no user calls in between.
- One foregone conclusion instead of real alternatives.
- File-by-file edit lists (that is cut-it's altitude).
- A one-way door decided by vibe instead of a spike.
- A decision doc at any filename other than the three valid locations
  above — `final_report.md`, `REPORT.md`, and similar are wrong every time.
- A headless run that stops at the first unanswered `ask_user` call instead
  of running the assumed-confirm treatment through every remaining step, or
  one that skips straight to Step 3 without ever running the loop out loud.
- Opening a task list for Steps 0-3; `tasks` is refused. Reaching for
  `bash` to explore the repo or verify the written doc; `bash` is not in
  this skill's tool surface and the call is refused — use `read`, `grep`,
  `find`, `git`, and `code_nav` instead.
- A claim in the doc that traces to neither the read intent/codebase nor a
  fact marked `assumed — confirm` — an invented detail reads as confident
  and is the hardest failure to catch after the fact.
