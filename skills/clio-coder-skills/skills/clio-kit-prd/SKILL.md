---
name: clio-kit-prd
description: 'Use when this workflow is requested: Turns an idea into a product requirements
  document through a phase-gated interview, ending in PRD.md plus per-milestone prompt
  files ready to drive a coding agent. Not for the problem-first product thesis; use
  product-intent. Triggers on "write a PRD", "spec this product out".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/planning/prd
  upstream-name: prd
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# PRD

Guide the user from a raw idea to a locked product spec and executable
milestone prompts. The discipline is the phase gate: each phase produces a
small locked artifact that the next phase builds on. No phase reopens without
the user saying so. Markdown only, no external templates, repo-aware.

## Arguments

```text
/skill clio-kit-prd <brain dump or idea, in a few sentences>
```

- The text is the raw brain dump that starts phase 1. An existing intent
  doc, PRD, or evidence file named or pathed in the request is material to
  read first (see "Read the repo before asking" below), not more arguments.
- Nothing is required beyond some text; a blank invocation gets phase 1's
  own prompt — let the user describe the idea raw — rather than an invented
  idea.

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered, and
stalling a gate to wait for it never resolves.
When a gate goes unanswered, do not skip the phase and do not go quiet: run
it as a monologue instead — state the phase's question, your
recommendation (grounded in the repo and any evidence read, or the most
defensible product default when nothing grounds it), and the reasoning,
adopt the recommendation, mark it `assumed — confirm`, and move to the next
phase. All nine phases still run, end to end, in one turn — the phase list
below is the plan to execute, not an outline to abbreviate because no one
answered the first gate. Never invent evidence or a fact to back an
assumption; anything genuinely unknown stays an open item, marked as such,
not a plausible guess.

The nine phases below are the plan; do not open a task list for them.
`tasks` sits outside this skill's tool surface and any call to it is
refused. `bash` is also outside this skill's tool surface — verify what you
wrote with `grep`, `read`, and `find`, never `bash`.

## Interview mechanics

- Use the `ask_user` tool for every confirmation and choice, with your
  recommendation as the first option: post the question, stop, wait for the
  answer. See Arguments above for what a gate that goes unanswered means and
  how to carry every phase through anyway.
- **Read the repo before asking.** Stack, conventions, existing entities, and
  integrations are facts; discover them and *confirm*, never ask cold. An
  entity or module that already exists gets reused and marked as such, never
  re-specced as new work.
- Keep each phase to one or two exchanges. Synthesize, propose, lock, move on.

## The phases (in order, each locks before the next)

1. **Brain dump.** Let the user describe the idea raw. Do not structure yet;
   capture it.
2. **Core purpose.** Synthesize a 1–3 sentence mission statement. Confirm.
3. **Top-level features.** Propose 4–8 in-scope features derived from the
   dump. Confirm the set.
4. **Out of scope.** Propose explicit v1 cuts — what this deliberately does
   not do. This list prevents scope creep later; make it real.
5. **Stack and foundation.** Detect the stack from the repo (manifests, lock
   files, configs); confirm it and inventory what foundation already exists.
6. **Integrations and credentials.** Map external services and the API
   keys/credentials each requires. Note which exist versus which the user
   must provision.
7. **Data model.** Entities, fields, and relationships in plain language.
   Reuse existing repo entities where they fit.
8. **Per-feature scoping.** For each locked feature: granular in/out
   boundaries. This is where v1 honesty lives.
9. **Milestones.** Propose a dependency-ordered milestone sequence, each with
   a one-line scope. Confirm.

## Output artifacts (phase 10 — write files)

- **`PRD.md`** at the repo root: purpose, features, out-of-scope, stack,
  integrations, data model, per-feature scope, milestone overview. Markdown
  only. Never write it anywhere else or under another name — a nested
  `docs/PRD.md`, a slugged filename, or a report-style name are all wrong.
- **`milestones/N-<slug>/prompt.md`** for each milestone: a self-contained
  prompt that a coding agent can execute cold — context, scope, constraints
  from the PRD, and done-when criteria. A reader must not need the PRD open
  to act on it.

Offer the natural next step: run `cut-it` on a milestone prompt to slice it
into a sprint.

## Red flags (you are doing it wrong)

- Asking about the stack when package.json answers it.
- A phase "locked" without the user confirming it, and — in a headless run —
  a phase left unconfirmed instead of run as the assumed-confirm monologue.
- Out-of-scope list that is empty or generic ("no mobile app").
- Milestone prompts that say "see PRD for details".
- An existing entity or module re-specced as new work instead of reused.
- Reaching for `bash` to grep or verify what was written: `bash` is not in
  this skill's tool surface and the call is refused. Use `grep`/`read`/`find`.
- Opening a task list for the nine phases; `tasks` is refused.
