---
name: cut-it
description: 'Use when this workflow is requested: Slices an existing plan, PRD, or
  milestone into an executable sprint of dependency-ordered vertical slices sized
  for one agent run each, with done-when verification per slice; never fabricates
  a plan. Not for deciding the approach; use architecture. Triggers on "cut it", "slice
  this plan".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/workflow/cut-it
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Cut It

Transform an existing plan into ordered execution slices that a coding agent
can run one at a time, leaving the build green after every slice. The output
is a `SPRINT.md` another agent can execute cold — no conversation context
required.

## Arguments

```text
cut it [<path to plan>]
```

There is no flag syntax; the trigger is conversational — "cut it", "slice
this plan", "turn this milestone into a sprint". A path the user names in
the same request (a specific `PLAN.md`, `PRD.md`, or `milestones/*/prompt.md`)
is the plan to slice; a path with no plan words near it, or a bare
destination like "write it to docs/SPRINT.md", is Step 3's output location,
not the input. When neither is named, Step 1 finds the plan and Step 3
writes to the repo-root default.

This run has no back-and-forth. `ask_user` is not registered in a headless
run, so any call is refused as an unregistered tool rather than answered.
The happy path here rarely needs a question at all — Step 1's "stop and say
so" for a missing or vague plan is already headless-safe, and Step 3's output
path defaults without asking. If several plans or milestones are plausible
candidates and the choice matters, do not stop on an open question: pick the
most recently modified or most specifically named one, state that choice and
the alternative you set aside, mark it `assumed — confirm`, and keep going in
the same turn. Do not end a turn on an unanswered question in `ask_user` or
in plain text.

The steps below are the plan; do not open a task list for them. This skill's
tool surface is exactly `read`, `grep`, `ls`, `find`, `git`, `context`,
`code_nav`, `write`, and `ask_user` (`context` and `ask_user` are always
available regardless). `tasks` and `bash` both sit outside it and any call to
either is refused — track progress by walking the steps below, not a task
board; check for a build/test/lint setup (a `package.json`, a Makefile, a
node/toolchain version) with `find`, `ls`, and `read`, not `bash node
--version` or `bash find`. The read-only `git` tool (`status`, `log`) is
useful in Step 1 when locating the plan benefits from recent history or
uncommitted changes.

## Step 1 — Locate the plan

In priority order: a file the user names, a plan in the conversation,
`PLAN.md` / `PRD.md` / `milestones/*/prompt.md` in the repo. **Never fabricate
the plan.** If none exists, or what exists is too vague to slice honestly,
stop and say so — recommend `grill-me` to resolve intent first. Artificial
slicing of a vague plan hides gaps; flagging them is the deliverable.

## Step 2 — Apply the cutting rules

- **Vertical slices.** Each slice delivers end-to-end behavior, however thin.
  No "all the types first, all the wiring later" horizontal layers.
- **Dependency order.** A slice may depend only on earlier slices. State the
  dependency explicitly.
- **One agent run each.** Sized so a focused agent completes it in a single
  run: small enough to hold in context, large enough to be meaningful.
- **Green after every slice.** Build, lint, and existing tests pass at every
  cut point. A slice that leaves the tree broken is two slices cut wrong.
- **Self-contained.** Real file paths, real commands, concrete steps. A reader
  with zero conversation context can execute it.

## Step 3 — Write SPRINT.md

Use the `write` tool. `SPRINT.md` is a plain file in the working tree, not a
generated report — do not call a tool literally named `artifact` for this;
that tool is not on this skill's surface and, in this harness, is a
terminal call that ends the run the instant it is invoked, before you can
report back. Default output path is `SPRINT.md` at the repo root; honor a
caller-supplied path from Arguments instead. Format:

```markdown
# Sprint: <name>

## Battle order
1. <slice 1 title>
2. <slice 2 title>  (depends on: 1)
...

## Slice 1 — <title>
**Goal**: <the behavior this delivers>
**Depends on**: <slice numbers or "nothing">
**Files**: <paths touched>
**Steps**:
1. <concrete step with real paths/commands>
**Done when**: <observable, testable criteria — the verification anchor>
**Out of scope**: <what this slice deliberately does not do>
```

"Done when" is the contract, not decoration. If you cannot write a testable
done-when for a slice, the slice is not ready to cut — go back to the plan.

## Step 4 — Report back

End the turn with a short final reply, not silence after the write: the path
you wrote (`SPRINT.md` or the caller-supplied path), the number of slices,
and a one-line summary of the battle order. This is the only confirmation
the caller gets that the write actually happened.

## Red flags (you are doing it wrong)

- A slice whose steps say "and related changes" or "etc."
- Done-when criteria that restate the goal instead of naming a check.
- A slice that only compiles when a later slice lands.
- Slicing a plan you had to invent on the spot.
- Calling a tool literally named `artifact` because Step 3 talks about "the
  artifact" — that word here means "the deliverable document," not the
  `artifact` tool. That tool is off this skill's surface and, in this
  harness, terminates the run on the spot, writes to
  `.clio-coder/artifacts/` instead of `SPRINT.md`, and skips Step 4 entirely.
  Use `write`.
- Ending the run right after the write with no final reply — Step 4 is not
  optional.
- Opening a `tasks` list for the steps above; `tasks` is refused.
- Reaching for `bash` (`node --version`, `find`, `ls -la`, ...) to check the
  repo's toolchain or structure; `bash` is refused, use `find`/`ls`/`read`.
- Stopping to wait for an `ask_user` reply that headless runs never send;
  see Arguments.
