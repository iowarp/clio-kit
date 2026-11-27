---
name: clio-kit-grill-me
description: 'Use when this workflow is requested: Stress-tests a plan, design, or
  idea through a phased one-question-at-a-time interview before code is written, ending
  in a compact decision log. Not for a multi-perspective debate; use design-council.
  Triggers on "grill me", "interview me about this plan".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/workflow/grill-me
  upstream-name: grill-me
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Grill Me

Run a rigorous, repo-aware interview that turns a vague plan into explicit
decisions. The point is not to interrogate for sport; it is to surface hidden
branches before anyone writes code.

## Arguments

```text
grill me on <plan, feature, or idea>
```

There is no flag syntax; the trigger is conversational — "grill me on X",
"stress-test this design", "poke holes in this idea". Whatever the user
names is the subject. A referenced file, doc, or repo path in the same
request is Step 1's grounding to read first, not a separate argument.

**This run has no back-and-forth.** There is no second turn in which a user
reads your question and replies to it — whatever you ask, you must also
answer yourself, in this same turn, before it ends. Do not write a question
and stop to wait for a reply, in `ask_user` or in plain chat text; nothing
is coming. Ending the turn on an open question — even one question, even a
well-posed one — is this skill's single most common failure and worse than
skipping the interview format entirely.

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered, every
time, with no exceptions. Calling it again will not produce a different
result, so one call is enough to confirm it (not required: reasoning from
this paragraph alone is just as valid as calling it and observing the
refusal). Whichever phase this lands in — even round
1 — switch immediately to the assumed-confirm monologue for every phase
from here on, in the same turn: state the question you would have asked,
give your own best/recommended answer with the reasoning behind it, mark
it `assumed — confirm`, and move to the next phase. Do not re-call
`ask_user` hoping a later round behaves differently — that only burns the
`max_rounds` budget without ever converging. Keep working the phase map,
phase by phase, all the way through Step 5's decision log before ending
the turn — never end on "Answer 1/2/3" or any other place a reply is
expected.

The phase map below is the plan; do not open a task list for it. This
skill's tool surface is exactly `read`, `grep`, `ls`, `find`, `git`,
`context`, `code_nav`, and `ask_user` (`context` and `ask_user` are always
available regardless). `tasks` and `bash` both sit outside it and any call
to either is refused — inspect a file with `read`, not `bash cat`/`bash
head`/`bash wc`; locate files with `find` or `ls`, not `bash find`/`bash
ls`; check repo state with the `git` tool, not shell `git`.

## Operating Contract

- In a live session, use `ask_user` for the interview and actually wait for
  the user's answer between rounds. In a headless run, see Arguments above.
- For every interview round, call `ask_user` with `mode: "single_question"` and
  exactly one question.
- On the first ask for a normal grill-me run, set `max_rounds` to a bounded
  value, usually `12` and at most `16` unless the user explicitly asked for a
  very deep interview.
- Put your recommended answer first when options are natural. Include 2-3 real
  alternatives with short tradeoff descriptions.
- The user answers in natural language. You translate answers into compact
  decision keys and rationale when you call `ask_user` with `action: "complete"`.

## Phase Map

Walk phases in order unless the user names a narrower target. Do not skip the
scan. A phase can be "review" when context already contains a plausible answer
or "fill" when the decision is genuinely missing.

| Phase | Scope | Default mode |
|---|---|---|
| 0 | Scan available context: user prompt, named files, repo structure, git state, existing plans/specs | No questions unless the subject is missing |
| 1 | Frame: user, problem, outcome, non-goals, success criteria | Fill or review |
| 2 | Current state: existing code, constraints, conventions, integration points, prior attempts | Review |
| 3 | Shape: data model, API/UX surface, ownership boundaries, naming, compatibility | Fill |
| 4 | Risk: failure modes, migrations, rollout, test strategy, observability, reversibility | Fill |
| 5 | Delivery: first slice, done-when checks, deferrals, handoff target (`prd`, `cut-it`, or direct implementation) | Review then complete |

## Workflow

### Step 1 - Scan

Read what the user already gave you. If the task references files, plans, code,
tests, or project conventions, inspect them before asking. Prefer
`context(scope="workspace")`, `grep`, `read`, `code_nav` (symbol and call-graph
lookups), and codewiki tools over guessing. Use the `git` tool (`status`,
`log`, not shell `git`) when the plan references repo state — recent
history, uncommitted changes, what "decided" actually means for this repo
right now.

Privately build a phase map:

- known facts
- assumptions worth challenging
- missing decisions
- dependencies between decisions
- likely deferrals and why they might be safe

Never spend the user's attention on facts that the repo answers. If you found
the answer in code, summarize it briefly and ask only whether it should remain
true.

### Step 2 - Choose Review Or Fill

For each phase:

- **Review mode**: context already has an answer. Present the finding in one or
  two sentences, then ask a targeted question such as "Is this still accurate?"
  or "Should we keep this constraint?"
- **Fill mode**: context is sparse or ambiguous. Ask the highest-leverage
  missing decision first.

Always resolve root decisions before leaves, in the Question Priority order
below.

### Step 3 - Ask One Question

Each `ask_user` round contains one question only:

- one stable `header`
- one concrete question
- recommended option first when options fit
- no multi-part wording hidden inside the question

Bad: "Who is this for, what should v1 include, and how should we test it?"

Good: "Which user should v1 optimize for first?"

If an answer is vague, ask a follow-up on the same branch. Do not jump to a new
branch while the current one is still unresolved.

**Headless: there is no reply coming, whether `ask_user` comes back
refused as unregistered or you never call it at all.** Neither is a vague answer to
follow up on and neither is a signal to try again or to wait — both mean
there is no operator this run, from round 1 on. Do not call `ask_user`
again for this or any later phase, and do not phrase a question in plain
text as if a reply is pending. From here, run every remaining phase
(including this one) as the assumed-confirm monologue described in
Arguments, in this same turn, through to the Step 5 decision log. See
Arguments for the exact treatment.

### Step 4 - Respect Stop Signals

This step applies to a live session with a real operator. Stop immediately
when the user says "stop", "enough", "later", "done", "next time", or cancels
the modal. Do not ask another question to confirm stopping.

A headless refusal of `ask_user` is not a stop signal from a user — it is the
absence of an operator (see Step 3). Do not treat it as "the user cancelled
this session"; treat it as the cue to switch to the assumed-confirm
monologue and keep going to a complete decision log, not to stop early.

If you have enough decisions to be useful, call:

```json
{
  "action": "complete",
  "summary": "Short interview closeout.",
  "decisions": [
    {
      "key": "primary_outcome",
      "value": "Smallest useful slice",
      "rationale": "The user prioritized fast validation over broad architecture.",
      "confidence": "high",
      "source_question": "What should this plan optimize for first?"
    }
  ]
}
```

Then provide the decision log. If the stop happened before enough context,
state the partial decisions and the next unresolved root question.

### Step 5 - Complete

In a live session, close with `ask_user` `action: "complete"` and a compact
`decisions` array before final prose. In a headless run where `ask_user`
already came back refused, skip straight to the decision log below — do
not attempt another `ask_user` call just to close out; it will be refused
too and adds nothing. Write the final decision log:

```markdown
## Decision Log - <topic>
1. <decision> - chosen over <alternative> because <reason>
2. ...

Deferred:
- <item> - safe because <reason>

Open risks:
- <risk or unresolved branch>

Recommended next step:
- <prd | cut-it | direct implementation> - <why>
```

## Question Priority

Use this ordering when several questions are possible:

1. User and problem being solved.
2. Primary success measure.
3. Explicit non-goals.
4. Existing constraints from repo or environment.
5. Data/API/UX boundary.
6. Failure modes and recovery.
7. Tests and done-when checks.
8. First implementation slice.
9. Naming and polish.

## Decision Rules

- If the user says "whatever you think", record your recommendation as the
  decision and say so.
- If two choices are both viable, choose the one that reduces irreversible
  work unless the user explicitly values speed or breadth more.
- If a decision is safe to defer, record why and what later signal will force
  it.
- If the plan is too vague to slice or implement, say that clearly and continue
  interviewing instead of fabricating certainty.

## Red Flags

- Asking multiple questions in one `ask_user` round.
- Asking about facts discoverable from the repo.
- Letting `ask_user` hit the round limit without completing the interview.
- Ending with a summary paragraph instead of the decision log.
- Re-calling `ask_user` for a later phase after an earlier round already came
  back refused — the answer will not be different; that budget is wasted.
- Ending a turn on "Answer 1/2/3", "let me know which you prefer", or any
  other wording that expects a reply in a headless run — there is no next
  turn for a reply to land in. This is the single most common failure mode
  of this skill and the one to watch hardest for: asking one question, then
  stopping, instead of running the assumed-confirm monologue through every
  remaining phase to the decision log in the same turn.
- Opening a `tasks` list for the phase map; `tasks` is refused.
- Treating a headless `ask_user` refusal as the user's stop signal (Step 4)
  instead of the absence-of-operator cue it actually is.
