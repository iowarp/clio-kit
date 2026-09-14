---
name: design-council
description: 'Use when this workflow is requested: Convenes several expert perspectives
  that challenge each other on a design decision with real trade-offs before code
  is written; quick mode runs a single round. Not for a one-question-at-a-time interrogation
  of a plan; use grill-me. Not for splitting implementation across workers; use dispatch
  directly. Triggers on "convene a design council", "debate this design".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/workflow/design-council
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Design Council

Run a bounded multi-perspective debate on a real design decision. The council
surfaces the crux of a disagreement before code commits to one side. It is not
a ritual: if experts would agree, do not convene it.

## Arguments

```text
convene a design council on <decision>
```

There is no flag syntax; the trigger is conversational — "convene a design
council on X", "debate this design", "get multiple expert perspectives".
Whatever the user names is the decision. A referenced file, doc, or repo path
in the same request is Step 0/1's grounding to read first, not a separate
argument.

**Headless is the enforced default, not a suggestion.** A council is several
worker runs; nobody is waiting between rounds in a headless run, and every
round that actually runs costs real wall-clock time on top of the
orchestrator's own turns. `ask_user` is not registered in a headless run,
so the single question Step 1 asks is refused as an unregistered tool
rather than answered. Treat that refusal (or
skip the call and reason from this paragraph directly — one is not more valid
than the other) as the fixed answer **quick mode: exactly three perspectives,
exactly one round (Positions) plus synthesis.** Do not compose four or five
perspectives headlessly and do not run a Responses or Convergence round
headlessly, no matter how contested the topic looks — four/five perspectives
and multi-round debate are for a live session with an operator who actually
asked for the deeper pass. This is the one rule the skill's own prior smoke
history says a model will not reliably self-infer from prose alone, so treat
the number 3 and the number 1 as hard, not as defaults to raise if the topic
seems to deserve more.

**Dispatch call shape.** Compose the round's perspectives, then make exactly
one `dispatch` call with all of them in `tasks` and `mode="parallel"`. If
that call comes back admission-denied for endpoint or target capacity (a
single local model instance commonly allows only one concurrent worker, so a
3-task parallel wave can be denied outright rather than queued), retry the
identical `tasks` batch in one dispatch call with `mode="sequential"` instead
— the tool runs them one after another itself. Never split a round into
several separate one-task `dispatch` calls made one at a time waiting on each
result before deciding the next; that is the serial-perspectives pattern that
produced the round-trip cost the skill's own timeout history is about, and it
does not fix the capacity problem the parallel call already reported. Do not
call `dispatch(list:true)` to probe capacity first — it answers nothing about
concurrency and only spends a call.

Declare `intent: {read_roots: [...], relevant_paths: [...]}` with paths
relative to the repo root on every dispatch call instead of pasting an
absolute path into `task`/`briefing` prose (a config value, a mount point).
An absolute path token in briefing/task text with no declared `intent` is
rejected as `legacy_scope_path_absolute`.

A dispatch call's own synchronous result already carries every worker's
output — do not follow it with a `bash`/`read` pass over the receipt file on
disk to re-read what you already have. The steps below are the plan; do not
open a `tasks` list for them. This skill's tool surface is exactly
`dispatch`, `read`, `grep`, `find`, `ls`, `context`, `code_nav`, and
`ask_user` (`context` and `ask_user` are always available regardless).
`tasks` and `bash` both sit outside it and any call to either is refused —
locate files with `find`/`ls`, not `bash find`/`bash ls`; inspect a receipt
with `read`, not `bash cat`. This skill never writes: `write` and `artifact`
are not on its surface, so the synthesis in Step 4 is chat output, never a
file.

## Step 0 — Check the question is contested

Before composing anyone, ask: would credible experts actually disagree on the
answer? If every perspective you can imagine picks the same option and differs
only in caveats, stop here — do not dispatch anything. Say the council is not
needed, give the consensus answer with the caveats attached, and end. Do not
dispatch a round "just to confirm" a consensus call you already reached —
that is the ritual this step exists to skip, and it still costs the same
wall-clock time and worker slots as a real debate. Trust this self-check the
same way you trust the rest of your own reasoning; a council you convened to
double-check yourself is not more rigorous than the judgment behind it.

## Step 1 — Compose perspectives

Derive perspectives from the topic itself, never from a generic role menu.
On a genuinely contested topic, call `ask_user` once with `mode:
"single_question"` asking whether this should be a quick pass (three
perspectives, one round) or the full debate (up to five perspectives, up to
three rounds). See Arguments for what a headless run does with that call.
In a live session where the user answers, honor the requested depth. Compose
three perspectives by default; go to four or five only in that live full-
debate case, and only when the decision genuinely has that many independent
stances.

Each perspective gets:

- a name and a stance (what it argues for);
- the expertise it argues from;
- the specific thing it must attack in the other positions.

Example, "HDF5 vs Zarr for checkpoints": an HPC I/O veteran defending
single-file HDF5 on parallel filesystems; a cloud-native engineer arguing
object-store-first Zarr; an operator worried about tooling and recovery; a
numerics lead demanding bit-exact round-trips. Never "optimist, pessimist,
pragmatist".

## Step 2 — Dispatch each perspective as a read-only worker

The recipe fixes capability; your task prompt supplies the persona. Pick per
perspective from the read-only recipes in the live catalog:

- `scout`: stance grounded in this repository's code.
- `researcher`: stance leaning on external docs, standards, or papers.
- `provenance`: stance arguing from runtime evidence and receipts.

See Arguments for the exact call shape (one batched `tasks` call, the
`mode="sequential"` capacity fallback, `intent` for scope, no literal shell
syntax). Rounds are sequential; a round's own perspectives are the one
dispatch call. Each task prompt carries the persona block, the decision
context, and the full transcript so far. Workers never edit files; the
debate is analysis only. Dispatch receipts link every statement to a worker
run.

## Step 3 — Run the rounds

1. **Positions.** Each perspective states its position, its strongest
   argument, and what evidence would change its mind.
2. **Responses.** Each perspective receives the round 1 transcript and must
   respond to named points from the others: concede, rebut, or sharpen.
3. **Convergence.** Each perspective states what it now agrees with, where it
   still disagrees and why that crux is the crux, and its final
   recommendation.

**Quick mode** (headless default, or a user asking for a light pass): round 1
plus synthesis. No responses round, no convergence round.

**Early termination** (full-debate mode only). After round 1, judge
disagreement on the decision question itself, not on side conditions. If
every position picks the same option and differs only in caveats, toggles,
or requests to measure later, that is consensus: skip rounds 2 and 3, report
that the council was not needed, and return the consensus with caveats.
Never manufacture friction.

## Step 4 — Synthesize

After the last round, you (the orchestrator) write:

```markdown
## Council Synthesis - <decision>

Agreements:
- <point> (all perspectives, round <n>)

Live disagreements:
- <point> - crux: <the fact or value judgment that would settle it>

Recommendation:
- <choice> because <reasoning grounded in the transcript>

Dissent preserved:
- <perspective>: <the objection that survives the recommendation>
```

Cite the transcript (perspective and round) for every claim. Done when every
synthesis line has a citation and the recommendation names its crux.

## Degraded mode

If dispatch is unavailable or admission-denied for a reason other than the
capacity retry in Arguments (the tool itself is missing from the surface, or
every retry is denied), run the same rounds inline: write each perspective's
contribution yourself, sequentially, same round structure and synthesis
format. Label the output as degraded (single-model debate, no receipts).

## Boundaries

Stress-testing a plan by questioning its author one question at a time is
`grill-me`, not a council. Splitting implementation work across workers is
plain dispatch, not a council. Council workers analyze; they never build.
For a lightweight built-in alternative with no skill workflow, the TUI
ships `/council [--roster] [--rounds] [--synthesis judge|vote|none]`; use
this skill when the debate needs the full round structure and receipts.

## Red Flags

- Perspectives named "optimist" and "pessimist" (role menu, not topic).
- Composing four or five perspectives, or running a Responses/Convergence
  round, in a headless run — the enforced default is exactly three and
  exactly one, not a ceiling to raise because the topic looks deep.
- Splitting a round into several single-task `dispatch` calls issued one at a
  time instead of one batched `tasks` call (retried as `mode="sequential"`
  on a capacity denial).
- Calling `dispatch(list:true)` before dispatching the round.
- Pasting an absolute path into a dispatch `task`/`briefing` instead of a
  declared `intent`, or quoting literal shell syntax inside a persona's
  argument.
- Reaching for `bash`/`read` on a receipt file the dispatch call's own result
  already contains.
- Opening a `tasks` list for the round structure; `tasks` is refused.
- A synthesis that averages positions instead of naming the crux.
- Manufactured disagreement on a settled question.
- A worker asked to edit files as part of the debate.
- Synthesis statements that no transcript line supports.
