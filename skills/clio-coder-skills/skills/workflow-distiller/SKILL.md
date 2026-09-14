---
name: workflow-distiller
description: 'Use when this workflow is requested: Packages a workflow that just happened
  into a reusable SKILL.md: reconstructs it from the session record, checks overlap
  with installed skills, gates on approval, then writes it following skill-craft.
  Not for authoring a skill with no prior workflow; use skill-craft. Triggers on "make
  this workflow a skill", "package what we just did".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/workflow/workflow-distiller
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Workflow Distiller

Turn a workflow that actually ran into a reusable skill. The distiller's
identity is runtime truth: it reconstructs what happened from the visible
session record, not from anyone's memory of it. Skip the ceremony when there
is no prior workflow; a from-scratch skill is a SKILL.md written directly,
following skill-craft. skill-craft also governs how every skill this
distiller produces is written: description, body, and pruning rules live
there, not here.

## Arguments

```text
/skill workflow-distiller [<what to distill>]
```

With arguments, the text names the workflow to distill; without, distill
whichever workflow just ran in this session. Everything else in the request
(the conversation so far) is the session record Phase 1 reconstructs from,
not more arguments.

A live operator can answer Phase 2's interview and Phase 4's design gate for
real - treat a real, answered `ask_user` reply as proof an operator is
present and continue asking one question per round. There is no operator in
a headless run: `ask_user` is not registered, so any call is refused as an
unregistered tool rather than answered. When that
is the run's condition, do not wait for a reply that cannot come: answer
every remaining interview question and the Phase 4 gate yourself as an
assumed-confirm monologue (state the question, give your best-grounded
answer or design choice, mark it `assumed - confirm`) and proceed straight
through to Phase 5, the same way as a live "looks fine, write it." A skill
file is a plain, reversible artifact under version control, not an
irreversible external action - write it and say so in the final reply,
rather than stopping at the gate with nothing produced.

`tasks` sits outside this skill's tool surface (`read`, `grep`, `find`,
`ls`, `context`, `write`, `ask_user`) and any call is refused; the six
phases above are the plan, not a task list. This skill never has `bash` or
`git`, so it never commits or pushes the file it writes - say so in the
final reply and let the user commit it.

## Phase 1 - Reconstruct From Evidence

Before asking anything, list the concrete steps that visibly executed in this
session, in order: tools called, commands run, files touched, dispatches made,
skills activated. Cite the conversation record for each step. Anything the
user describes that did not visibly run is tagged `assumption` and becomes an
interview question. This grounding step is mandatory and comes first; when the
reconstruction and the user's memory disagree, the reconstruction wins and the
disagreement is worth a question.

## Phase 2 - Interview

Use `ask_user` with `mode: "single_question"`, one question per round, bounded
rounds (default 8, max 12), following the grill-me operating contract:
recommended answer first when options are natural, stop signals ("stop",
"enough", "later", cancel) respected immediately, answers translated into
compact decisions. Question bank, roots before leaves:

1. Confirm the reconstruction and resolve tagged assumptions.
2. Inputs and outputs: what varies between runs, what is fixed.
3. Recurrence: how often has this actually happened? A workflow that ran once
   is usually not worth a skill; ask early and offer to stop.
4. Per-step rigidity: must-match versus any-reasonable-approach.
5. Failure behavior per step: ask the user, retry an alternative, or fail
   loudly.
6. Scope cuts: what the skill should refuse to do.
7. Name (lowercase-hyphen).

## Phase 3 - Overlap Check

Call `context(scope="skills")` with no name to list installed skills. Judge
overlap per step, not per workflow: if an installed skill's triggers cover a
step, that step is referenced by name in the new skill body, never
reimplemented, with a one-line rationale for the dependency. A different end
goal does not excuse reimplementing a covered step. Record each reference as
a `requires: [skill:<name>]` entry in the generated skill's frontmatter in
Phase 5, so the loader warns when the dependency is missing.

## Phase 4 - Design Gate

Present a compact design summary and wait for explicit approval:

```markdown
## Skill Design - <name>
Description draft: <triggers only, third person>
Steps: <step, rigidity, on-failure> per line
References: <skill:<name>, why> or none
Scope: project | user (user only if the workflow crosses repositories)
Validation scenario: <prompt, expected observable behavior>
```

No skill file is written before the user approves. "Looks fine, but change X"
means revise and re-present. In a headless run with no operator (see
Arguments), treat the design as approved once it is internally consistent
with Phase 2's decisions, mark it `assumed - confirm` in your final reply,
and proceed to Phase 5 - do not stop the run with a presented-but-unwritten
design.

## Phase 5 - Create

Write `SKILL.md` under `draft-skills/<approved-name>/`, outside protected active
skill and installed package roots, with the
frontmatter contract below - a triggers-only third-person description, and
the pruning pass - with `requires: [skill:<name>]` for every skill the
overlap check referenced. Do not try to load skill-craft's own file mid-run
to check this: only one skill can be active at a time, and a second
`context(scope="skills", name="skill-craft")` call is refused while this
skill is still pending. The frontmatter contract, current as of this
writing: `name`, `description` (third-person triggers, no "I"/"you"),
`triggers` (a non-empty list), `version` (start `0.1.0`), `license`,
`allowed-tools` (canonical lowercase Clio tool names only - `bash`/`git`
capitalized or spelled differently is rejected), and `requires` when Phase 3
found a reference. If in doubt about the exact shape, `read` an already-
installed skill's `SKILL.md` (this skill's own file is always available) and
mirror its frontmatter keys rather than guessing or trying to load
skill-craft. The proposed installation scope defaults to project; suggest user
scope only when the user said the workflow crosses repositories. Creation here
produces a draft. The operator later reviews a portable package envelope and
installs through the Library; never write into an active skill store directly. Placeholders replace every
session-specific path, name, and value; distill the pattern, not the
incident. Keep the generated skill under 120 lines; reference instead of
inlining. If the session repeatedly dispatched the same worker pattern, also
offer a recipe sketch for the agents surface, but do not write recipe files.

## Phase 6 - Validate

Record one RED-GREEN scenario agreed with the user: the prompt, and the
observable behavior that distinguishes with-skill from without. `clio-coder
library validate draft-skills/<approved-name>/SKILL.md` needs `bash`, which is outside this skill's tool surface -
never attempt it; instead `read` the file back and confirm the frontmatter
contract from Phase 5 by eye (required keys present, `allowed-tools` entries
canonical lowercase, under the line budget), and say plainly that a real
`clio-coder library validate` pass is still owed once bash is available.
Otherwise record the scenario in the skill body's example section as the
standing validation obligation. This skill has no `bash`/`git`, so it never
commits or pushes the file it writes; say so and let the user commit it.

## Worked Example

Session: the user fetched a CSV export, normalized column names with a small
script, and verified row counts against the source, three sessions in a row.

1. Reconstruct: `bash curl ...` (ran), `write normalize.py` + `bash python3
   normalize.py` (ran), `bash wc -l` comparison (ran). User also mentions "and
   I always spot-check five rows" - not visible this session, tagged
   assumption.
2. Interview: confirms reconstruction; spot-check confirmed as a real step;
   recurrence "weekly"; normalization must-match, fetch any-approach; failure
   on row-count mismatch must fail loudly; name `csv-ingest`.
3. Overlap: the `context(scope="skills")` listing shows no fetch or CSV skill
   installed; no references.
4. Gate: summary presented; user approves after tightening the description.
5. Create: write `draft-skills/csv-ingest/SKILL.md`, placeholders for the
   export URL and column map; frontmatter contract confirmed by reading the
   file back (no `bash`, so no `clio-coder library validate` this turn).
6. Validate: scenario "ingest this month's export" must show fetch,
   normalize, count-verify, spot-check in that order; recorded in the body.

## Red Flags

- Writing any skill before the design gate is approved by a live operator,
  or before a headless run has marked it `assumed - confirm`.
- Stopping the run at Phase 4 with a presented-but-unwritten design when the
  run is headless - that's the backlog pattern (guard an irreversible
  action), and a written skill file is not that; it is reversible.
- A reconstruction that lists steps nothing in the session shows.
- Reimplementing an installed skill's job instead of referencing it.
- Session-specific paths or values surviving into the generated skill.
- Batching interview questions or ignoring a stop signal.
- Distilling a one-off without asking about recurrence.
- Calling `bash` for `clio-coder library validate`, or `context` with a
  second skill's name to consult it mid-run: both are outside this skill's
  tool surface and refused.
