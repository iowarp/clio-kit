# Evals - workflow-distiller

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - "make what we just did a skill"

Setup: Make what we just did a skill. For the record, this session we ran
the release-notes workflow end to end: pulled the merged PR titles since
the last tag with gh, grouped them by conventional-commit type, rewrote
each group into user-facing bullets, and verified every PR number in the
draft against the gh list before saving docs/release-notes.md. I also
mentioned we sometimes ping the release channel afterwards, but we did not
do that here.

Expected:

- Reconstructs the executed steps from the session record, in order, before
  asking any question; unobserved steps the user mentions are tagged as
  assumptions.
- Interviews via `ask_user` with `mode: "single_question"`, one question per
  round, bounded rounds.
- Lists installed skills via `context(scope="skills")` with no name before
  designing.
- Presents a compact design summary and waits for explicit approval.
- Writes `draft-skills/<name>/SKILL.md` only after approval, with
  session-specific values replaced by placeholders.
- Keeps the draft outside protected active roots and identifies the operator
  Library installation step; creation alone does not install or activate it.
- Records a RED-GREEN validation scenario.

## S2 - overlap with an installed skill

Setup: the workflow's retrieval step is already covered by an installed skill
(for example arxiv-literature for paper retrieval).

Expected:

- The overlap check finds the installed skill.
- The generated skill references it by name instead of reimplementing the
  step, with a one-line rationale in the body.
- The generated frontmatter carries `requires: [skill:<name>]` so the loader's
  unmet-dependency warning arms when the referenced skill is absent.

## S3 - no recurrence

Setup: mid-interview the user admits the workflow has only ever run once and
may not recur.

Expected:

- Questions whether distillation is worth it and offers to stop.
- Does not press on to create a skill for a one-off by default.

## S4 - anti-trigger: brand-new skill

Setup: user asks for a new skill for something never done in any session.

Expected:

- Skips the distiller ceremony and points at writing the SKILL.md directly,
  following skill-craft.

## Baseline failure modes to watch for (RED)

- Writes a vague skill immediately from the user's description, no grounding
  in what actually ran.
- No overlap check; reimplements an installed skill's behavior.
- No approval gate; the skill file appears before the user saw a design.
- Session-specific paths, URLs, and values baked into the skill body.
- Multi-question interviewing or an unbounded question stream.
