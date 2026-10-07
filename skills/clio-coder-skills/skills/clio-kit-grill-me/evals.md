# Evals - grill-me

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - vague feature request

Setup: repo with an existing config domain. Prompt: "grill me on adding plugin
support."

Expected:

- Scans the existing config/extension code before asking anything the repo can
  answer.
- Starts with a root decision such as outcome, target user, or scope boundary.
- Uses `ask_user` with `mode: "single_question"` and exactly one question in
  the round.
- Puts its recommended answer first when options are supplied.
- Ends by calling `ask_user` with `action: "complete"` and then writes a
  decision log, not a summary paragraph.

## S2 - answerable-from-repo question

Setup: plan mentions "the test runner". The repo's package.json defines it.

Expected:

- Does NOT ask the user which test runner is used; reads package.json instead.
- States what it found and asks only whether that constraint should remain
  true.
- Treats the area as review mode, not fill mode.

## S3 - user defers

Setup: mid-interview, user answers "whatever you think is best."

Expected:

- Records its own recommendation as the decision and says so explicitly.
- Does not silently skip the branch.
- Continues only if another root decision remains.

## S4 - long phased interview

Setup: prompt asks for a deep stress test of a large design.

Expected:

- First `ask_user` call sets a bounded `max_rounds` value such as 12 or 16.
- Still asks one question per round.
- Completes before the limit when decisions are sufficient.
- If the limit is near, closes with current decisions and open risks instead
  of continuing to ask.

## S5 - stop signal

Setup: mid-interview, user says "stop", "enough", "later", or cancels the
modal.

Expected:

- Stops immediately and does not ask a confirmation question.
- Calls `ask_user` complete when possible with partial decisions.
- Final response includes partial decisions and the next unresolved root
  question.

## Baseline failure modes to watch for (RED)

- Question batching: more than one question in an `ask_user` round.
- Interview starts cold without reading repo facts that are clearly relevant.
- Interview ends when the user gets tired, with no decision log.
- Asks about facts discoverable via grep/read.
- Hits the round limit without a useful closeout.
