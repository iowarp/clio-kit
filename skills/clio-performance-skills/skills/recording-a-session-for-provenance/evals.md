# Evals - recording-a-session-for-provenance

Current revision review (2026-09-10): tool names and workflow claims were checked
against the shipped server schemas and implementation. Historical records below
apply to earlier text; they are not fresh model evaluations of this revision.

Current acceptance criterion: Record only the requested decision trail and relevant artifacts, excluding credentials and unrelated private content. Verify a known record can be retrieved, report chronicle/story and retrieval path, and stop the acquired session even if recording fails.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - the handle lifecycle

Setup: Prompt: "record this session so I can look at it later."

Expected:

- `start_chronolog` precedes any `record_interaction`.
- `stop_chronolog` is called when the work ends, not left open.
- The chronicle and story are named for how they will be searched later, not
  called something generic.

## S2 - nothing is captured implicitly

Setup: `start_chronolog` has been called and several exchanges have happened
with no further ChronoLog calls. Prompt: "what have we recorded so far?"

Expected:

- The answer states that recording is explicit per interaction, so nothing was
  captured, rather than implying the session was logged automatically.

## S3 - the wrong server for the question

Setup: Prompt: "use chronolog to work out why the job was slow."

Expected:

- The answer states ChronoLog records conversation, not job execution, and
  redirects to the profiler.

## Baseline failure modes to watch for (RED)

- Calling `record_interaction` with no live handle.
- Leaving the story handle open at the end of a session.
- Assuming interactions are captured without explicit calls.
- Treating ChronoLog as a source of job performance data.

## Trigger record (2026-08-21)

Ran through `evals/trigger_eval.py`, which loads the skill plugins into the
Agent SDK with an empty `setting_sources` and only the Skill tool allowed, so
selection is measured without the operator's own configuration influencing it.

Prompt: "Record this session so I can look back at what we decided."

This skill fired, and no sibling fired alongside it. Across the suite: 20 of 20
skills selected correctly on their own prompt, and 3 control prompts outside the
kit fired nothing.

Selection is checked. Whether the skill improves the final answer, versus an
agent working without it, is still not measured.

