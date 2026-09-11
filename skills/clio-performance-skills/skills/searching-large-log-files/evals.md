# Evals - searching-large-log-files

Current revision review (2026-09-10): tool names and workflow claims were checked
against the shipped server schemas and implementation. Historical records below
apply to earlier text; they are not fresh model evaluations of this revision.

Current acceptance criterion: Report original and derived paths, timestamp format/timezone, filters, match count and truncation. Verify the filter against a known matching line before accepting an empty result; preserve the original log.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - a log too big to read

Setup: A 400 MB job log. Prompt: "find what went wrong in this log."

Expected:

- The file is never read wholesale into context.
- Filtering happens before reading: `filter_by_log_level` or
  `apply_filter_preset` rather than a full read.
- `detect_log_patterns` is used to find the error cluster, and the answer does
  not stop at the first error line.

## S2 - already-ordered input

Setup: A log whose timestamps are already sorted. Prompt asks for errors in a
time window.

Expected:

- No sort is performed on an already-ordered file.
- `filter_by_time_range` is used to bound the window.

## S3 - handoff to analysis

Setup: Prompt: "get me these errors in a form I can chart."

Expected:

- `export_to_csv` is used rather than pasting rows into the answer.

## Baseline failure modes to watch for (RED)

- Reading the log file directly when a filter would answer the question.
- Sorting a file that is already ordered.
- Reporting the first error as the cause without checking for a cluster.
- Keyword-filtering on "error" where a level filter is meant.

## Trigger record (2026-08-21)

Ran through `evals/trigger_eval.py`, which loads the skill plugins into the
Agent SDK with an empty `setting_sources` and only the Skill tool allowed, so
selection is measured without the operator's own configuration influencing it.

Prompt: "This file is huge, what is in it? It is app.log from last night."

This skill fired, and no sibling fired alongside it. Across the suite: 20 of 20
skills selected correctly on their own prompt, and 3 control prompts outside the
kit fired nothing.

Selection is checked. Whether the skill improves the final answer, versus an
agent working without it, is still not measured.


## S4 - bracketed levels must not hide known errors

Setup: A log containing a known `[ERROR]` line.

Expected:

- The agent checks a sample before interpreting an empty filter result.
- Any normalized copy preserves the original and records the format change.
- A zero count is not reported as proof that the log contains no errors.
