# Evals — architecture

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`).

## S1 — brownfield feature on an existing repo
Setup: repo with an existing codebase and a short brief. Prompt: "figure
out the architecture for adding X."
Expected:
- Reads the relevant existing surfaces before proposing.
- 2-3 genuinely different approaches with trade-offs; recommendation with
  reasoning; user makes the calls via ask_user.
- One-way-door decisions get spikes with decision rules.
- Doc written with the required shape; skipped sections noted.

## S2 — greenfield with unfamiliar stack temptation
Setup: user knows Python; the "modern" answer is a stack they don't know.
Expected:
- Familiarity weighed explicitly; the recommendation reflects the user's
  goals, not fashion.

## S3 — altitude check
Setup: mid-session the user asks "just list the files to change for this".
Expected:
- Pulls back up to decisions; hands task-level planning to cut-it.

## Baseline failure modes to watch for (RED)
- One-shot architecture doc with silent convergence.
- No alternatives, no trade-offs, no spikes.
- Implementation task lists in the decision doc.
