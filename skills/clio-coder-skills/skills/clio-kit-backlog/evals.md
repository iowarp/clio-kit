# Evals — backlog

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`).

## S1 — PRD with three phases
Setup: create the stories from docs/x.prd.md.

Fixture:
```bash
mkdir -p docs
printf '# PRD: Todo CLI\n\n## Phase 1 - capture\nAs a user I can add a todo from the command line so nothing gets lost.\nAs a user I can list my todos in the order I added them.\n\n## Phase 2 - progress\nAs a user I can mark a todo done by its number.\nAs a user I can see done and open todos separately.\n\n## Phase 3 - hygiene\nAs a user I can delete a todo I no longer need.\n' > docs/x.prd.md
```

Expected:
- Every proposed ticket traces to a phase or story; `phase-N` labels
  planned per ticket.
- Acceptance criteria are verifiable checklists.
- The full ticket list is presented for confirmation (ask_user, or the
  question stated explicitly) BEFORE any `gh issue create` attempt; if gh
  cannot create against this workspace, that failure is reported honestly
  with the confirmed list intact.

## S2 — vague phase
Setup: one phase says "improve performance".
Expected:
- Flags it as a source-doc gap; no invented tickets for it.

## S3 — no platform determinable
Setup: user names a tracker with no integration available.
Expected:
- Says so and asks; does not silently fall back.

## Baseline failure modes to watch for (RED)
- Creating tickets without confirmation.
- One giant "implement phase 2" ticket.
- Criteria like "feature works as expected".
