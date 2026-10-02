# Evals — product-intent

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`).

## S1 — greenfield idea
Prompt: "I want to build a log-triage tool for HPC operators, write the PRD."
Expected:
- Interviews in gated clusters; never answers its own questions.
- Hypothesis block has both RIGHT and WRONG conditions.
- Output lands at `docs/<slug>.prd.md`, not `PRD.md`.
- Zero engineering decisions (no stack, no schema, no libraries).
- Unknowns say "TBD — needs validation", not invented requirements.

## S2 — user declines the interview
Prompt: "skip the questions, just write it."
Expected:
- Asks only 2-3 highest-leverage questions, names what goes TBD.
- No fabricated evidence or metrics.

## S3 — solution-shaped request
Prompt: "PRD for adding a reply button."
Expected:
- Reframes to the underlying problem; the problem statement admits more
  than one solution.

## Baseline failure modes to watch for (RED)
- One-shot PRD generated from thin air.
- Hypothesis with no WRONG condition.
- "React + Postgres" appearing anywhere in the document.
