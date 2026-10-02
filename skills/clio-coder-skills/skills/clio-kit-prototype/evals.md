# Evals — prototype

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`).

## S1 — state-model sanity check
Setup: sanity-check whether the retry state machine in retry.js feels
right before we build on it.

Fixture:
```bash
printf 'const STATES = ["idle", "trying", "backoff", "failed", "done"];\n\nfunction next(state, event) {\n  if (state === "idle" && event === "start") return "trying";\n  if (state === "trying" && event === "ok") return "done";\n  if (state === "trying" && event === "err") return "backoff";\n  if (state === "backoff" && event === "timer") return "trying";\n  if (state === "backoff" && event === "giveup") return "failed";\n  return state;\n}\n\nmodule.exports = { STATES, next };\n' > retry.js
```

Expected:
- Question stated first; logic branch chosen; references/LOGIC.md read.
- Single HTML artifact with free-play plus guided walkthrough cases.
- Full state surfaced after each action; no persistence.
- Verdict captured; prototype kept off the main branch.

## S2 — UI exploration
Prompt: "explore what the dashboard header could look like."
Expected:
- UI branch; several genuinely different variants on one route, switchable.
- Runs from one task-runner command.

## S3 — scope creep
Setup: mid-prototype, the user says "this looks good, add error handling."
Expected:
- Flags the transition: validated decision folds into real code through a
  normal implementation path; the prototype is not promoted.

## Baseline failure modes to watch for (RED)
- Building production-shaped code with tests and abstractions.
- No stated question; prototype answers nothing decidable.
- Prototype committed to main.
