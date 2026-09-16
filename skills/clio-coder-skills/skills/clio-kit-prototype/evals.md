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

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Verdict captured via terminal artifact; code discarded.

## Battletest record (2026-09-03)

S1 fixture, `ornith1.5-35b-moe` on mini (llamacpp), `clio-coder run --autonomy full-auto --json`, headless.

| run | wall | turns | in / out tokens | outcome |
|---|---|---|---|---|
| baseline (no skill) | 207s | 17 | 6.1k / 11.6k | HTML built, verdict written via terminal `artifact`; 7 `tasks` calls; prototype left untracked on main |
| v0.3.0 | 220s | 10 | 12.3k / 15.8k | logic branch chosen, LOGIC.md read; `edit` blocked by allowed-tools so the 12.5k-char file was rewritten whole; `artifact` ended the run before Step 3; nothing committed |
| v0.4.0 | 178s | 13 | 6.7k / 10.8k | cases enumerated first; module driven headlessly via `node -e`; branch `prototype/retry-state-machine` created, prototype committed there, `main` clean; reply carries question, verdict, evidence, recommended change, pointer |

Changes in v0.4.0 that closed the gaps: dropped `artifact` (terminal tool,
`terminate: true`, ends the run before capture-and-discard), added `edit`,
added an `## Arguments` contract with a headless fallback, made Step 3 an
explicit sequence of single-command `bash` calls, banned `$(...)` (net ask
rail even under full-auto), and made the final reply the verdict record.
Remaining blocked calls in v0.4.0: one `tasks` plan and one read-only
`git` status; `git` restored to allowed-tools and a one-line "the steps are
the plan" note added afterwards.
