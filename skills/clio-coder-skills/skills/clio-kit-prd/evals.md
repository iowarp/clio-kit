# Evals — prd

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 — brain dump to spec
Setup: Okay, brain dump. Our users keep asking for saved searches — they
build the same filter combos every day and lose them on refresh. I am
thinking they can name a search, get it in a sidebar list, maybe share a
link to it with a teammate. There is also the question of limits, do free
users get three saved searches or unlimited, and should a shared link work
for someone without an account? I have not decided. Somewhere down the
line I would also love alerts, like notify me when a saved search has new
results, but that might be its own thing. Spec this out.

Fixture:
```bash
mkdir -p src
printf '{\n  "name": "search-app",\n  "version": "1.0.0",\n  "private": true,\n  "dependencies": {\n    "typescript": "5.5.0"\n  }\n}\n' > package.json
printf 'export function runSearch(filters: string[]): string[] {\n  return filters;\n}\n' > src/search.ts
```

Expected:
- Proceeds phase by phase; each phase confirmed before the next opens.
- Detects the stack from the repo and confirms rather than asking cold.
- Produces PRD.md plus milestones/N-slug/prompt.md files.
- Each milestone prompt is executable without the PRD open (self-contained).

## S2 — scope honesty
Setup: brain dump implies ten features. Prompt: "spec this out for v1."
Expected:
- Proposes 4–8 core features and pushes the rest to an explicit out-of-scope list.
- Out-of-scope entries are specific to this product, not boilerplate.

## S3 — existing foundation
Setup: repo already implements half the data model the idea needs.
Expected:
- Inventories the existing entities in phase 5/7 and reuses them.
- Does not spec already-built functionality as new work.

## Baseline failure modes to watch for (RED)
- One giant questionnaire instead of phase-gated exchanges.
- PRD written without any user confirmation of features/scope.
- Milestone prompts that are headings with no executable content.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. WEAK PASS. Engaged the brain dump and asked the first phase-gate question; single-turn headless ends there by design, so no PRD file was produced in-run.

## Battletest record (2026-09-03)

Fixture: `/home/akougkas/eval-temp/harness/test_prd.py`, continuing the
planning category's shared HPC log-triage domain from `product-intent`.
Seeds the actual `docs/hpc-log-triage.prd.md` product-intent output, its two
evidence docs, and a partial codebase (`src/scanner.py`: a working
`FailureEvent` + `scan_oom`, OOM only — ECC/Xid not yet implemented) inside a
git repo. The brain-dump prompt names ten scope-creep features (dashboard,
Slack, always-on pipeline, auto-remediation, learned ranking, federation,
audit export, RBAC, mobile app) and one explicit one-way-door tension
(on-demand reads vs. an always-on ingestion pipeline), combining S1
(existing foundation, stack detection), S2 (scope honesty), and S3 (existing
foundation reuse) into one gradable run. Graded 14 checks against real
post-run disk state (`PRD.md` at the exact promised path, all eight required
sections, the out-of-scope section itself — not just anywhere in the
document — actually containing the pushed-out features, `FailureEvent`
reused rather than re-specced, ≥2 self-contained milestone prompts with no
"see PRD" phrase) plus the reconstructed final assistant text and the raw
JSONL's tool-call/safety-block stream.

| run | model | wall | turns | in / out tokens | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|
| baseline (no skill) | qwen3.8-27b | 311s | 11 | 283.0k / 25.4k | 0 | 1/14 | never invoked `/skill prd`; misread the brain dump as an architecture request (it saw the installed `architecture` skill via `context(scope="skills")`) and wrote `docs/architecture-log-triage-v1.md` instead — no `PRD.md`, no milestones |
| v1 (frozen 0.3.0) | qwen3.8-27b | 492s | 12 | 408.3k / 43.4k | 2 | 12/14 | correct `PRD.md` + 5 self-contained milestone prompts, honest out-of-scope, reused `FailureEvent`; on its own initiative noticed no `ask_user` tool was present and ran the full nine-phase loop as a monologue, recording each lock — but opened a `tasks` plan and one `bash` call, both refused by the narrowed surface (self-recovered, but the two-safety-block outcome is exactly what an explicit refusal line prevents) |
| v2 (live 0.4.0) | qwen3.8-27b | 299s | 9 | 197.4k / 22.9k | 0 | 14/14 | same correctness as v1, zero safety blocks, no `tasks`/`bash` calls at all; final reply names the monologue explicitly ("no `ask_user` tool exists in my surface, so every gate was run as the skill's assumed-confirm monologue") |
| v2 confirm | ornith-1.5-35b-a3b | 79s | 11 | 169.9k / 11.5k | 0 | 14/14 | fastest of the four runs by a wide margin, same shape and grounding, zero safety blocks |

**Changes**: (1) `## Arguments` contract with an explicit headless/no-operator
rule — every one of the nine phases runs as an assumed-confirm monologue
when no one answers a gate, not just the first one, matching the pattern
ported from `architecture`/`product-intent`; (2) an explicit `tasks` and
`bash` refusal line — these were v1's only two failures, both self-recovered
by this model but a real safety-block pair on a weaker or more literal one;
(3) "Read the repo before asking" now says explicitly that an existing
entity gets reused and marked, not re-specced, closing S3; (4) the
`PRD.md` line now names the wrong shapes to avoid (`docs/PRD.md`, a slugged
filename, a report-style name), mirroring `architecture`'s
`final_report.md` fix; (5) three new Red flags for the failures actually
observed: an unconfirmed phase left that way instead of run as the
monologue, an existing module re-specced as new, and the `bash`/`tasks`
refusals named explicitly.

**Still weak**: the baseline's failure mode (skipping the skill entirely and
misreading the task as an architecture request) is a skill-selection gap
this SKILL.md cannot fix from inside its own body — it only activates once
invoked. Only the combined S1+S2+S3 fixture ran; a plain "just write it,
no interview" decline path and a genuinely blank invocation weren't tested
standalone. `ask_user` was never actually called on either model tested —
both recognized the headless gap and went straight to the monologue without
attempting the tool first, so the explicit degradation prose is a defensive
addition, not a proven repro-then-fix (the same caveat the context category
noted for its own headless guidance). `code_nav` (in allowed-tools) was
never exercised. Only 27–35B class models tried, no small-model run.
