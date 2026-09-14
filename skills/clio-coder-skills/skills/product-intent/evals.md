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

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Interview degraded gracefully headless; PRD written to docs/, judge 5/5.

## Battletest record (2026-09-03)

Fixture: `/home/akougkas/eval-temp/harness/test_productintent.py`. S1's own
domain ("a log-triage tool for HPC operators") made concrete: a repo with
`docs/evidence/support-tickets.md` (3 tickets, a 45-min OOM triage, a
silent-ECC lost queue, a tmux/grep cope with a ~12-node ceiling) and
`docs/evidence/interview-notes.md` (3 operator quotes, including an explicit
switch signal). Task: "write the PRD," grounding docs named but not pasted,
so Step 0's read-first behavior is load-bearing. Graded 12 checks against
real post-run disk state (file at `docs/<slug>.prd.md`, all 9 sections, a
hypothesis with distinct RIGHT/WRONG, >=3 seeded facts grounded, zero
stack-term leaks, non-goals, checkbox open questions) plus the reconstructed
final assistant text (names the path, offers `architecture` next) and
process (zero safety blocks, no `tasks` call). `qwen3.8-27b` on `dynamo`
throughout; one confirm run on `ornith-1.5-35b-a3b`.

| run | model | wall | turns | in / out tok | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|
| baseline (no skill) | qwen3.8-27b | 145s | 11 | 189.1k / 13.1k | 0 | 2/12 | wrote `PRD.md` at repo root (wrong name/path); no interview at all, no hypothesis RIGHT/WRONG block, no non-goals/open-questions sections; opened a task list (harmless here, no skill narrowing the surface) |
| v1 (frozen 0.3.0) | qwen3.8-27b | 137s | 7 | 102.5k / 12.8k | 1 | 10/12 | correct path, sections, hypothesis, grounding, non-goals; opened a `tasks` call refused by the narrowed surface (self-recovered); degraded past the interview on its own reasoning ("`ask_user` isn't in this session's tool surface" — false, it is listed, the model just never tried it) rather than on any instruction in the skill |
| v2 (first hardened cut) | qwen3.8-27b | 224s | 8 | 170.9k / 20.2k | 1 | 11/12 | no `tasks` call; ran the assumed-confirm monologue explicitly through all 5 clusters citing evidence; one `bash` call (a `$(...)` count-check on the written PRD) refused — `bash` was never in this skill's surface, model reached for it anyway to self-verify, then recovered with `grep` |
| v3 (final 0.4.0) | qwen3.8-27b | 227s | 7 | 132.5k / 20.5k | 0 | 12/12 | same correctness as v2, self-verified with `grep`/`read` instead of `bash` after the added Red flags line; zero safety blocks, zero stack leaks, explicit "Process notes" section narrating the headless degradation cluster by cluster |
| confirm (0.4.0) | ornith-1.5-35b-a3b | 69s | 11 | 158.6k / 10.8k | 1 | 11/12 | same content correctness; independently reached for a `bash` echo ("attempting ask_user via context") once, blocked, self-recovered with `grep` — the Red flags line reduced but did not eliminate the `bash` reflex on a second model family |

**Changes** (0.3.0 -> 0.4.0): (1) an `## Arguments` contract stating there is
no operator in a headless run, that `ask_user` returns immediately with no
answers every time regardless of how many times it's called, and that the
fix is to apply the existing "user declines" treatment cluster by cluster
from wherever the first empty response lands — including Step 0's evidence
check, which the old text left ungated but unaddressed for headless; (2) the
decline paragraph in "The interview" now cross-references that headless
default explicitly instead of leaving the model to infer it (v1 inferred a
*wrong* reason — a nonexistent tool-surface gap — and got lucky); (3) an
explicit "the clusters below are the plan; `tasks` is refused" line, which
closed v1's one real safety block; (4) a new `## Red flags` section (the
skill had none) naming the concrete failures seen across runs: stack-term
leaks, a WRONG condition that's just RIGHT negated, the literal `PRD.md`
name, re-calling `ask_user` after an empty response, skipping the loop
outright instead of degrading into it, ungrounded claims, and reaching for
`bash` (not in this skill's surface) to self-verify instead of `grep`/`read`.

**Design note on the biggest named risk**: the mission brief flagged gating
hard on Step 0/cluster 1 and never reaching Generate as the single biggest
risk for this skill. It did not reproduce on either model tested, on any
version including the unhardened v1 baseline snapshot — `ask_user`'s
headless behavior in this harness (confirmed by reading
`src/tools/ask-user.ts`: with no operator handler wired by `clio-coder run`,
every `ask_user` call resolves immediately to `{cancelled: true}`, framed as
an ok result with "proceed with defaults" guidance, never an error or a
hang) means a stalled interview was never actually the failure mode to
defend against here. What *was* real and reproduced on both models: an
unprompted reach for `bash` to self-verify a written document, refused
because `bash` is correctly outside this skill's surface. The hardening
therefore targets the reproduced failure (`tasks` in v1, `bash` in v2/
confirm), not the hypothesized one — matching context/context-handoff's
own finding that the ask_user-stall defense is precautionary, not
repro-driven, here too.

**Still weak**: the `bash`-reach-to-verify reflex was reduced (v2 -> v3 on
qwen3.8-27b: fixed) but not eliminated on ornith-1.5-35b-a3b, which hit the
identical refused-tool pattern even after the Red flags line existed — a
prose warning did not fully generalize across model families, only across
runs of the same one. S2 (explicit "skip the questions, just write it") and
S3 (solution-shaped request, "PRD for adding a reply button") from the
scenario list above were not run standalone against 0.4.0 — only the S1-style
combined evidence fixture ran, five times. The `git` tool (in allowed-tools)
was never exercised in any run; a fixture with prior commits/branches to
reference might exercise it. Only 27-35B class models tried, no small-model
run.
