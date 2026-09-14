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

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. SMOKE ACTED but substance bullets failed (wrote final_report.md, skipped the loop). Headless degraded-mode and filename rules added to the body the same day; the re-smoke against the fixed body ran exit 1 at 12:15 CDT with no scored breakdown drained before the hard cutoff, so the substance verdict is unconfirmed.

## Battletest record (2026-09-03)

Fixture: `/home/akougkas/eval-temp/harness/test_architecture.py`, continuing
the planning category's shared HPC log-triage domain from `product-intent`/
`prd`. Brownfield: seeds the actual `docs/hpc-log-triage.prd.md` intent doc
and a partial codebase (`src/scanner.py`: working `FailureEvent` + OOM-only
`scan_oom`, ECC/Xid not implemented; `requirements.txt` with click/pyyaml)
inside a git repo. The prompt asks for the v1 engineering approach for
dragon-cluster and blade-cluster and names the one genuine one-way-door
tension from the fixture family: an always-on log-ingestion pipeline vs.
on-demand reads. Combines S1 (brownfield: read existing surfaces, reuse
`FailureEvent`/`scan_oom` rather than re-spec) with a spike-worthy
one-way-door call into one gradable run. Graded 13 checks against real
post-run disk state (`docs/architecture-<slug>.md` at the exact default
path — no `final_report.md`/`REPORT.md` miss, all seven required sections,
>=2 genuinely distinct approaches inside "Approaches considered", the
pipeline-vs-on-demand tension named *and* backed by a spike with a decision
rule in "Spikes & experiments", grounding in the seeded PRD/code) plus the
reconstructed final assistant text and the raw JSONL's tool-call/safety-block
stream. Ran on `mini`/`ornith1.5-35b-moe` only this pass (see Still weak).

| run | model | wall | turns | in / out tokens | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|
| baseline (no skill) | ornith1.5-35b-moe | 110s | 5 | 11.1k / 8.5k | 0 | 3/13 | never invoked `/skill architecture`; investigated correctly (read the PRD and scanner.py) but then called the terminal `artifact` tool mid-reasoning with no arguments and ended the run before writing anything — no decision doc, no alternatives, no spike |
| v1 (frozen 0.3.0) | ornith1.5-35b-moe | 145s | 9 | 9.7k / 10.0k | 2 real + 1 benign | 10/13 | wrote the correct `docs/architecture-hpc-log-triage.md` with all sections, 3 real alternatives (a table: on-demand / always-on / hybrid-with-gate), the pipeline tension named with a spike and an explicit decision rule, and ran the full headless assumed-confirm loop entirely on its own initiative (the existing body prose already covers this well) — but reached for `bash` once (refused) and opened a `tasks` plan once (refused); a third "error" was a harmless `ls` on a directory the fixture never created |
| v2 (first hardened cut, 0.4.0) | ornith1.5-35b-moe | 125s | 9 | 5.5k / 7.6k | 1 real + 2 benign | 10/13 | `tasks` call eliminated entirely by the new explicit refusal line; still reached for `bash find .` once before self-recovering with the `find` tool — the added Red flags/Arguments refusal line reduced but did not eliminate the bash reflex, matching the pattern `prd`/`product-intent` saw on this same model family; 2 benign `read` ENOENTs (evidence docs the PRD cites but the fixture never seeded) also counted as errors under this harness's blanket isError grading |
| v3 (final 0.4.0) | ornith1.5-35b-moe | 159s | 7 | 8.5k / 9.7k | 0 real + 1 benign | 12/13 | zero `bash` calls, zero `tasks` calls, correct path, all 7 sections, 2 distinct approaches, pipeline tension with a spike and decision rule; the one remaining "error" is a harmless duplicate `context` call the harness nudged back onto the loaded skill (not a tool-surface refusal, `context` is always exempt) |

**Changes** (0.3.0 -> 0.4.0): (1) an `## Arguments` contract — the skill had
solid headless-degradation prose in its intro already, but no formal
contract section; added slash-invocation syntax, what's required vs.
inferred, and the explicit no-operator/`ask_user`-returns-empty rule ported
from `product-intent`/`prd`, extended to name Step 0's evidence-check and
Step 1's greenfield/brownfield call explicitly (not just Step 2's decisions)
as places the degradation applies from; (2) explicit `tasks` and `bash`
refusal lines in Arguments and Red flags — these were v1's two real safety
blocks; (3) Step 3's filename rule tightened from a permissive "default
location... never invent another" into an enumerated three-item list with
"no other filename or location is ever correct," closing the exact gap the
2026-08-13 smoke record flagged (`final_report.md` on a different model);
(4) a one-line `find`/`ls`-not-`bash` hint added directly in Step 0, where
the exploration happens, which is what took the bash-reach from 1/run (v2)
to 0/run (v3); (5) five new Red flags entries naming the concrete failures
observed: wrong filename, a stalled or skipped headless loop, `tasks`/`bash`
reaches, and ungrounded claims.

**Still weak**: per this pass's coordinator note, only `ornith1.5-35b-moe`
on `mini` was run — no `qwen3.8-27b`/dynamo confirmation this session, so
cross-model-family generalization is unverified (the sibling `prd`/
`product-intent` runs found the bash-reflex fix that worked on `qwen3.8-27b`
did *not* fully generalize to `ornith-1.5-35b-a3b`, so the reverse — a fix
tuned on ornith not holding on qwen — is a live risk here too, untested).
The v3 run's one remaining logged "error" is a benign duplicate `context`
call, not a real defect, but the harness's blanket isError-counting means a
literal 13/13 may not be reachable without also silencing genuinely benign
tool responses — not worth chasing. `web_fetch` and `code_nav` (both in
allowed-tools) were never exercised — this fixture never needed them
(brownfield, no external research). The baseline's specific failure (an
unprompted terminal `artifact` call mid-investigation, with no arguments)
is a skill-selection/tool-choice issue outside this SKILL.md's own body.
Only one fixture shape ran (brownfield + one-way-door); a pure-greenfield
scenario (S2's familiarity-vs-fashion pull) and a mid-session altitude
check (S3, "just list the files to change") from the scenario list above
were not exercised standalone against 0.4.0.
