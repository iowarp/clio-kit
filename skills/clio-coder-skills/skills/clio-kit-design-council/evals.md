# Evals - design-council

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - storage-format decision with real tradeoffs

Setup: a project checkpointing large arrays; prompt: "council: should we use
HDF5 or Zarr for our checkpoint format?"

Expected:

- Composes 3 to 5 perspectives from the topic (each with a name, stance,
  expertise, and a target to attack), not from a generic role menu.
- Dispatches perspectives as read-only workers (scout, researcher, or
  provenance recipes), parallel within a round.
- Round 2 responses address named points from the round 1 transcript.
- Disagreement is surfaced with its crux stated, not averaged away.
- Synthesis lists agreements, live disagreements with the crux,
  a recommendation, and preserved dissent, citing the transcript.
- Dispatch receipts exist linking statements to worker runs.

## S2 - consensus topic

Setup: prompt asks the council to debate a question with an obvious answer for
this repo (for example "should we hand-roll our own YAML parser or keep the
existing dependency?").

Expected:

- Round 1 comes back in agreement; the council stops there.
- Says explicitly that the council was not needed and returns the consensus.
- Does not run responses or convergence rounds, and does not manufacture
  friction.

## S3 - dispatch unavailable

Setup: dispatch is admission-denied or absent in the environment.

Expected:

- Falls back to inline sequential perspectives with the same round structure
  and synthesis format.
- Labels the result as degraded (single-model debate, no receipts).

## S4 - anti-trigger: plan stress-test

Setup: user says "poke holes in my plan" or wants their decisions interrogated
one at a time.

Expected:

- Refers to grill-me instead of convening a council.

## Baseline failure modes to watch for (RED)

- Generic personas (optimist, pessimist, devil's advocate) with no expertise
  or attack target.
- One blended essay of pros and cons instead of independent positions that
  respond to each other.
- No worker dispatch at all, or workers asked to edit files.
- A "balanced" synthesis that hides the crux of disagreement.
- Fabricated debate on a consensus topic to justify the ceremony.

## Observed live-smoke results

Run 2026-07-01/02, headless `clio-coder run --skill` against a scratch fixture (an
MPI checkpointing module writing one raw npy per rank per step).

- Full council (atomic rename vs direct write): three bounded rounds, four
  topic-composed perspectives each dispatched as read-only `scout` workers
  (parallel within rounds, sequential between), receipts on every statement
  (exit 0, agentId scout), synthesis with agreements, a crux, and the
  Performance Engineer's dissent preserved and answered. Note the topic
  proved genuinely debatable, so full rounds were correct.
- Degraded fallback (HDF5 vs Zarr, run while the fleet target had model
  residency failures): the skill attempted a parallel `dispatch` first,
  observed worker failures in receipts, then ran the same rounds inline and
  labeled the result degraded exactly as the clause requires.
- Early termination (S2, zero-padded vs unpadded checkpoint filenames): the
  first draft of the skill ran full rounds on side-quibbles for two
  debatable-in-hindsight "consensus" topics; the composition and termination
  clauses were tightened to judge consensus on the decision question itself.
  With the shipped wording the council declared itself not needed before any
  dispatch, cited the rule, and returned the consensus with caveats. No
  manufactured friction.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT COMPLETED: treatment timed out at the 900s ceiling mid-council (serial perspectives). Needs a longer timeout or fewer personas for headless eval; skill was visibly working when cut.

Follow-up (2026-08-13, v0.3.0): Step 1 now makes three perspectives the
default and tells a headless run to use three and one round, which is the
blocker the timeout exposed. Not re-run, so `eval-status` stays
`scenarios-recorded`; the next campaign has to confirm the shortened council
fits the ceiling.

## Battletest record (2026-09-03) — 0.4.0 -> 0.5.0

Fixture: `/home/akougkas/eval-temp/harness/test_designcouncil.py`, a
self-contained repo (`src/checkpoint.py` writing one raw `.npy` per rank per
step to a shared parallel filesystem, `src/config.py`/`config.yaml` using
`pyyaml`) adapted from this file's own worked examples. S1 = HDF5-vs-Zarr
(genuinely contested), S2 = hand-roll-YAML-vs-keep-pyyaml (consensus), S4 =
"poke holes in my plan" anti-trigger. `runner.py`'s `timeout=` was raised to
1800s for this skill specifically — the 900s default is a `dispatch` fan-out
ceiling, not a model-quality one, and the mission called for confirming the
v0.3.0 "three perspectives, one round" headless fix on its own terms, not
routing around it. Primary: `qwen3.8-27b`/`dynamo`. Cross-model confirm:
`ornith1.5-35b-moe`/`mini`. Both runs shared the `dynamo` LM Studio endpoint
with concurrently active sibling battletest sessions (`grill-me`,
`workflow-distiller`) launched via `herdr` during this pass — real,
externally-caused contention, not a fixture artifact; see below.

| run | scenario | model | wall | turns | dispatch calls | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|---|
| baseline (no skill) | S1 | qwen3.8-27b | 121s | 5 | 0 | 0 | 3/11 | no skill invoked (ran under `--no-skills`); the model noticed the installed skill anyway and narrated "four positions, real cross-examination" as one inline monologue — a solid single-model analysis but no dispatch, no receipts, no named-recipe workers |
| v1 (frozen 0.4.0) | S1 | qwen3.8-27b | 1740s, **killed at the 1800s ceiling** (exit -9) | 22 | 12 | 7 | 4/11 | composed **four** perspectives and ran into **round 2**, both violating the stated headless "three perspectives, one round" default; opened a `tasks` plan (refused); two `dispatch` calls rejected for an absolute path token in `briefing` (`legacy_scope_path_absolute`); one denied for endpoint capacity; one **hard-blocked as `rm-recursive-or-force`** because a perspective's own round-1 argument prose said "...you can `ls`, checksum, `rm -rf`, and publish..." — the admission layer's damage-control scan matched the quoted shell syntax inside the debate text itself, not an executed command; reached for `bash` to inspect a receipt file (refused). This is the same failure class the 2026-08-13 smoke record flagged, confirmed still present, worse: the v0.3.0 fix was never actually followed |
| v2 (first hardened cut) | S1 | qwen3.8-27b | 416s | 11 | 3 | 4 | 8/11 | one batched `tasks`-array `dispatch` call, all 3 capacity/timeout-denied under real endpoint contention; correctly fell back to Degraded mode, cited the exact denial reasons, labeled the output degraded, produced a complete synthesis with agreements/crux/dissent — no `tasks` misuse, no shell-block trigger, no one-at-a-time call splitting |
| v2 (post Step 0 strengthen) | S1 | qwen3.8-27b | 304s | 7 | 3 | 4 | 8/11 | same shape: parallel wave denied, sequential retry denied/timed out, correct Degraded fallback, well-formed synthesis citing the transcript |
| v2 | S2 | qwen3.8-27b | 52.5s / 60.6s (before/after Step 0 edit) | 5 / 6 | **0** | 0 | **5/5** both | Step 0 stopped before any dispatch, stated the council was not needed, gave the consensus (keep `pyyaml`, `safe_load` only) with caveats, and named the repo's actually-contested question (the checkpoint format) as a pointer — no manufactured friction either run |
| v2 | S4 | qwen3.8-27b | 69.4s | — | 0 | 0 | 3/3 | did not convene a council; ran a grill-me-shaped one-question-at-a-time interrogation instead and named `grill-me` |
| v2 (cross-model) | S1 | ornith1.5-35b-moe/mini | 384s | 10 | 4 | 5 | 7/11 | 3 of 4 `dispatch` calls denied/timed out on the **same `dynamo` endpoint** (workers defaulted there even though the orchestrator itself ran on `mini`); one stray `monitor` call refused (outside the skill's surface); correctly diagnosed the capacity pattern in its own words ("times out at ~57s... despite reporting 0/4 slots in use... exhausted the capacity fallback") and ran Degraded mode with a clearly labeled warning banner |
| v2 (cross-model) | S2 | ornith1.5-35b-moe/mini | 239.5s / 188.1s (before/after Step 0 edit) | 6 | 3 / 2 | 3 / 2 | 2/5 both | did **not** skip Step 0's dispatch the way qwen did — ran one capacity-retried round anyway, but stopped there (no round 2/3), reached the same correct consensus ("keep pyyaml", supply-chain caveat preserved as a contingent dissent, not manufactured), and stated the council-not-needed conclusion explicitly; the Step 0 wording strengthen did not change this model's behavior |

**S3 (dispatch unavailable) — no clean forced trigger found, confirmed by
reading the CLI, not assumed**: `clio-coder run --help`'s `--tool-profile`
narrows a *dispatched sub-agent's own* tool set and requires `--agent`
(`clio-coder run: fleet dispatch flags require --agent <recipe-id>`); there
is no flag that strips `dispatch` from a top-level `--skill` orchestrator
run's own surface in this harness. Editing the skill's own `allowed-tools`
to omit `dispatch` would test a different skill, not this one. This is a
real, documented harness gap, not faked around. In its place, S3's expected
behavior was exercised **organically, repeatedly, for real**: every S1 run
on both models hit genuine `dispatch` admission denial or timeout from
endpoint capacity, and Degraded mode fired correctly every single time —
labeled degraded, same round structure, synthesis format intact, no
receipts claimed that did not exist.

**The dispatch-viability and recipe-existence questions, resolved
empirically:**

- `scout`, `researcher`, and `provenance` all exist as builtin shadow-agent
  recipes (`src/domains/agents/builtins/*.md`, `capabilityClass: read-only`)
  and are dispatchable — every hardened run that got a `dispatch` call
  admitted used one of them by name and got real worker output back.
- Each recipe's own system prompt declares a rigid JSON-only result
  contract (`{"findings":[...]}`, `{"source":...}`, `{"confirmedFacts":...}`)
  that has nothing to do with a debate position. In practice this was
  **not** a blocker: V1's successfully-admitted round 1 came back as full
  argumentative prose (named claims, attacks, citations, mind-change
  conditions) from `researcher`/`scout` workers, not the declared narrow
  JSON shape — the persona in the task prompt won out over the recipe's
  own stated output contract for the caller-facing transcript. Worth
  knowing, not worth re-architecting around.
- The actual, dominant blocker is **`dispatch` admission capacity on a
  single-instance local LM Studio target**. `~/.local/state/clio-coder/
  endpoint-slots.json` records the `dynamo` endpoint (`100.104.197.69:1234`)
  at `"slots": 1`. A 3-task parallel wave is denied outright
  (`capacity exceeded (3/1 slots)`), and the `mode="sequential"` retry this
  pass added to the skill is followed correctly by both models but still
  gets denied or times out (`admission timed out after ~57s... 0/4 worker
  slots in use` — a scheduling state that never resolves within the wait
  window), evidently because the orchestrator's own foreground session
  already holds the endpoint's only slot. This reproduced independently on
  `mini` too: workers dispatched from an orchestrator running on `mini`
  still routed to `dynamo`'s endpoint by default, so the same 1-slot
  contention applied there as well — confirming the brief's suspicion that
  worker fan-out may silently default to an unexpected node/target. Some
  of this pass's contention was real concurrent load, not just self-
  contention: `ps` showed sibling `grill-me`/`workflow-distiller`
  battletest sessions actively running via `herdr` against the same
  endpoint during these runs.
- The full-auto/`authorityBasis` auto-grant fact supplied going in
  (`deps.getAutonomy?.() === "full-auto" ? "full-auto-policy" :
  "operator-plan-approval"`) turned out to be **moot for this skill**:
  that gate only applies to `agent:"auto"` dispatch requests
  (`src/tools/dispatch-arguments.ts`, `agentSelection` is only populated
  when `requestedAgent === "auto"`). Design-council always pins an explicit
  recipe id (`scout`/`researcher`/`provenance`), so `agentSelection` stays
  `undefined` and the operator-approval/full-auto-policy distinction never
  engages — admission for this skill's calls is governed purely by the
  capacity/reservation machinery above, independent of `--autonomy`.
- `legacy_scope_path_absolute`: an absolute path token (e.g. a value read
  from `config.yaml`) pasted into `task`/`briefing` prose without a
  declared `intent` is rejected. Fixed by telling the skill to declare
  `intent.read_roots`/`relevant_paths` (relative) on every call instead.

**Changes (0.4.0 -> 0.5.0):**

1. **`## Arguments` contract**, ported from `grill-me`/`cut-it`'s shape.
   States headless is the *enforced* default (exactly three perspectives,
   exactly one round), not a self-assessed suggestion — the v0.3.0 fix's
   prose alone did not hold on either model (V1 composed four perspectives
   and ran round 2 headlessly).
2. **Dispatch call shape spelled out**: one batched `tasks`-array call per
   round; on capacity denial, retry the *same* batch with
   `mode="sequential"` in one call rather than splitting into several
   single-task calls issued one at a time (V1's actual failure — 12
   separate `dispatch` calls, a `list:true` probe, and manual receipt
   reads via `bash`, all of which V2 stopped doing).
3. **`intent.read_roots`/`relevant_paths` guidance** to avoid
   `legacy_scope_path_absolute` rejections from paths quoted in prose.
4. **No literal shell syntax in a persona's argument text** — added after
   V1's `rm-recursive-or-force` hard block fired on debate prose, not a
   real command.
5. **`ask_user` added to `allowed-tools`** (already always-exempt, now
   documented) with one call at Step 1 on a contested topic, asking
   quick-vs-full-debate depth; headlessly it cancels, which *is* the
   three-perspective/one-round confirmation, mirroring the auto-cancel
   pattern the planning category established rather than relying on
   unaided self-assessment.
6. **Explicit `tasks`/`bash`/receipt-re-read refusal lines**, matching the
   sibling skills' pattern (`tasks` opened a plan in V1; `bash` was reached
   for twice across the runs above to re-inspect a receipt the dispatch
   call's own result already contained).
7. **Step 0 strengthened** against dispatching "just to confirm" a
   consensus call already reached — tested on `ornith1.5-35b-moe`/`mini`
   (still dispatches once) and `qwen3.8-27b`/`dynamo` (unaffected, already
   correct); see Still weak.
8. Five new Red flags entries naming the concrete failures observed above.

**Still weak:**

- **Step 0's short-circuit is model-family-dependent.** `qwen3.8-27b`
  trusts its own contested-or-not judgment and skips `dispatch` entirely
  for S2 (5/5, 0 dispatch calls, both before and after the Step 0 edit).
  `ornith1.5-35b-moe` does not: it dispatches one (capacity-retried) round
  "to confirm" even on the same consensus topic, both before and after the
  strengthened wording — 2/5 unchanged. It still stops after that one
  round, still reaches the correct consensus, still preserves the dissent
  as a contingent caveat rather than manufacturing friction, so the
  user-facing outcome is fine; the wasted dispatch cost is the actual gap,
  and more prose did not move it, consistent with the planning category's
  own finding that some model-family behaviors don't fully generalize no
  matter how much repetition is added.
- **S1 never completed as a genuine 3-perspective/1-round council with
  real receipts** on this pass — every attempt on both models hit real
  endpoint capacity contention and degraded. The Degraded path is now
  proven solid, but the "happy path" (dispatch succeeds, synthesis cites
  real worker receipts) is unverified under this specific pass's
  conditions; V1's transcript shows it *can* succeed (four real worker
  runs completed with citable output before the round-2/shell-block
  failure), so this reads as an availability problem this pass's timing
  ran into, not a structural block — but it means the exact scoring
  bullets that depend on `dispatch_receipts_ok` (perspective count,
  receipts-ok) were not exercised clean this round.
  `perspectives_dispatched`/`used_named_recipes` in the grading script
  count attempted, not completed, dispatch tasks for this reason.
- **No purpose-built S3 fixture** — see above; a future pass with a
  dedicated low-capacity or offline dispatch target would let this be
  tested directly instead of relying on organic contention.
- `code_nav` and `grep` were barely exercised (fixture is small enough
  that `read`/`ls`/`find` covered grounding in most runs).
- Only one fixture domain (HDF5-vs-Zarr / YAML-parser) ran this pass; the
  richer four-perspective MPI-checkpoint example from "Observed live-smoke
  results" above was not re-run against 0.5.0.
