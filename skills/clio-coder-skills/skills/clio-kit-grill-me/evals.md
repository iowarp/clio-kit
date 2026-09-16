# Evals - grill-me

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 - vague feature request

Setup: repo with an existing config domain. Prompt: "grill me on adding plugin
support."

Expected:

- Scans the existing config/extension code before asking anything the repo can
  answer.
- Starts with a root decision such as outcome, target user, or scope boundary.
- Uses `ask_user` with `mode: "single_question"` and exactly one question in
  the round.
- Puts its recommended answer first when options are supplied.
- Ends by calling `ask_user` with `action: "complete"` and then writes a
  decision log, not a summary paragraph.

## S2 - answerable-from-repo question

Setup: plan mentions "the test runner". The repo's package.json defines it.

Expected:

- Does NOT ask the user which test runner is used; reads package.json instead.
- States what it found and asks only whether that constraint should remain
  true.
- Treats the area as review mode, not fill mode.

## S3 - user defers

Setup: mid-interview, user answers "whatever you think is best."

Expected:

- Records its own recommendation as the decision and says so explicitly.
- Does not silently skip the branch.
- Continues only if another root decision remains.

## S4 - long phased interview

Setup: prompt asks for a deep stress test of a large design.

Expected:

- First `ask_user` call sets a bounded `max_rounds` value such as 12 or 16.
- Still asks one question per round.
- Completes before the limit when decisions are sufficient.
- If the limit is near, closes with current decisions and open risks instead
  of continuing to ask.

## S5 - stop signal

Setup: mid-interview, user says "stop", "enough", "later", or cancels the
modal.

Expected:

- Stops immediately and does not ask a confirmation question.
- Calls `ask_user` complete when possible with partial decisions.
- Final response includes partial decisions and the next unresolved root
  question.

## Baseline failure modes to watch for (RED)

- Question batching: more than one question in an `ask_user` round.
- Interview starts cold without reading repo facts that are clearly relevant.
- Interview ends when the user gets tired, with no decision log.
- Asks about facts discoverable via grep/read.
- Hits the round limit without a useful closeout.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS (smoke). Skill loaded and interviewed; judge emitted nothing (truncation).

## Battletest record (2026-09-03)

Fixture: `/home/akougkas/eval-temp/harness/test_grillme.py`, continuing the
planning category's shared HPC log-triage domain. Seeds the *decided* v1
architecture doc (`docs/hpc-log-triage-architecture.md`: on-demand reads
chosen over an always-on pipeline, alerting explicitly listed "Not
decided"), the partial `src/scanner.py` (`FailureEvent` + OOM-only
`scan_oom`), and a `pyproject.toml` declaring `pytest` as the test runner
— a repo fact the interview must find via `read`, not ask about (evals.md
S2). The prompt is S1's vague-but-grounded feature request: "grill me on
adding Slack alerting... nothing about notifications or Slack is decided
yet." `ask_user` in this harness auto-cancels immediately with no operator
(confirmed by the planning category and reconfirmed here), which makes
this skill's entire premise — a one-question-at-a-time live interview — the
thing under test. Graded 10 checks against the reconstructed final
assistant text and the raw JSONL's tool-call/safety-block stream: zero
safety blocks; zero `tasks` calls; no wasted `ask_user` retries (<=3
calls); repo facts (`scan_oom`, `FailureEvent`, on-demand, `click`,
`pytest`, `scan_ecc`/`scan_xid`, top-3) grounded in the final text; the
Step 5 decision-log shape present (`Decision Log`/`Deferred`/`Open
risks`/`Recommended next step`, not a summary paragraph); decisions
numbered; the `assumed — confirm` monologue used 3+ times; the Question
Priority order respected (user/problem framing before naming/polish); and
**the phase map actually completed** (4+ distinct phase numbers named,
not a stall after phase 0/1). S3 (user defers) and S5 (stop signal) are
not separately exercised — a headless run cannot produce a live "whatever
you think" or "stop" reply to react to; by construction, the
assumed-confirm monologue *is* the "whatever you think" case running for
every decision, so S3's expected behavior (record the recommendation,
say so explicitly) is exercised implicitly on every phase, every run.
S4 (long phased interview, near round-limit closeout) was not exercised
standalone. Primary model `qwen3.8-27b` on `dynamo` (LM Studio); cross-model
confirm on `ornith1.5-35b-moe` on `mini` (llama.cpp), required per this
session's brief because a prior pass in the same mission found fixes tuned
on one model family did not always transfer to another.

| run | model | wall | turns | in / out tokens | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|
| baseline (no skill) | qwen3.8-27b | 42s | 3 | 32.5k / 3.7k | 0 | 5/10 | never invoked `/skill grill-me`; read both key files and grepped for "alerting", grounded correctly, but produced one live-style question and stopped — no phase map, no decision log (expected: this is what the skill exists to fix) |
| v1 (frozen 0.4.0) | qwen3.8-27b | 57s | 7 | 91.8k / 5.2k | 2 (1 real: `tasks` refused; 1 benign ENOENT on a fixture-dangling doc path) | 3/10 | opened a `tasks` plan for its own phase map (refused — `tasks` was never called out as off-surface in the frozen body), scanned thoroughly, then asked exactly **one** question in plain text and ended the turn — never called `ask_user` at all, never produced a decision log. This is the exact failure the session was built to catch: the frozen skill's fallback line ("if `ask_user` is unavailable, ask in plain text... keep an internal decision log") reads as permission to have an ordinary single-question live conversation, and the model took it literally |
| v2 (first hardened cut, 0.5.0) | qwen3.8-27b | 121s | 6 | 78.9k / 11.2k | 1 (benign ENOENT, same dangling doc path) | 9/10 | scanned via `git`/`read`/`ls`, reasoned "ask_user is not in my tool surface" (not quite accurate — it's always exempt — but harmless), and correctly ran the full assumed-confirm monologue through Phases 0-5 to a complete, correctly-shaped decision log, entirely without ever calling `ask_user`. Fixture's dangling PRD reference removed after this run to isolate real vs. benign blocks |
| v2 re-run (fixture fixed) | qwen3.8-27b | 134s | 7 | 101.0k / 12.1k | 1 real: `bash wc`/`bash head` on the sample log, refused | 9/10 | same strong monologue and decision log; the one new gap was a `bash` reflex the frozen and first-cut bodies never explicitly forbade (grill-me never had `bash` in `allowed-tools`, but nothing told the model not to reach for it) |
| v3 (bash-refusal added) | qwen3.8-27b | 97s | 5 | n/a | 0 | 6/10 | `bash` reflex gone, but a **regression**: stated "this harness has no `ask_user` tool" (still not quite accurate) and, without ever calling `ask_user`, asked one plain-text question and ended the turn on "Answer 1 / 2 / 3" — the exact frozen-skill failure recurring under the hardened body, proving the earlier fix's trigger ("switch on the first `cancelled` result") had a gap: a model that never calls `ask_user` at all never receives a `cancelled` result to trigger the switch |
| v4 (no-back-and-forth line added) | qwen3.8-27b | 124s | 6 | 85.1k / 11.0k | 0 | **10/10** | full 9-round monologue (Phases 0-5), correctly-shaped decision log, zero `tasks`/`bash` calls |
| v5 (stability re-run) | qwen3.8-27b | 95s | 5 | n/a | 0 | **10/10** | repeat of v4's result, confirming v3's regression was not the new steady state |
| final (cross-model confirm) | ornith1.5-35b-moe (mini) | 64s | 5 | 12.5k / 5.4k | 0 | **10/10** | full monologue and decision log on the second model family too; independently proposed a Slack-message trigger threshold not in the fixture and, in its closing line, distinguished what it would still "press on live" from what it correctly resolved on its own headless — the clearest sign the headless/live distinction actually landed as a real distinction, not just prose the model echoes back |

**Changes** (0.4.0 -> 0.5.0):

1. **`## Arguments` contract**, the section the skill never had — slash-free
   conversational syntax, and the ported no-operator/`ask_user`-auto-cancels
   rule from the planning category, adapted for a skill whose entire
   contract is "one question at a time" rather than a document write.
2. **The critical fix, found empirically, not guessed up front**: the
   frozen skill's fallback line ("if `ask_user` is unavailable, ask in
   plain text... one question at a time... internal decision log") reads,
   correctly, as "have a normal one-question conversation" — which is
   exactly wrong when there is no second turn for a reply to land in. The
   fix that actually held (v4/v5, and the cross-model run) is a **"this run
   has no back-and-forth" statement that does not gate on receiving a
   `cancelled` result** from `ask_user` — v3's first attempt gated the
   switch-to-monologue on "the first round comes back cancelled," which
   left a real gap: a model that reasons "`ask_user` isn't available" and
   never calls it at all never receives that trigger, and fell straight
   back into the frozen skill's exact failure. The held version states the
   no-reply-coming rule as a fact about the run itself, independent of
   whether `ask_user` was ever invoked.
3. **Explicit `tasks` and `bash` refusal**, in Arguments and Red Flags —
   `tasks` was v1's real safety block (opened a plan for its own phase
   map); `bash` was a reflex on the v2 re-run (`bash wc`/`bash head` on a
   file that should have been `read`). Both are outside `allowed-tools`
   already; the gap was that nothing said so in the body.
4. **Step 3 and Step 4 rewritten** to name the headless-cancellation
   handling explicitly at the exact point in the workflow it applies (not
   only in Arguments): Step 3 states that neither a `cancelled` result nor
   never calling `ask_user` at all is a reason to wait; Step 4 draws the
   line between a live stop signal (still respected) and a headless
   `cancelled` result (not a stop signal, a cue to keep going).
5. **`git` tool and `code_nav` called out explicitly in Step 1** for repo
   state and symbol lookups respectively — `code_nav` remains unexercised
   by every run this session (see Still weak); `git` was used in every
   hardened run once named.
6. Five Red Flags entries rewritten or added around the concrete failures
   observed: `tasks`/`bash` reaches, re-calling `ask_user` after a
   cancellation, treating `cancelled` as a user stop signal, and — the
   headline one — ending a turn on "Answer 1/2/3" instead of running the
   monologue to the decision log.

**Still weak**: `code_nav` (in `allowed-tools`) was never exercised by any
run this session — this fixture's grounding lived entirely in prose files
and one Python module small enough that `read`/`grep` sufficed; a fixture
with a larger call graph might exercise it, but none was built. S3 and S5
are reasoned about, not directly run (see the fixture note above) — a
harness that could inject a specific `ask_user` reply (rather than always
auto-cancelling) would let those two scenarios run for real instead of by
inference. S4's near-round-limit closeout behavior is unverified; every
hardened run here converged well under any plausible `max_rounds` value.
v3's regression is the one data point worth remembering past this session:
gating a headless-degradation rule on "the first tool result that comes
back a certain way" is fragile against a model that skips the tool call
entirely and reasons its way to the same wrong conclusion by a different
path — the fix needed to be a fact about the run, not a reaction to one
tool's return value. Two consecutive clean runs on the primary model and
one clean cross-model run is reasonable but not exhaustive evidence that
v3's failure mode is fully closed rather than just less frequent; only a
larger run count would raise that confidence further.

## Live interactive confirmation (2026-09-03, Herdr pane)

The headless harness can only prove the assumed-confirm monologue path; it
cannot produce a genuine "whatever you think" or "stop" reply because
`ask_user` always auto-cancels with no operator. To exercise S3 and S5 for
real, this session ran the hardened 0.5.0 skill interactively: a Herdr pane
running `clio-coder` (target `mini`, model `ornith1.5-35b-moe`, switched
in-session via `/model`) against the same fixture repo the battletest used
(`/home/akougkas/eval-temp/grillme-final`, skill installed project-locally
at `.clio-coder/skills/grill-me/` so the live session resolved this
repo's edited SKILL.md rather than the separately npm-installed package
copy — `/skill grill-me` has no path-override flag the way `clio-coder run
--skill` does), with a human answering each `ask_user` round live via the
TUI's modal.

Result: Step 1 scanned the repo before asking anything (architecture doc,
scanner.py, sample log). Round 1 asked a single root-decision question
(trigger model) with the recommended option first and real tradeoffs on
the alternatives. After the human's real answer came back, the model's own
reasoning explicitly named the distinction the whole hardening pass turned
on: *"The modal returned round_answered — there is an operator here...
this is a live interview, not headless. I'll continue with one question
per round and wait for your answers."* — proof the headless/live branch is
a real fork in the model's behavior, not just prose it echoes. Five rounds
ran in priority order (trigger, success measure, non-goals, delivery/
secrets, payload shape); round 3 was answered "whatever you think is
best" and recorded the stated recommendation as the decision (S3,
confirmed live, not just implied by the monologue path); round 5 was
answered with an appended stop signal ("...; stop") and the model called
`ask_user` `action: "complete"` immediately, with no confirmation question
(S5, confirmed live). The final decision log matched the Step 5 shape
exactly (numbered decisions, Deferred, Open risks, Recommended next step)
and, unprompted, flagged a real cross-cutting risk the fixture didn't
spell out: the alerting feature's success measure depends on Phase 2's
ranking/CLI, which doesn't exist in `scanner.py` yet — a genuine "hole
poked," not filler. Zero safety blocks; zero `tasks` calls. One aside, not
a skill defect: the TUI's own usage nudge fired ("9+ read-only exploration
calls without a successful Scout dispatch") — Step 1's repo scan currently
reaches for `read`/`grep`/`ls` directly rather than delegating broad
reconnaissance to a Scout dispatch, worth a look in a future pass but out
of scope for this one since the scan itself was correct and grounded.

This closes the S3/S5-not-directly-run gap noted above for the specific
case of a genuine live operator; the headless assumed-confirm path
remains separately and repeatedly confirmed on its own terms.
