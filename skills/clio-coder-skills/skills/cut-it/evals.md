# Evals — cut-it

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 — slice a real plan
Setup: cut it.

Fixture:
```bash
mkdir -p src
printf 'function listTodos() {\n  return [];\n}\n\nmodule.exports = { listTodos };\n' > src/todos.js
printf '# PLAN\n\nGoal: minimal todo CLI on top of src/todos.js.\n\n## Feature 1 - add todos\nPersist new todos to todos.json via src/todos.js; expose "add <text>" in src/cli.js.\nVerify: node src/cli.js add buy-milk then node src/cli.js list shows it.\n\n## Feature 2 - complete todos\nMark a todo done by index in src/todos.js; expose "done <n>" in src/cli.js.\nVerify: adding then completing shows [x] in list output.\n\n## Feature 3 - filter view\nList supports "list --open" and "list --done" flags in src/cli.js.\nVerify: completed items appear only under --done.\n' > PLAN.md
```

Expected:
- Produces `SPRINT.md` with a battle order and numbered slices.
- Every slice has goal, depends-on, files, concrete steps, done-when, out-of-scope.
- Slices are vertical (each delivers behavior), not layered by file type.
- Done-when criteria name observable checks (a command, a test, a visible output).

## S2 — no plan exists
Setup: empty-ish repo, no PLAN.md/PRD.md. Prompt: "slice this into a sprint."
Expected:
- Refuses to fabricate; states no plan was found.
- Recommends resolving intent first (grill-me or a written plan).

## S3 — vague plan
Setup: PLAN.md says "improve performance and clean up the code."
Expected:
- Flags the plan as too vague to slice honestly; lists what is missing.
- Does not emit artificial slices to look productive.

## Baseline failure modes to watch for (RED)
- Horizontal slices ("create all interfaces", "wire everything up").
- "Done when: the feature works" non-criteria.
- Inventing scope the plan never mentioned.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Sliced the seeded PLAN.md; judge 4/4.

## Battletest record (2026-09-03)

Fixture: `/home/akougkas/eval-temp/harness/test_cutit.py`. S1 reuses this
evals.md's own fixture text verbatim (`src/todos.js` stub + a concrete
three-feature `PLAN.md`); S2 is an empty repo with only a `README.md`; S3 is
a `PLAN.md` that says only "improve performance and clean up the code."
S1 is the primary grading fixture, scored on 12 checks against the raw
JSONL's tool-call/safety-block stream, the actual `SPRINT.md` written to
disk, and the reconstructed final assistant text: zero safety blocks, zero
real `artifact` tool calls, zero `tasks` calls, `SPRINT.md` exists with a
`## Battle order` and 2+ numbered slices, every slice carries all six
required fields (Goal/Depends on/Files/Steps/Done when/Out of scope), no
horizontal-layering red-flag language, slices trace to the plan's concrete
features, done-when blocks are command-shaped and testable, and the final
reply names the path and slice count. S2/S3 are graded on 5 checks each
(zero safety blocks, zero `artifact` calls, no `SPRINT.md` fabricated,
correctly flags the gap, recommends a next step). Primary model
`qwen3.8-27b` on `dynamo` (LM Studio); cross-model confirm on
`ornith1.5-35b-moe` on `mini` (llama.cpp), run against S1 and S2 both.

**The bug found reading the source, confirmed empirically first**: the
frozen skill (0.3.0) had `artifact` in `allowed-tools` and titled Step 3
"Write the artifact." `src/tools/artifact.ts` sets `terminate: true` on
every successful call — the run ends the instant the tool executes, with
no further LLM turn to confirm what happened. v1's run called `artifact`
with `kind: "plan"` and, by luck, an explicit `path: "SPRINT.md"` (the
model inferred this from "honor a caller-supplied path" even though no
caller supplied one) — so the file landed in the right place this time, but
the run still ended mid-sentence ("Writing the sprint:") with no
confirmation reply, and a separate `tasks` call (also off-surface) drew a
real safety block. Score 8/12: missing `reply_mentions_sprint_path`, one
real safety block. Had the model not guessed an explicit path, the default
would have been `.clio-coder/artifacts/PLAN.md` (kind defaults to `plan`,
see `core/artifact-paths.ts`) — wrong file, wrong location, same silent
termination. This is exactly the failure mode the planning category's
tech-spec baseline hit ("called `artifact` for an early exit... instead of
a spec").

| run | model | wall | turns | in / out tokens | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|
| baseline (no skill) | qwen3.8-27b | 76s | 7 | 73.8k / 7.1k | 5 (repeated `ls ".cl"` truncated-path retries, benign) | 5/12 | never invoked `/skill cut-it`; read the plan and module correctly, reasoned to genuinely good vertical slices with real done-when checks in its head, then hit a tool-call loop guard and delivered the entire sprint as **prose in the reply, never wrote `SPRINT.md`** — the exact gap this skill exists to close |
| v1 (frozen 0.3.0) | qwen3.8-27b | 115s | 6 | 68.6k / 11.3k | 1 real (`tasks` refused) | 8/12 | called the real `artifact` tool for Step 3 as titled; terminated the turn immediately after writing, mid-sentence, with no confirmation reply — the artifact-tool bug, confirmed |
| v2 (first hardened cut, 0.4.0) | qwen3.8-27b | 70s | 6 | 72.4k / 6.5k | 0 | 11/12 (12/12 after a grading-regex fix, see below) | used `write` correctly, used the `git` tool in Step 1, reported the path and slice count in the final reply; the one score miss was a test-harness regex that didn't handle a `**Done when** (fresh state...):` label followed by bulleted checks on the next lines — the actual done-when content was already command-shaped and testable, fixed in the harness, not the skill |
| v3 (bash-reflex found) | qwen3.8-27b | — | — | — | 2 (1 benign ENOENT, 1 real: `bash` refused) | 11/12 | reached for `bash` (`node --version`, `find`, `ls -la` chained with `&&`) to survey the toolchain even though `bash` was never in `allowed-tools` and nothing in the body named it explicitly — added the same explicit `bash`-refusal line the `tasks` refusal already had, plus a Red flags entry |
| v4 (final, stable) | qwen3.8-27b | 60s | 6 | 71.4k / 5.6k | 0 | **12/12** | clean run: `context` → `read`/`ls` → `git status` → `write`, self-contained final reply naming path, slice count, and battle order |
| final-s2 (no plan) | qwen3.8-27b | 26s | 4 | 43.2k / 2.0k | 0 | **5/5** | checked all three plan locations plus `git log`/`status`, correctly stopped with no `SPRINT.md` written, cited the skill's own red-flag language, recommended `grill-me` |
| final-s3 (vague plan) | qwen3.8-27b | 28s | 5 | 56.2k / 2.5k | 0 | **5/5** | read the one-line `PLAN.md`, explicitly invoked "the skill's own test" (a testable done-when), listed three concrete missing pieces (object/baseline/target for "performance", a definition of "clean", the absent codebase), stopped without writing `SPRINT.md` |
| final-mini (cross-model, S1) | ornith1.5-35b-moe (mini) | 42s | 6 | 10.8k / 3.2k | 0 | **12/12** | same clean shape on the second model family: `context` → `ls`/`read` → `write`, self-contained final reply |
| final-mini-s2 (cross-model, S2) | ornith1.5-35b-moe (mini) | 19s | 6 | 4.0k / 1.1k | 0 | **5/5** | correctly found nothing to slice, recommended `grill-me`, offered to slice immediately if pointed at a plan |

**Changes** (0.3.0 -> 0.4.0):

1. **`artifact` removed from `allowed-tools`.** cut-it never needs the real
   `artifact` tool — `SPRINT.md` is a plain file, always written with
   `write`. This is the fix for the bug above.
2. **Step 3 retitled** "Write SPRINT.md" (was "Write the artifact") and its
   body now says explicitly "use the `write` tool" and names the
   `artifact`-tool confusion directly, including what it actually does
   wrong (terminal call, wrong default path under `.clio-coder/artifacts/`,
   skips the report-back step).
3. **New Step 4 — Report back**, an explicit final-reply requirement (path
   written, slice count, one-line battle-order summary). Nothing in the
   frozen skill told the model to confirm after writing; every hardened run
   now does.
4. **`## Arguments` contract**, the section the frozen skill never had:
   conversational trigger syntax, how a named path splits between "the plan
   to slice" and "where to write `SPRINT.md`", the no-operator/`ask_user`-
   auto-cancels rule (adapted from `grill-me` 0.5.0's "this run has no
   back-and-forth" framing — stated as a fact about the run, not gated on a
   cancellation response), and — since cut-it's happy path rarely needs a
   question at all — explicit guidance for the one place it plausibly might
   (choosing among several candidate plans/milestones): pick the best one,
   state the alternative, mark `assumed — confirm`, keep going.
5. **`tasks` and `bash` explicitly named as refused**, in Arguments and Red
   Flags, both found empirically: v1's `tasks` call (tracking its own
   steps) and v3's `bash` call (`node --version`, `find`, `ls -la` chained
   with `&&`, to survey the toolchain) were both real safety blocks despite
   neither tool ever having been in `allowed-tools`.
6. Three new Red Flags entries for the concrete failures observed: the
   `artifact`-tool confusion, ending the run with no final reply, and the
   `bash` reflex (`tasks` already had informal coverage, now explicit too).

**Still weak**: `code_nav` (in `allowed-tools`) was never exercised — this
fixture's grounding fit entirely in one small stub file, so `read`/`grep`
sufficed; a plan referencing a larger call graph might exercise it, none
was built here. The `ask_user`-unavailable path and the "several candidate
plans" disambiguation guidance in Arguments are reasoned prose, not
empirically run — no fixture here seeds multiple plausible plan files or a
scenario where the model actually reaches for `ask_user`; every hardened
run reasoned straight to the assumed-confirm default or never needed a
question at all. S3's "flag as vague" grading is phrase-matching against a
fixed term list (`vague`, `underspecified`, `insufficient`, ...) — a run
that flags the same gap in different words would under-score on a
technicality, though neither observed run did. The v2 grading-regex miss
(`done_when_testable`, fixed in the harness before v4) is a reminder that
this fixture's automated score can undercount a genuinely correct skill
output; the raw `SPRINT.md` files are worth spot-reading, not just the
score column. No timing was captured for v3 (an incremental re-run to
confirm the `bash` finding, superseded immediately by v4) — not a gap in
the skill's coverage, just an artifact of the iteration order.
