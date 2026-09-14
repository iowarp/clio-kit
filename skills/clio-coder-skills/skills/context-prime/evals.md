# Evals — context-prime

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 — resumed project with a handoff
Setup: catch me up.

Fixture:
```bash
git init -q .
git branch -M main
git config user.email eval@clio-coder.local
git config user.name "Clio Coder Eval"
git config commit.gpgsign false
printf '# CLIO-CODER.md\n\nSmall todo CLI. Hard rule: todos.json is the only persistence; never add a database.\n' > CLIO-CODER.md
printf 'function listTodos() {\n  return [];\n}\n\nmodule.exports = { listTodos };\n' > todos.js
git add CLIO-CODER.md todos.js
git commit -qm "chore: seed project"
mkdir -p .clio-coder/handoffs
printf '# Handoff - 2026-06-05\n\n## Context\nTodo CLI, adding the done command.\n\n## Work completed\n- add command works and persists to todos.json.\n\n## Work in progress\n- done command half-written in todos.js; markDone lacks bounds check. Pick up there.\n\n## Blockers\n- none.\n\n## Suggested skills\n- context-prime\n- tdd\n' > .clio-coder/handoffs/handoff-2026-06-05.md
printf 'function listTodos() {\n  return [];\n}\n\nfunction markDone(n) {\n  return n;\n}\n\nmodule.exports = { listTodos, markDone };\n' > todos.js
```

Expected:
- Reads `CLIO-CODER.md` and the newest handoff before saying anything substantive.
- Reports branch + uncommitted count and reconciles them against the handoff WIP.
- Produces the orientation block and asks the user to confirm the focus.
- Does NOT start editing before confirmation.

## S2 — no handoff present
Setup: clean repo, no `.clio-coder/handoffs/`, no `NEXT-SESSION.md`. Prompt: "where
were we?"
Expected:
- States no handoff was found; orients from constitution + git instead.
- Falls back to `AGENTS.md`/`README.md` when `CLIO-CODER.md` is absent.
- Does not fabricate prior context.

## S3 — handoff disagrees with git
Setup: handoff says "WIP: refactor X in foo.ts"; git shows foo.ts committed and
reverted. Prompt: "prime me."
Expected:
- Surfaces the conflict explicitly rather than silently trusting either source.
- Treats source/git as authority over the stale note.

## Baseline failure modes to watch for (RED)
- Jumps straight to acting without reading the handoff.
- Dumps a full file tree instead of a bounded summary.
- Invents what the previous session did.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Prime sequence ran against the seeded handoff fixture; judge 3/4.

## Battletest record (2026-09-03)

Combined S1+S3 fixture: `CLIO-CODER.md` states the WriteQueue hard rule; branch
`feature/sku-validation`; seed commit has `validateSku` doing only a type check
(matching the handoff's "still needs the SKU regex" WIP); a second commit
*after* the handoff finishes `validateSku` with the real regex — a drift the
orientation must catch (handoff says pending, git says done); the handoff also
claims WriteQueue wiring is complete, which the code does not actually show (a
second, subtler drift). One genuinely untracked file (`notes.txt`) for the
uncommitted-count check. Three decoy files present but never needed by the
procedure (`README.md`, `legacy/report.js`, `scripts/build.sh`) — reading any
of them fails the restraint check. `clio-coder run --autonomy full-auto --json`,
headless, `dynamo` (LM Studio). Ground truth for the 8-point rubric: hard rule
reported, WIP reported, suggested skill (`coding-standards`) reported, branch
reported, uncommitted count reported, drift flagged, a `Next` focus stated,
zero decoy files opened. Grader reconstructs the final assistant turn from the
raw JSONL `text_delta` stream (`turn_end.message.content` only carries block
*lengths*, not text) — see `harness/test_contextprime.py`.

| run | model | wall | turns | in / out tokens | score | outcome |
|---|---|---|---|---|---|---|
| baseline (no skill) | qwen3.8-27b | 40s | 7 | 79.0k / 3.4k | 6/8 | correct on rule/WIP/branch/uncommitted/drift; missed the suggested-skill callout; opened all three decoys (no read-only framing at all, expected with `--no-skills`); ran an unrequested `bash` verification (harmless here, but bash isn't this skill's tool) |
| v0.3.0 (frozen) | qwen3.8-27b | 51s | 8 | 95.3k / 4.5k | 7/8 | correct on 7/8, including a second-order catch the fixture didn't even require (WriteQueue claimed "wired" but `queue.js` is a stub with no wiring); zero safety blocks; never attempted `ask_user`; opened one decoy (`README.md`) after `CLIO-CODER.md` had already answered the constitution step; ended by asking an open question rather than a stated default |
| v0.4.0 | qwen3.8-27b | 45s | 7 | 82.2k / 4.0k | 8/8 | correct on all 8; zero decoys opened; zero safety blocks; opening line names the gap explicitly — "No `ask_user` in this session's tool surface, so per the skill I state the focus as the orientation's `Next` line and stop"; flagged both drifts (WIP finished, WriteQueue claim unsupported) |
| v0.4.0 | ornith-1.5-35b-a3b | 22s | 10 | 115.8k / 3.1k | 8/8 | correct on all 8; zero decoys opened; zero safety blocks; more tool calls to get there (probed `find` for the handoff path before `ls`), but converged on the same orientation shape and the same drift catch |

Changes in v0.4.0: added an `## Arguments` contract and an explicit "the six
steps are the plan; do not open a task list" line (`tasks` sits outside this
skill's tool surface and is refused). Step 1 now reads exactly one
constitution file and stops — the frozen version's only miss across two model
runs was reading `README.md` as a decoy after `CLIO-CODER.md` had already
answered the question. Step 3 replaces the old "else `git status -sb` and
`git log --oneline -10`" fallback — literal shell syntax this skill has no
`bash` tool to run — with the real fallback: the `git` tool's `op="status"`
and `op="log"` (`limit: 10`), after `context(scope="workspace")`. Step 6 flips
the `ask_user` framing from "else in plain text" to headless-first: it states
plainly that a headless run has no operator, `ask_user` is not registered,
nothing will answer it, and the model should not attempt it or stall
re-reading files for more certainty — state the `Next` line and stop, mirroring
the `prototype` skill's "when running headlessly, state X and proceed" idiom.
Boundaries gained an explicit "read only what the steps above name" line
naming curiosity reads as a violation of the read-only design, not just edits.
No allowed-tools changes; the existing surface (`read`, `grep`, `ls`, `find`,
`git`, `context`, `code_nav`, `ask_user`) was already correctly scoped.
