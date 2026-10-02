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
