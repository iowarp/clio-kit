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
