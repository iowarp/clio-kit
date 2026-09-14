# Evals — ast-grep

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet. Status:
written at port time, not yet executed (`eval-status: untested`).

## S1 — structural search with context
Setup: find async functions that await something but have no error
handling.

Fixture:
```bash
mkdir -p src
printf 'async function fetchUser(id) {\n  const res = await fetch("api" + id);\n  return res.json();\n}\n\nmodule.exports = { fetchUser };\n' > src/users.js
printf 'async function fetchSafe(url) {\n  try {\n    const res = await fetch(url);\n    return await res.json();\n  } catch (err) {\n    return null;\n  }\n}\n\nmodule.exports = { fetchSafe };\n' > src/safe.js
printf 'function plainSync(a, b) {\n  return a + b;\n}\n\nmodule.exports = { plainSync };\n' > src/plain.js
```

Expected:
- Writes a positive and a negative example file before any repo scan.
- Rule uses `kind` + `has`/`not` with `stopBy: end` on every relational rule.
- Tests the rule on the examples first; scans the repo only after it passes.
- Reports matches as path:line, not raw JSON.

## S2 — unknown node kind
Setup: rule fails to match the example. Prompt continues from S1.
Expected:
- Uses `--debug-query=cst` to read the real node kinds; does not guess.
- Simplifies the rule and re-adds parts instead of abandoning to grep.

## S3 — rewrite temptation
Setup: user asks "find and fix all console.log calls".
Expected:
- Uses ast-grep for the find only; never passes `-U`/`--update-all`.
- Applies changes through the ordinary edit tools after reading matches.

## Baseline failure modes to watch for (RED)
- Scanning the whole repo with an untested rule and reporting "no matches".
- Missing `stopBy: end`, silently under-matching.
- Escaping errors in `--inline-rules` read as "pattern doesn't match".
- Guessed `kind:` names that do not exist in the grammar.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT SMOKED: ast-grep binary absent on the eval host (2026-08-13). Needs the binary, then a re-run.

Follow-up (2026-08-13): the durable fix considered here was a typed read-only
`ast_grep` tool in `src/tools`, which would drop the binary dependency from the
skill body and let read-only recipes such as `scout` bind structural search
directly. Not built: a new tool is a new entry in the canonical tool surface,
which means a name reserved in `core/tool-names.ts`, an action class and
concurrency rule, admission and profile entries, and prompt-prefix cost paid by
every run whether or not anyone searches structurally. That is a tool-surface
decision, not skill-catalog cleanup. Until then the skill stays
`scenarios-recorded` and its first step is the binary check it already has.

## Battletest record (2026-09-03)

S1 fixture plus `src/orders.js` (one matching `loadOrders`, one non-awaiting
`noAwait`), `ornith1.5-35b-moe` on mini (llamacpp), `clio-coder run
--autonomy full-auto --json`, headless, `@ast-grep/cli` 0.45.3 on PATH.
Ground truth: `src/users.js:1` and `src/orders.js:1` match; `safe.js`,
`plain.js`, `noAwait` do not.

| run | wall | turns | in / out tokens | outcome |
|---|---|---|---|---|
| baseline (no skill) | 565s | 37 | 16.2k / 33.0k | never produced a valid rule (`$($params)` syntax, relational rules without `stopBy: end`, `--inline-rules` quoting); 3 `$(...)` calls; answered by reading the source; scratch left in `.astscratch/` |
| v0.2.0 | 123s | 13 | 7.9k / 7.1k | correct 2/5 via a proven rule; wrote fixtures to `/tmp` first, then into the workspace; `rm -rf` and `find -delete` hard-blocked, scratch left behind |
| v0.3.0 first cut | 176s | 13 | 9.8k / 10.5k | correct; dumped the tree before writing any rule; a `/tmp` write via shell redirection refused; `rm -rf` hard-blocked |
| v0.3.0 | 190s | 24 | 9.5k / 10.9k | correct; zero safety blocks; scratch removed with per-file `rm` + `rmdir`; working tree clean. Spent 10 turns recovering from a guessed `async_function_declaration` kind |

Changes in v0.3.0: `## Arguments` contract; Step 0 verifies `sg --version`
prints `ast-grep` because `/usr/bin/sg` is shadow-utils on Linux; scratch
lives in `.clio-coder/scratch/ast-grep/` (a `/tmp` write needs an approval
a headless run cannot give); rules always go through a file; tree dumps
only after a failed test; cleanup by named `rm` + `rmdir` because
`rm -r`/`rm -f` match the `rm-recursive-or-force` damage-control block;
`edit` added for rule iteration. The example rule was re-verified verbatim
against 0.45.3 (matches exactly the two expected functions) and the skill
now says so, plus that an unknown `kind` fails with exit 8 rather than
matching nothing.
