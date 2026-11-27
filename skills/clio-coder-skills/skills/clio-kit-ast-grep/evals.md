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
