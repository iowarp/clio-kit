# Evals — tech-spec

Baseline scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: written at port time, not yet executed (`eval-status:
untested`); skill is marked provisional and user-invoked only.

## S1 — sufficient context (Path A)
Setup: write a tech spec for adding per-client rate limiting to the ingest
endpoint in src/ingest.js; limits should be configurable and violations
must return 429 without dropping the connection.

Fixture:
```bash
mkdir -p src
printf 'const http = require("http");\nconst { parseEvent } = require("./parse-event.js");\n\nconst INGEST_ROUTE = "ingest";\n\nfunction createIngestServer(store) {\n  return http.createServer((req, res) => {\n    if (req.url.slice(1) !== INGEST_ROUTE || req.method !== "POST") {\n      res.writeHead(404);\n      res.end();\n      return;\n    }\n    let body = "";\n    req.on("data", (chunk) => { body += chunk; });\n    req.on("end", () => {\n      const parsed = parseEvent(body);\n      if (parsed.err) {\n        res.writeHead(400);\n        res.end(JSON.stringify({ error: parsed.err }));\n        return;\n      }\n      store.append(parsed.value);\n      res.writeHead(202);\n      res.end();\n    });\n  });\n}\n\nmodule.exports = { createIngestServer };\n' > src/ingest.js
printf 'function parseEvent(raw) {\n  try {\n    const value = JSON.parse(raw);\n    if (typeof value.client !== "string") return { err: "missing client" };\n    return { err: null, value };\n  } catch {\n    return { err: "invalid json" };\n  }\n}\n\nmodule.exports = { parseEvent };\n' > src/parse-event.js
```

Expected:
- Local conventions inspected before any pattern is proposed.
- 2-3 materially different alternatives compared before recommending.
- Typed contracts for every new/changed boundary; call stacks with data
  flow; file map; vertical TDD plan.
- Nothing implemented; spec returned inline (no file unless asked).

## S2 — thin context (Path B)
Setup: user asks for a spec with only a vague sentence.
Expected:
- States context is insufficient; interviews one question at a time with
  recommended answers; explores the codebase instead of asking what code
  answers.
- Converts to Path A only when context suffices.

## S3 — unknowns
Setup: a dependency's behavior is genuinely unknown.
Expected:
- Recorded as an open question, not filled with plausible design.

## Baseline failure modes to watch for (RED)
- Prose-only spec with no typed contracts or call stacks.
- Single foregone design presented as "the" answer.
- Implementation snuck in.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. PASS. Spec written and its claims exercised with node -e; judge 4/4.

## Battletest record (2026-09-03)

Fixture: `/home/akougkas/eval-temp/harness/test_techspec.py`, continuing the
planning category's shared HPC log-triage domain (`product-intent` -> `prd`
-> `tech-spec`). Seeds a plausible root `PRD.md` (purpose, features,
out-of-scope, stack, integrations, data model, milestones, and the
always-on-vs-on-demand ingestion tension explicitly marked as *not this
document's decision*) plus the existing partial codebase (`src/scanner.py`:
a working `FailureEvent` + `scan_oom`, OOM only) and two sample dmesg logs
carrying real OOM/ECC/Xid line formats, inside a git repo. Three task
variants, one fixture:

- **base** (S1, Path A): "spec ECC + Xid detection and cross-signature
  top-3 ranking" — sufficient context, no save request. Graded on 9 checks
  against the *reconstructed final assistant text* (this skill's default
  output is inline, not a file): all 11 outline-derived sections present,
  >=5 domain grounding terms, >=2 materially different alternatives,
  `FailureEvent` reused not respecced, nothing implemented (`scanner.py`
  byte-identical to seed), the ingestion trade-off left unresolved, zero
  safety blocks, no `tasks` call, and — the check this run exists to catch —
  **no file written when nothing asked for one**.
- **save** (S1 variant, confirmation only, run once on the final version):
  same task plus an explicit "save it to docs/tech-spec-log-triage.md" —
  10 checks, same 8 plus the file existing at exactly that path and no
  other new file appearing.
- **thin** (S2, Path B, confirmation only, run once on the final version):
  a genuinely vague "improve our failure detection, you'll need to ask me
  stuff" request with 5 lighter checks — zero safety blocks, no silent
  stall, no `tasks` call, and either a real `ask_user` exchange or the
  assumed-confirm monologue (`assumed` + `confirm` both present), with a
  real spec still produced.

| run | model | wall | turns | in / out tokens | safety blocks | score | outcome |
|---|---|---|---|---|---|---|---|
| baseline (no skill) | ornith-1.5-35b-a3b | 80s | 5 | 54.9k / 12.1k | 0 | 4/9 | never invoked `/skill tech-spec`; discovered the installed skill itself via `context(scope="skills")`, read its SKILL.md directly, then called `artifact` and terminated early with a `.clio-coder/artifacts/PLAN.md` instead of a spec — no alternatives, no sections, wrong output shape |
| v1 (frozen 0.2.0) | ornith-1.5-35b-a3b | 95s | 8 | 115.0k / 15.1k | 1 | 6/9 | ran Path A correctly and produced a genuinely strong spec (11/11 sections, 3 material alternatives, `FailureEvent` reused, nothing implemented) but opened a `tasks` plan (refused, safety block) and **wrote the spec to `docs/tech-spec-scanner-ecc-xid.md` without being asked to** — the exact Path-A/B default-output risk flagged going in |
| v2 (live 0.3.0) | ornith-1.5-35b-a3b | 68s | 5 | 57.2k / 10.4k | 0 | 9/9 | same spec quality, zero safety blocks, no `tasks` call, correctly returned inline with no file written; final text states explicitly "I did **not** write a file, since nothing in the request asked to save it" |
| v2 confirm — save | ornith-1.5-35b-a3b | 110s | 11 | 184.8k / 16.5k | 0 | 10/10 | explicit "save it to docs/tech-spec-log-triage.md" correctly produces exactly that file at that path, nothing else |
| v2 confirm — thin (Path B) | ornith-1.5-35b-a3b | 69s | 9 | 111.8k / 11.7k | 0 | 5/5 | correctly identified insufficient context, ran Path B, and carried all five scope decisions (S1-S5) through as an explicit assumed-confirm monologue headlessly instead of stalling or silently skipping to Path A |

**Changes**: (1) `## Arguments` contract with the slash-invocation syntax,
what's required vs. inferred, and — the section that mattered most here —
an explicit "output defaults to inline" rule stated as its own bullet
before the headless-monologue prose, so the fix for Path B's ask_user gap
can't be misread as license to always write a file; (2) the headless
no-operator paragraph, ported from `product-intent`/`prd`, applied to Path
B's grill-me interview: every question runs as state-question /
grounded-recommendation / reasoning / adopt / mark `assumed — confirm`,
end to end, not just the first one; (3) explicit `tasks` and `bash`
refusal lines — `tasks` was v1's only safety block; (4) `Done when` and
`Red flags` both gained a line naming the unrequested-file failure and the
unanswered-Path-B-question failure by name, plus the existing `tasks`/`bash`
refusal repeated as a red flag (matching `prd`'s and `product-intent`'s
pattern of naming the exact observed failure, not a generic reminder).
Version 0.2.0 -> 0.3.0.

**Still weak**: per this pass's coordinator note, no secondary-model
confirmation was run (qwen3.8-27b was skipped in favor of running one full
cycle on ornith-1.5-35b-a3b at speed, concurrently with a sibling agent
hardening `architecture` on `mini`); the fix is validated on one model
class only. The baseline's failure mode (discovering and improvising from
the installed skill file directly, without ever invoking it, then calling
`artifact` for an unrelated early exit) is a skill-selection/tool-scoping
gap this SKILL.md cannot fix from inside its own body. `code_nav` (in
`allowed-tools`) was never exercised — the fixture's one-file codebase
never needed it. `requires: [skill:tdd]` is a diagnostic-only reference in
this harness (unmet requires warn, never block `--skill`-path invocation);
the TDD Test Plan section reads fine without the `tdd` skill installed, but
that was not tested with `tdd` actually present to see if the reference
changes. Genuine unknowns (S3 from the original evals) were exercised only
incidentally via the Xid-severity and ECC-correctable open questions, not
as an isolated scenario.
