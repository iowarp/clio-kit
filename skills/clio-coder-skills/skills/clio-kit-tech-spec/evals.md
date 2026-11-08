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
