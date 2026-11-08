# Evals — archify

RED-GREEN scenarios (subagent WITHOUT the skill vs WITH). Pass/fail per
bullet. Status: scenarios recorded, not yet executed against a model
(`eval-status: scenarios-recorded`).

Both scenarios assume the skill is installed at project scope
(`.clio-coder/skills/archify/`) and run from the project root.

## S1 — describe a system

Setup: empty git repository. Prompt: "Draw the architecture: a browser
talks to an API, the API uses Redis as a cache and PostgreSQL as its
store."

Expected:
- Chooses `architecture` and reads exactly one schema plus one example
  before writing.
- The very next tool action writes
  `.clio-coder/artifacts/maps/<name>.architecture.json` with four
  components (`frontend`, `backend`, `database`, `database`), three
  connections, and `meta.quality_profile: "showcase"`.
- Runs `validate architecture ... --quality showcase --json`; the receipt
  reports 0 errors and 0 warnings.
- Runs `deliver` once; exit 0; `.clio-coder/artifacts/maps/<name>.html`
  exists and the receipt carries a SHA-256 for both spec and artifact.
- Runs `verify(check="frontend", path=<html>)` and reports its result.
- Never invokes `scripts/check-update.mjs` and never runs `visual-check`
  unless asked; if it does run it on a node without Chrome, reports
  `viewer/chrome-unavailable` as a skipped check, not a pass.

RED (baseline without the skill): draws Mermaid or ASCII in the reply,
never writes a JSON spec, never validates, and claims a diagram that does
not exist on disk.

## S2 — map this repository

Setup: a small fixture repository (about 15 TypeScript files across
`src/cli`, `src/domains/store`, and `src/domains/api`) with a GitHub
`origin` remote, a committed `HEAD`, and a codewiki index already built by
`clio-coder context index`. Prompt: "Map this repository."

Expected:
- Runs `clio-coder context map --json` before authoring anything.
- Keeps `meta.repository` (when present) and every seeded `sources` entry;
  adds line ranges only from `code_nav mode=outline`.
- Component count stays at or below 12; the `store` area carries
  `type: "database"`.
- Runs `validate architecture <seed> --repo-root . --quality standard
  --json`; 0 errors.
- Runs `deliver ... --repo-root .`; exit 0; the HTML exists under
  `.clio-coder/artifacts/maps/`.
- If the index is missing, tells the user to run `clio-coder context
  index` instead of inventing components.

RED (baseline without the skill): lists directories from `ls` as
components with invented edges, no line evidence, no `--repo-root`
verification, and no delivered HTML.

## Baseline failure modes to watch for

- Authoring coordinates in prose before the first candidate.
- Reading renderer source before the first candidate.
- Editing the spec after the final passing validation.
- Reporting a non-zero `deliver` or a missing Chrome as success.
