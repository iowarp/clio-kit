# Evals — prd

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet.

## S1 — brain dump to spec
Setup: Okay, brain dump. Our users keep asking for saved searches — they
build the same filter combos every day and lose them on refresh. I am
thinking they can name a search, get it in a sidebar list, maybe share a
link to it with a teammate. There is also the question of limits, do free
users get three saved searches or unlimited, and should a shared link work
for someone without an account? I have not decided. Somewhere down the
line I would also love alerts, like notify me when a saved search has new
results, but that might be its own thing. Spec this out.

Fixture:
```bash
mkdir -p src
printf '{\n  "name": "search-app",\n  "version": "1.0.0",\n  "private": true,\n  "dependencies": {\n    "typescript": "5.5.0"\n  }\n}\n' > package.json
printf 'export function runSearch(filters: string[]): string[] {\n  return filters;\n}\n' > src/search.ts
```

Expected:
- Proceeds phase by phase; each phase confirmed before the next opens.
- Detects the stack from the repo and confirms rather than asking cold.
- Produces PRD.md plus milestones/N-slug/prompt.md files.
- Each milestone prompt is executable without the PRD open (self-contained).

## S2 — scope honesty
Setup: brain dump implies ten features. Prompt: "spec this out for v1."
Expected:
- Proposes 4–8 core features and pushes the rest to an explicit out-of-scope list.
- Out-of-scope entries are specific to this product, not boilerplate.

## S3 — existing foundation
Setup: repo already implements half the data model the idea needs.
Expected:
- Inventories the existing entities in phase 5/7 and reuses them.
- Does not spec already-built functionality as new work.

## Baseline failure modes to watch for (RED)
- One giant questionnaire instead of phase-gated exchanges.
- PRD written without any user confirmation of features/scope.
- Milestone prompts that are headings with no executable content.
