---
name: skill-craft
description: 'Use when this workflow is requested: Writes, reviews, or prunes a SKILL.md,
  judging whether its description, body, and length earn their cost. Not for packaging
  a workflow that just happened; use workflow-distiller. Triggers on "write a SKILL.md",
  "create a new skill".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/meta/skill-craft
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Skill Craft

A skill exists to wrangle predictability out of a stochastic system: the same
process every run, not the same output. Every rule below is a lever on that
one virtue. A disputed line is settled by running the skill, not by debate;
in Clio that means `clio-coder eval skill`, and a shipped skill's `evals.md`
Expected bullets are its completion criteria.

## Format and Location

A skill is a folder holding a `SKILL.md`. Author with ordinary write/edit tools
in a draft directory such as `draft-skills/<name>/`, outside active skill and
installed package roots. In this repository, curated sources live under
`library/skills/<category>/<name>/`. Validate a draft with
`clio-coder library validate draft-skills/<name>/SKILL.md`.

For distribution, use the portable envelope in
`library/_authoring/templates/skill/` and validate the complete package root.
The operator installs it through the Library into the user or project package
store (`<configDir>/plugins/` or `.clio-coder/plugins/`). Loose skills under
`.clio-coder/skills/` and `<configDir>/skills/` are also operator-managed; Clio's
model tools cannot author directly in those protected roots. Other hosts may
consume the same `SKILL.md` through their own installation routes.

Frontmatter contract (Agent Skills compatible):

- `name`: lowercase-hyphen, ≤64 chars. Catalog convention requires it to match
  the folder; the runtime loader warns on a mismatch and uses the frontmatter
  name.
- `description`: required, ≤1024 chars; its craft is the section below. Quote
  it when it contains ` #` — an unquoted YAML scalar is truncated there.
- `version` and `license`: required for catalog publication. Follow the
  marketplace versioning policy for body, trigger-surface, and sibling-file
  changes; the normalized `SKILL.md` hash handles drift separately.
- `disable-model-invocation: true`: hides the skill from the agent; only the
  user can activate it.
- `allowed-tools` / `disallowed-tools`: a *narrowing* declaration, not a
  grant. While every loaded skill declares `allowed-tools`, the turn's tool
  surface shrinks to their union; `disallowed-tools` always denies. Neither
  ever admits a tool the safety level would refuse. Canonical lowercase Clio
  names only (see library/skills/README.md, "Claude Code interop").
- `requires`: `skill:<name>` dependencies; the loader warns when one is
  missing. Reference an installed skill by name instead of restating its job.
- `clio-coder:`: the reserved publication block (`registry-id`, `source-url`,
  `audit`, `provenance` designed|adapted|imported with `origin` when not
  designed, `eval-status`, optional `model-size` and `agents`). Required for
  catalog skills; approval is judged against it (library/skills/README.md).

Sibling files (`references/*.md`, `evals.md`, scripts) ride along in the
folder and load only when the body points at them.

## Invocation: Choose Which Load You Pay

A model-invoked skill's description sits in the agent's context every turn:
that is context load, paid forever. A hidden skill costs nothing in context
but the user must remember it exists: cognitive load. Keep a description only
when the agent must reach the skill on its own or another skill requires it;
if it only ever fires by hand, set `disable-model-invocation: true`.

## Description and Triggers

Two frontmatter fields share the trigger surface, and every word in both
competes with every other skill's:

- `triggers`: the phrases a user actually types, one per distinct branch.
  Synonyms restating one branch ("API key", "credential", "token") are
  duplication; keep the strongest.
- `description`: one lead sentence saying what the skill does, then one
  "Not for X; use <other-skill>" clause per real boundary. It is what the
  agent reads in the `context(scope="skills")` listing on every skill-shaped
  turn, so it carries no trigger list and no identity the body already
  states.

## Body: Steps and Reference

Two content types, mixing freely. Steps are ordered actions; reference is
material consulted on demand. Rank by how immediately the agent needs each
piece: steps first, in-file reference second, sibling-file reference behind a
pointer last.

- End every step on a checkable, exhaustive completion criterion: "every
  modified file accounted for", never "understanding reached". A vague bound
  invites the agent to declare done early.
- Keep one concept's definition, rules, and caveats under one heading, not
  scattered.
- Push material only some runs need into `references/<topic>.md`. The
  pointer's wording decides whether the agent follows it: state the condition
  ("when the tests use the mock provider, read references/harness.md"), not
  just the link.
- Prefer one pretrained word over a restated sentence: "fast, deterministic,
  low-overhead" collapses to "tight". A strong word anchors the same behavior
  every time it appears; a weak one ("be thorough") changes nothing and
  should be deleted or strengthened ("relentless").

## Pruning

Before finishing any skill, pass the body line by line:

- Relevance: does the line still bear on what the skill does? Stale layers
  accumulate because adding feels safe and removing feels risky; remove.
- Duplication: each meaning lives in exactly one place. A repeated meaning
  also inflates its apparent importance.
- No-op: would the agent do this by default? Then the line buys nothing;
  delete the whole sentence, not words within it.
- Sprawl: past ~120 body lines, either disclose reference into sibling files
  or split the skill; a split description must earn its permanent context
  load.

Done when: the description is one sentence plus routing clauses and the
triggers carry the phrases, every step has a checkable
completion criterion, no line fails the relevance/duplication/no-op pass, and
an `evals.md` records at least one RED-GREEN scenario distinguishing
with-skill from without.
