---
name: clio-coder-dev
description: 'Use when this workflow is requested: Governs changes to Clio Coder''s
  own source and harness (TUI, skills, agents, tools, prompts, domains): what may
  change freely, what needs explicit user intent, and how to change it without breaking
  the architecture. Not for verifying a change against the harness; use clio-coder-test.
  Triggers on "modify Clio Coder", "change the Clio harness".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/meta/clio-coder-dev
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Clio Dev (self-development)

Working inside Clio Coder's own source tree is ordinary repository work with
one extra discipline: the contribution boundary. This skill governs that
boundary and the change workflow.

**REQUIRED SUB-SKILL:** `clio-coder-test` for test mechanics (which layer to run,
the hot-reload loop). This skill decides *whether* a change may leave the
machine; `clio-coder-test` decides *how* to verify it.

## The contribution boundary

Two categories. They are not the same:

- **Local development and testing**: editing source, running tests,
  reconfiguring the local install, dogfooding skills, making a local commit
  when the user asked for the work. **Permitted freely.**
- **Contribution to the shared project**: pushing, opening PRs, publishing
  releases, tagging, or altering git remotes. **Requires explicit user intent,
  every time.** No exception for "tiny" changes.

### STOP — red flags

Any of these thoughts means you are about to cross the boundary. Stop,
validate locally, report, and ask:

- "The change works, I'll just push it so we're done."
- "It's a tiny PR, I'll open it real quick."
- "Let me tag a release / bump the version while I'm here."
- "I'll commit and push so the next session has it."
- "We're out of time, ship it."

### Rationalization table

| Excuse | Reality |
|---|---|
| "Pushing is the obvious next step." | It is the user's step. Local is done; stop there. |
| "The user clearly wants it shipped." | "Clearly" is an assumption. Get explicit intent. |
| "A commit isn't a push." | A local commit is fine; never push or open a PR without intent. |
| "Tagging is harmless." | Releases/tags/remotes are contribution. Out of bounds. |
| "I'll just fix the remote/branch quickly." | Altering remotes is never implied work. Ask. |

## Self-development workflow

Follow in order for every change:

1. **Classify the touched surface.** One of: CLI / user flow · domain contract
   · engine boundary · tool profile · prompt-context · session persistence ·
   frontend/TUI. The surface determines which contract and tests matter.
2. **Read the contract and tests before editing.** Open the domain's
   `contract.ts` / `index.ts` and its `tests/contracts/*` file first.
3. **Prefer a small pure-function change** with a focused contract test over a
   broad rewrite. Side effects live in `extension.ts`; testable policy lives
   in sibling pure modules.
4. **Respect all six boundary invariants.** Pi SDK imports stay in
   `src/engine/**`; worker value imports from domains stay within the declared
   provider rehydration seams; domains never import another domain's
   `extension.ts`; tools never import the interactive layer;
   `src/interactive/turn-*.ts` and `chat-loop.ts` never import entry modules;
   Stage 0 is entered only through declared seams. Add or reuse a contract
   instead.
5. **Validate narrowly, then report.** Run the narrowest meaningful layer per
   `clio-coder-test`, then state exactly what ran and what remains unverified. Done
   when the report names both.

## Source is truth

- `src/domains/**` is the product architecture, `src/engine/**` the pi-ai
  adapter boundary, `src/tools/**` the model-visible action surface. A tool or
  contract change ripples into safety, dispatch, ACP, and telemetry: check the
  consumers, not just the file you edited.
- `CLIO-CODER.md` is the audited constitution; codewiki and the `.clio-coder/state.json`
  fingerprint are mutable hints. Never trust a stale summary over source. If
  source topology changed, refresh via `clio-coder context init` — but a regenerated
  `CLIO-CODER.md` is contribution-adjacent; do not commit it without intent.

## Continuity

Pair with the session bookends: `context-prime` to orient before self-dev
work, `context-handoff` to brief the next session when a change spans
sessions.
