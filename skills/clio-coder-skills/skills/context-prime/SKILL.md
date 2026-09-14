---
name: context-prime
description: 'Use when this workflow is requested: Orients a fresh session in a repository
  by loading the last handoff, git state, the project constitution, and active-work
  signals before acting. Not for writing the handoff; use context-handoff. Triggers
  on "prime this repository", "catch me up".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/context/context-prime
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Context Prime

Reconstruct the state a previous session left behind, in one bounded pass,
before taking any non-trivial action. A fresh agent has the code but not the
*intent*; this rebuilds the intent.

This is the read-side bookend of `context-handoff`. It consumes what `context-handoff` wrote.
It is not `/resume` — that restores an engine session's transcript; this
reconstructs orientation a transcript alone doesn't carry.

## When to use

- The first substantive request of a new session.
- Resuming a project after time away, or picking up another agent's work.
- The user says "catch me up", "where were we", "prime", "get up to speed".

Skip it for a one-line question in a repo you already have full context on.

## Arguments

```text
/skill context-prime [<focus hint>]
```

- With a focus hint: treat it as a candidate for the orientation's `Next`
  line, not as ground truth — confirm or contradict it against the handoff
  and git state, the same as any other suggested focus.
- Without one: orient from the handoff, constitution, and git state alone.

The six steps below are the plan; do not open a task list for them. `tasks`
sits outside this skill's tool surface and any call to it is refused.

## Procedure

Work top to bottom; stop early once you have enough to state where things stand.

1. **Constitution.** Read exactly one: the first of `CLIO-CODER.md`,
   `AGENTS.md`, `CLAUDE.md`, `README.md` that exists, in that order. Note
   hard invariants and workflow rules, then stop — do not also open the
   others "for completeness"; a fallback file is read only when every
   name ahead of it is absent.

2. **Last handoff.** Read the newest `.clio-coder/handoffs/handoff-*.md`; if none,
   fall back to `NEXT-SESSION.md` at the repo root. This is the previous
   session's brief: focus, work-in-progress, blockers, suggested skills.

3. **Git state.** `context(scope="workspace")` already carries a git
   snapshot; read it first. Fill any gap with the `git` tool directly:
   `op="status"` for branch and dirty files, `op="log"` (`limit: 10`) for
   recent commits. `bash` is not in this skill's tool surface — there is no
   shell fallback, "run `git status` yourself" is never the move here.
   Reconcile against the handoff's "work in progress": flag anything that
   drifted — work committed since the handoff, work reverted, a WIP item
   that is now finished, or a "completed" claim the code plainly doesn't
   back up.

4. **Active signals.** Check `.clio-coder/state.json` and codewiki freshness if
   present. Treat stale summaries as hints, never as authority over source.
   A v1 or missing `.clio-coder/codewiki.json` may rebuild on demand through
   codewiki tools, but source files remain authoritative.

5. **Skills.** List installed skills with `context(scope="skills")` and note
   any the handoff suggested for the next step; do not scan the filesystem
   for them.

6. **Orient and confirm.** Produce the short orientation (template below)
   ending with the focus to confirm. `ask_user` is only registered in an
   interactive session with an operator present; call it there, offering
   the handoff's suggested focus as the first option. **A headless run has
   no operator: `ask_user` is not registered, so any call is refused as an
   unregistered tool rather than answered.** If it is not among your available tools, do not
   attempt it and do not keep re-reading files hoping for more certainty
   first — state the focus as the orientation's `Next` line, in plain
   text, and stop; that written statement is the confirmation for this
   run. If the handoff and git state disagree, surface the conflict rather
   than picking silently.

## Orientation template

```markdown
## Session orientation — [YYYY-MM-DD]

- **Project**: [name] · branch `[branch]` · [N] uncommitted file(s)
- **Last session focus**: [from handoff, or "no handoff found"]
- **State**: [1–2 lines: what's done, what's mid-flight]
- **Next**: [the focus to confirm]
- **Blockers / open questions**: [if any]
- **Suggested skills**: [from handoff, or omit]
```

## Boundaries

- Bounded by design: summarize and reference by path; do not dump file trees or
  copy long documents into context. Read only what the steps above name — the
  one constitution file that wins the fallback order, the newest handoff, git
  and context state, `.clio-coder/state.json`/codewiki freshness. A source
  file, script, or doc none of those steps named stays unread; curiosity
  reads work against the read-only design as surely as an edit would.
- Read-only. context-prime orients; it does not start editing. The user confirms
  the focus first.
- Degrade gracefully: missing `CLIO-CODER.md` → next constitution file; missing Clio
  tools → plain `git`; no handoff → say so and orient from git + constitution.
