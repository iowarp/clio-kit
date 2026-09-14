---
name: backlog
description: 'Use when this workflow is requested: Turns a finished PRD or architecture
  doc into a ticket backlog of small stories with verifiable acceptance criteria,
  confirmed, then created as GitHub issues or in another configured tracker. Not for
  local sprint slicing into a SPRINT.md; use cut-it. Triggers on "create the stories",
  "turn this PRD into issues".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/planning/backlog
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Backlog

Turn a finished planning doc into small, engineer-ready tickets on a real
tracker. Decomposition is tracker-agnostic; creation branches on the target.

## Arguments

```text
/skill backlog <path to the finished PRD or planning doc>
```

- Required: the doc path. Named inline, or clearly the file just discussed
  in context — never a doc invented from memory. If no path can be found or
  inferred, say so at Step 1 and stop; do not decompose without a doc.
- Optional, inferred rather than asked for by default: target platform
  (Step 1's own rule picks it — see below) and a milestone/epic to attach
  tickets to.

There is no operator in a headless run: `ask_user` is not registered, so
any call is refused as an unregistered tool rather than answered; that is
not a stall, and calling it again will not produce a different result. **This skill treats headless
degradation differently at each step below, because Step 3 gates an
outward, not-cleanly-reversible action — real tickets, GitHub issues or
persisted local tasks — not a document write:**

- **Step 1 (platform).** The default rule (`gh` when a remote and the `gh`
  binary are both present; the `tasks` fallback otherwise) is a fact to
  detect, not a judgment call, and runs the same whether or not anyone is
  watching — no confirmation needed either way. It is genuinely ambiguous
  only when the user names a platform with no available integration; if
  that `ask_user` call goes unanswered, do not guess a fallback and do not
  silently substitute `gh` or `tasks` for the platform actually requested —
  state plainly that the named platform is unavailable and confirmation of
  what to use instead was not obtainable, then stop exactly as Step 3 does.
- **Step 2 (decompose).** Always run to completion, headless or not.
  Decomposition itself creates nothing yet, and every source-doc gap still
  needs surfacing whether or not a human is present to read it.
- **Step 3 (confirm before creating).** An unanswered `ask_user` here is
  never a yes, and this gate does not get the assumed-confirm-and-proceed
  treatment the other planning skills use for their document gates. Finish
  Step 2, print the full proposed ticket list and target platform exactly
  as Step 3 already requires, then stop: say explicitly that confirmation
  is required before any ticket is created, that it was not obtainable in
  this run, and that Step 4 did not execute. This is the one gate in the
  planning category's headless story that ends in a stop rather than an
  assumed yes — filing unrequested public GitHub issues, or persisting
  local tickets nobody confirmed, on a guess is worse than an incomplete
  run; the full decomposed list is still delivered as the answer.

The steps below are the plan; hold it in your head, not in a tool — do not
call `tasks` with `plan`/`add`/`done`/`block`/`drop` to track your own
progress through Steps 1–5 as if they were a workflow board. What actually
matters when Step 3 stops without confirmation: the board must end this
run holding zero net-open items. `tasks` calls here are Step 4's — one
entry per ticket *already confirmed at Step 3* — and Step 4 never runs
before that yes. If you do reach for `tasks` before confirmation (to
stage the proposal, or to check the board is clean), any item you opened
with `plan`/`add` must be `drop`ped again before you finish, in the same
run — an item left open on the board is exactly "created without
confirmation," whether or not you called it that. A read-only
`tasks(action="list")` changes nothing and is always fine. Do not re-invoke
`context(scope="skills")` once the skill is already loaded this turn — it
is refused as a redundant call and wastes a turn; if you need to recheck
what you already read, use `read`/`grep` on the files themselves.

Shell rules for every `bash` call: one command per call, plain and direct.
Never use `$(...)` or backticks; they trigger an approval gate a headless
run cannot answer, and the call is refused outright. Never redirect output
to `/tmp` (`> /tmp/...`) to stage or capture something for a later step —
that write needs an approval a headless run cannot give either, and the
call is refused the same way; just run the command and read its output
directly, nothing needs staging on disk first. The `git` tool only covers
`status`/`diff`/`log`; it has no `remote` op, so Step 1's remote and
`gh`-availability check still goes through `bash` (e.g. `git remote -v`,
`command -v gh`) — use `git` for a plain status check, `bash` for
everything else this skill needs from git or `gh`.

## Step 1 — Inputs

Required: the path to the PRD or planning doc (see Arguments). Target
platform: GitHub issues via `gh` when a git remote exists and the `gh`
binary is present — check both in one `bash` call (`git remote -v;
command -v gh`) and trust the result; empty output from `git remote -v` is
a definitive "no remote", not an inconclusive check that needs a second or
third method (`git config`, reading `.git/config` directly — the latter is
a hard-blocked path regardless of this skill). Fall back to local tracking
automatically (see Step 4) when there is no remote, `gh` is unavailable, or
the user asked for local tracking, and say which was detected and why. A
platform the user names with no integration available is genuinely
ambiguous → ask, never guess, and never silently substitute a different
platform than the one requested (see Arguments for the headless case).
Optional: a milestone or epic to attach tickets to.

## Step 2 — Decompose

Read the doc in full. Each implementation phase becomes a ticket group;
each user story becomes one or more tickets. Per ticket, draft:

- **Title**: imperative and specific ("Add token refresh endpoint", never
  "Auth").
- **Description**: what and why, linked back to the source doc's phase.
- **Acceptance criteria**: a checklist a reviewer can verify — observable
  behavior, not "works correctly".
- **Phase label**: `phase-N` from the source doc.

Sizing rules: a ticket is at most about a day of work; a ticket that needs
more than one screen to describe is two tickets. A phase too vague to
decompose is a gap in the source doc — stop and flag it rather than
inventing tickets. Flag it by name (which phase, what is missing) in the
Step 3 list and again in the Step 5 report; do not paper over it with a
plausible-sounding ticket the source doc never actually specified.

## Step 3 — Confirm before creating

Print the full proposed list, grouped by phase — title *and* acceptance
criteria per ticket, not titles alone, plus any flagged gaps from Step 2 —
and the target platform, and get explicit confirmation via `ask_user`.
Ticket creation is outward-facing and not one-click reversible; nothing is
created before the yes. When that confirmation cannot be obtained (see
Arguments), stop here — do not proceed to Step 4.

## Step 4 — Create

Runs only after Step 3's explicit yes.

GitHub:

```bash
gh label create "phase-N" 2>/dev/null || true
gh issue create --title "..." --body "..." --label "phase-N" [--milestone "..."]
```

Acceptance criteria go in the body as a markdown checklist. Capture every
created issue number and URL. A failed create is reported per ticket, not
papered over.

Local fallback: when the workspace has no GitHub remote, gh is unavailable,
or the user asks for local tracking, create each confirmed ticket with the
`tasks` tool instead (one task per ticket, acceptance criteria in the
description) and capture the task ids. Say which target was used and why.

Every report in this skill — confirmed-and-created or stopped-before-
confirmation — is delivered as a chat message, never a file. Do not call
`write` to save the proposal or the report to disk as a backlog/proposal
markdown file; `write` is not in this skill's tool surface and the call is
refused. If the user wants the backlog persisted as a document, that is a
different, file-producing skill (`prd`, `architecture`), not this one.

## Step 5 — Report

Confirmed-and-created run: a table of title → phase → issue number/URL,
plus the source doc path and any failures. Done when every proposed ticket
is either created with its URL captured or listed as failed with the
error.

Stopped-before-confirmation run (see Arguments): one self-contained final
message — never split across turns and never "see above" — that repeats
the full proposed ticket list with title *and* acceptance criteria per
ticket (a status column of "proposed" is not a substitute for the
criteria themselves), plus one explicit line stating that confirmation was
required and unavailable and no ticket was created. A reader who sees only
this last message must get the complete list and the complete status;
never let the table alone imply creation happened, and never make the
stop notice depend on an earlier turn still being visible.

## Red flags

- Tickets created before the Step 3 confirmation — including a headless
  run that treats an unanswered `ask_user` there as a yes and proceeds to
  Step 4 anyway; that gate stops, it does not assume.
- A final report whose ticket table doesn't say, in words, whether creation
  happened or confirmation was unavailable.
- Acceptance criteria that restate the title.
- A mega-ticket hiding a week of work.
- Tickets with no trace back to a phase or story in the source doc.
- A vague phase decomposed into invented tickets instead of flagged as a
  source-doc gap.
- Any item left net-open on the `tasks` board when a run stops without
  Step 3 confirmation; `tasks` here is Step 4's confirmed-ticket store, and
  anything opened before that as a scratchpad must be dropped again in the
  same run.
