---
name: prototype
description: 'Use when this workflow is requested: Answers a design question with
  throwaway code: a clearly marked, runnable, no-persistence prototype whose verdict
  is kept and whose code is discarded. Not for pre-registered performance experiments;
  use experiment-protocol. Not for production implementation. Triggers on "throwaway
  prototype", "mock up this UI".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/coding/prototype
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Prototype

A prototype is throwaway code that answers a question. Name the question
first; the question decides the shape.

## Arguments

```text
/skill prototype [--branch logic|ui] [--subject <path>] <question>
```

- `--branch`: force the branch from Step 1. Omit to infer it.
- `--subject`: the file or module the prototype is about. Omit to find it
  from the question text.
- Everything else is the question the prototype must answer. If no
  question is stated, write the one you infer as the first line of your
  reply and proceed; do not stop to ask when running headlessly.

Examples:

- `/skill prototype sanity-check whether the retry state machine in retry.js feels right`
- `/skill prototype --branch ui what should the dashboard header look like`

The three steps below are the plan; do not open a task list for them.

## Step 1 — Pick the branch

From the question, the surrounding code, or the `--branch` flag:

- **"Does this logic / state model feel right?"** → read
  `references/LOGIC.md`. Build a single self-contained HTML file: free-play
  controls plus guided walkthroughs that push the state model through the
  cases that are hard to reason about on paper, drivable by a
  non-developer.
- **"What should this look like?"** → read `references/UI.md`. Generate
  several radically different UI variations on one route, switchable via a
  URL parameter.

The branches produce very different artifacts; getting this wrong wastes
the prototype. Ambiguous → default by neighborhood (backend module →
logic; page or component → UI) and state the assumption at the top of the
prototype and in your reply.

Read the subject code once, then write down in your reply, before any
code: the question, the branch, and the three to five cases the prototype
must exercise. That list is the acceptance bar for Step 2.

## Step 2 — Build under the prototype rules

1. **Throwaway from day one, marked as such.** Place it next to the code
   it prototypes for, named so a casual reader sees it is not production
   (`<subject>-prototype.html`, `prototype-<slug>/`). Follow the project's
   existing routing/layout conventions; invent no new top-level structure.
2. **Trivial to run.** One command in the project's own task runner, or
   one double-clickable HTML file. No setup thinking required.
3. **No persistence.** State lives in memory. If the question is itself
   about a database, use a scratch DB or file named "PROTOTYPE — wipe me".
4. **Skip the polish.** No tests, no error handling beyond runnable, no
   abstractions. Speed of learning is the only quality bar.
5. **Surface the state.** After every action (logic) or variant switch
   (UI), print or render the full relevant state so the change is visible.
6. **Write once, then edit.** Write the file once. Subsequent changes go
   through `edit`; never rewrite the whole file to change a few lines.
7. **Exercise it headlessly.** For a logic prototype, drive the real
   module through the Step 1 cases with one `node -e` (or the project's
   runtime) call and read the output. That transcript is the evidence
   for the verdict; a verdict from reading code alone is a guess.

Shell rules: run one command per `bash` call, plain and direct. Never use
`$(...)` or backticks; they trigger an approval gate that ends a headless
run.

## Step 3 — Capture and discard

Do these in order. Do not stop after building; a prototype without a
recorded verdict answered nothing.

1. **Decide.** Write the verdict in one sentence, then the evidence: which
   Step 1 cases behaved as expected, which did not, and what the model is
   missing.
2. **Park the code on a throwaway branch.** Run these as separate `bash`
   calls, substituting a short slug:

   ```bash
   git checkout -b prototype/<slug>
   ```
   ```bash
   git add <prototype files>
   ```
   ```bash
   git commit -m "prototype: <question> (throwaway, verdict in message)"
   ```
   ```bash
   git checkout -
   ```

   Returning to the original branch removes the committed prototype from
   the working tree, which is the point: the main line keeps only the
   validated decision, never the prototype. If the directory is not a git
   repository, leave the file in place and say so.
3. **Report.** Your final reply is the record. It names, in this order:
   the question, the verdict, the evidence, the recommended change to the
   real code (or "none"), and the branch pointer `prototype/<slug>`. When
   the work is tracked elsewhere (issue, plan, handoff), the user copies
   this block there; you do not need a separate report file, and you must
   not end the run with the `artifact` tool.

Done when the verdict is in the reply, the pointer exists, and
`git status` on the working branch shows no prototype files.

## Red flags

- A prototype quietly growing tests, error handling, or abstractions: it
  is becoming production without a decision.
- Persistence added "just to make it work".
- The prototype merged to the main line, or left untracked on it.
- Code built before the question was stated.
- A verdict written without running the cases.
- Ending the run by writing a report artifact instead of finishing Step 3.
