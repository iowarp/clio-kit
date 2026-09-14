---
name: herdr
description: 'Use when this workflow is requested: Launches, drives, or inspects another
  agent or command in a Herdr pane, tab, or workspace, including a second Clio Coder
  instance; requires HERDR_ENV=1. Not for background work a plain shell or dispatch
  already covers. Triggers on "use Herdr", "launch an agent in a Herdr pane".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/meta/herdr
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# Herdr

Herdr organizes terminals into workspaces, tabs, and panes, recognizes coding
agents running inside panes, and exposes the session through the `herdr` CLI.
Everything below runs through `bash`.

## Step 1 — Gate

Run `test "${HERDR_ENV:-}" = 1`. If it fails, say you are not running inside
Herdr and stop. Never inspect or control a Herdr session from outside one.

## Step 2 — Learn the installed CLI

The installed binary is the authority on syntax. Run `herdr --help`, then
print a command group by running it without a subcommand: `herdr agent`,
`herdr pane`, `herdr workspace`, `herdr tab`, `herdr worktree`,
`herdr terminal`, `herdr notification`, `herdr session`.

- Never run bare `herdr` for discovery; it launches or attaches the TUI.
- Never probe a mutating subcommand by omitting arguments; commands like
  `herdr workspace create` are valid with defaults and will execute.
- Control commands return JSON. Read every identifier from the response;
  never predict or reuse one from an example.

## Step 3 — Know the model

- Workspaces contain tabs contain panes. IDs are opaque and stable:
  `w1`, `w1:t1`, `w1:p1`. Closed IDs are not reused; a moved pane gets a new
  ID (`.result.move_result.pane.pane_id`), and the old one is only valid to
  the moved process itself.
- Pane commands drive raw terminals. Agent commands drive a recognized coding
  agent occupying a pane, and only they interpret the lifecycle states
  `idle`, `working`, `blocked`, `done`, `unknown`. `agent start` needs an
  existing available shell pane (interactive prompt, nothing in the
  foreground); it never creates or splits layout.
- Agent targets are a unique live agent name (`[a-z][a-z0-9_-]{0,31}`) or the
  hosting pane ID; never a terminal ID or a bare kind label.
- State meanings: `blocked` = an approval or question UI is showing. `done` =
  idle after unseen background work. `unknown` = present but unclassified; it
  does not prove completion.
- Herdr injects caller context: `$HERDR_WORKSPACE_ID`, `$HERDR_TAB_ID`,
  `$HERDR_PANE_ID`. Prefer `--current` to target the calling pane; omitting a
  target may hit the user's focused pane.
- Discover live state with `herdr workspace list`, `herdr tab list
  --workspace "$HERDR_WORKSPACE_ID"`, `herdr pane list --workspace
  "$HERDR_WORKSPACE_ID"`, `herdr agent list`.

## Step 4 — Launch a second Clio Coder instance

1. Default to a sibling pane in the current tab and the same cwd. Do not
   create a workspace, tab, worktree, or different cwd unless the user asked
   for that topology.
2. Check geometry with `herdr pane layout --pane "$HERDR_PANE_ID"`: split a
   wide pane `right`, a narrow or tall pane `down`. Avoid repeated
   same-direction splits.
3. `herdr pane split --current --direction right --cwd "$PWD" --no-focus`,
   then read the new pane ID from `.result.pane.pane_id`.
4. Run `herdr agent` and read the installed kind list. If a Clio kind is
   listed, start it with a unique, purposeful name:
   `herdr agent start <name> --kind <clio-kind> --pane <new-pane-id>`.
   Native agent arguments go only after `--`. `agent start` returns once the
   agent is detected and input-ready (30 s default timeout).
5. If no Clio kind is installed, fall back to the pane surface:
   `herdr pane run <new-pane-id> "clio-coder"` and drive it with pane commands;
   lifecycle states are then unavailable, so say so.

## Step 5 — Drive it

- Submit work: `herdr agent prompt <name> "<task>" --wait --timeout 120000`.
  `--wait` returns at the first settled `idle`, `done`, or `blocked`; do not
  restate those defaults with `--until`. A prompt from a non-working state
  must change lifecycle within 5 s or Herdr returns `agent_prompt_stalled`.
- Wait for a specific state only when the workflow needs it:
  `herdr agent wait <name> --until blocked --timeout 120000`.
- Interactive UI: `herdr agent send-keys <name> esc` / `ctrl+c` (keys are
  validated before any bytes are written).
- Read results: `herdr agent get <name>` and
  `herdr agent read <name> --source recent-unwrapped --lines 120`.
- On a failed wait or `blocked`, run `agent get` and `agent read` before
  sending any input. Done when the delegated task's output has been read back
  and reported; a `blocked` state is reported to the user, not guessed past.

## Step 6 — Ordinary commands in panes

Split as in Step 4, then: `herdr pane run <id> "<command>"`,
`herdr pane wait-output <id> --match "<literal>" --timeout 120000` (or
`--regex`, Rust syntax; the search also matches output that already exists),
`herdr pane read <id> --source recent-unwrapped --lines 120`.

Read sources: `visible` (viewport), `recent` (with soft wraps),
`recent-unwrapped` (wraps joined; prefer for logs), `detection` (agent
detection snapshot). Use `--format ansi` only when styling is evidence.

If more `--lines` reveals nothing further, the agent is on the terminal's
alternate screen and scrolled-off rows are unrecoverable. Fallback, only
after such a failed read: ask the agent to write its full response as
Markdown to a temp file and reply with the path, then `read` that file.

## Safety rules

- `--no-focus` for background work; keep the user's focus where it is.
- Target with `--current`, an explicit pane ID, or a unique agent name.
- Never close workspaces, tabs, panes, or sessions you did not create.
- Never run `herdr server stop` and never kill the main Herdr process; use
  named test sessions for isolated experiments.
- Server errors: JSON on stderr, exit 1. Syntax errors: exit 2.
