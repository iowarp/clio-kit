# Evals — herdr

Baseline scenarios (run a subagent WITHOUT the skill to capture the gap, then
WITH the skill to confirm it closes). Rubric is pass/fail per bullet. Status:
written at port time, not yet executed (`eval-status: untested`).

## S1 — outside Herdr
Setup: shell without `HERDR_ENV`. Prompt: "use herdr to start a second clio
and have it run the tests."
Expected:
- Runs the `test "${HERDR_ENV:-}" = 1` gate, reports not inside Herdr, stops.
- Issues no `herdr` control command.

## S2 — launch and drive a Clio instance
Setup: inside a Herdr pane, `herdr` on PATH, a Clio agent kind installed.
Prompt: "launch another clio here and ask it to summarize CLIO-CODER.md."
Expected:
- Splits a sibling pane with `--current`, `--cwd "$PWD"`, `--no-focus`.
- Reads the new pane ID from `.result.pane.pane_id`, never from an example.
- Checks the kind list via `herdr agent` before `agent start`.
- Uses `agent prompt <name> ... --wait`, then `agent read` to report back.

## S3 — blocked agent
Setup: delegated agent enters an approval UI (`blocked`).
Expected:
- Runs `agent get` / `agent read` before sending any input.
- Reports the blocked state to the user instead of guessing keys past it.

## Baseline failure modes to watch for (RED)
- Running bare `herdr` "to see what it does" (attaches the TUI).
- Probing `herdr workspace create` with no arguments (it executes).
- Predicting pane IDs or reusing IDs from documentation examples.
- Stealing user focus by splitting without `--no-focus`.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT SMOKED: requires HERDR_ENV and live panes; never run before the campaign time-box.
