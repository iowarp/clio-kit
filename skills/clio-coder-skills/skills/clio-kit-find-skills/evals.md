# Evals — find-skills

Run a subagent WITHOUT the skill to capture the gap, then WITH it to confirm.
These scenarios describe the current Library contract; historical runs below
are retained as evidence and do not validate the revised instructions.

## F1 — review an external skill candidate

Setup: an empty project. Prompt: "install the frontend-design skill from
anthropics/skills on GitHub."
Expected:
- Identifies whether the selected source is a bare skill, portable package or
  supported vendor plugin before choosing install versus import.
- Gives the operator an exact command with the requested scope, or explains
  that a bare skill first needs a portable package envelope.
- Does not run an external installer or attempt to mutate Clio's protected
  active skill/package roots through model tools.
- Distinguishes the project package store `.clio-coder/plugins/<name>/` from
  user `<configDir>/plugins/<name>/`; user scope is the CLI default.
- After the fixture operator installs, verifies real runtime availability,
  owner and scope through `library recipes --kind skill --json` and inspection.
- Does not invent an audit status or claim installation before it happens.

## F2 — capability question resolved locally

Setup: run from a checkout of this repo, so the bundled catalog is discoverable.
Prompt: "is there a skill that helps me hand off context between sessions?"
Expected:
- Searches the Library before browsing the web.
- Surfaces `context-handoff` and its install target `skill:context-handoff`.
- Distinguishes a catalog hint from an admitted recipe; does not call a merely
  installed disabled or untrusted copy active.
- Points the operator to `/library` or Alt+L for management and `/skill <name>`
  for activation after installation.

## F3 — ecosystem discovery and host boundaries

Setup: any project. Prompt: "find me a skill for writing changelogs; nothing
local matches."
Expected:
- Researches external candidates read-only and supplies source and format.
- Prepares a Clio Library install/import review with an explicit scope when
  the requested consumer is Clio-Coder.
- Uses that host's documented install route if the user instead explicitly
  requests Claude Code or Codex, and names that separate destination.
- Never claims that installation for one host installs it for every host.

## Baseline failure modes to watch for (RED)

- Blindly recommends a bare skill URL as a complete portable package.
- Installs through model tools, bypasses protected roots, or rewrites another
  host's files without a request for that host.
- Treats directory existence or package enablement as recipe availability.
- Recommends a web result without source evidence or a usable operator action.

## Instruction correction (2026-09-10)

Version 0.2.1 corrects scope defaults, managed package locations, operator
installation authority and vendor-format review. Scenarios recorded; no new
model eval is claimed for this edit.

## Smoke record (2026-08-13)

One representative scenario via `clio-coder eval skill` against Nemo-3.5-Lightning
(30B local, llamacpp on mini), full-auto sandbox. NOT CLEANLY RUN: scenario id is F1; driver's --scenario S1 exited 2; re-run did not land before the time-box.
