---
name: file-ticket
description: 'Use when this workflow is requested: Turns something noticed mid-session
  into a tracker issue: captures evidence, dedups against existing issues, composes
  a labeled issue with acceptance criteria, confirms, and creates it via gh. Not for
  batch ticket creation from a PRD; use backlog. Not for diagnosing an existing issue;
  use fix-issue. Triggers on "file a ticket", "open an issue for this".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/git/file-ticket
  upstream-eval-status: scenarios-recorded
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

# File Ticket

Turn one observed behavior into one well-formed tracker issue, then stop. The ticket is the deliverable; the fix belongs to a later skill in the pipeline (`fix-issue` picks it up from here).

See [issue discovery](references/issue-discovery.md) for template, label, and milestone discovery details, and [issue template](assets/issue-template.md) for the default body layout.

## Arguments

Arguments are passed in the user invocation message. Interpret them structurally from the prompt:

```text
/skill:file-ticket [--repo owner/repo] [--type type] [--milestone name] <observation>
```

### Examples
- `/skill:file-ticket settings overlay crashes when terminal is narrower than 40 columns`
- `/skill:file-ticket --type bug --milestone v1.2 "memory leak in worker pool"`
- `/skill:file-ticket --repo owner/custom-repo --type enhancement add dark mode support`

### Positional Arguments
- `<observation>`: Free-text description of the observed defect, symptom, or enhancement.
  - Required if not already established in recent session conversation.
  - If omitted, prompt the user for the observed behavior.

### Options
- `--repo <owner/repo>`: Target GitHub repository.
  - Default: detected canonical repository (`gh repo view --json nameWithOwner -q .nameWithOwner` or Git remote tracking).
  - Validation: Must match `^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`.
- `--type <type>`: Issue category.
  - Allowed: `bug`, `enhancement`, `documentation`, `question` (or repository-defined issue types).
  - Default: inferred from `<observation>` or confirmed with user via `ask_user`.
- `--milestone <name>`: Open milestone to attach to the ticket.
  - Default: none (or repository triage default).
  - Validation: Must match an open milestone returned by the GitHub API.

### Unknown Arguments and Safe Quoting
- Unknown options must be rejected with an error; do not pass unknown flags to `gh`.
- Never interpolate raw user strings into double-quoted shell command strings. Write the issue body to a temporary file and pass `--body-file "$tmpfile"` to prevent command substitution or backtick evaluation.

## Step 1 — Capture

From the session that surfaced the behavior, record before anything scrolls away:
- **Type**: bug, enhancement, documentation, or question. Uncertain -> ask via `ask_user`; never guess between bug and enhancement.
- **Observed vs expected**: what actually happened, in the reporter's words, and what should have happened.
- **Reproduction**: the exact commands, slash commands, or session steps that showed the behavior. Only steps actually performed; a repro you did not run is written as "unverified".
- **Evidence**: when the code location is already known or findable in one short grep pass, cite it as `file:line`. Stop after at most a few minutes of searching.

Complete when every bullet above is either filled or explicitly marked unknown.

## Step 2 — Resolve Repository and Dedup

1. Determine `--repo <owner/repo>` (provided flag or detected via `gh repo view`).
2. Search for existing issues:
   ```bash
   gh issue list --repo <owner/repo> --search "<key terms>" --state all --limit 20
   ```
   Requires an authenticated `gh` CLI; if the command fails, report the exact failure and stop rather than filing blind.
3. **Open duplicate found**:
   - Do NOT file a new issue.
   - Compose a comment with the new evidence and repro details.
   - **Mandatory confirmation**: Present the target issue URL and the exact comment text to the user via `ask_user`. Execute `gh issue comment <id> --repo <owner/repo> --body-file "$tmpfile"` ONLY after the user confirms.
   - Report that issue's URL as the outcome and end.
4. **Closed duplicate found**:
   - Continue to file a new issue and link the closed issue in the body's Links section.

## Step 3 — Discover Taxonomy and Compose

1. Discover repository issue templates, labels, and open milestones per [issue discovery](references/issue-discovery.md).
2. Format the ticket:
   - **Title**: conventional tag plus a specific imperative summary (`fix: ...`, `feat: ...`, `docs: ...`). Never a bare noun.
   - **Body**: Problem (observed vs expected), Reproduce, Evidence (`file:line` pointers and session excerpts), Acceptance criteria as a verifiable checklist, and Links. Use [the standard template](assets/issue-template.md) when no repository template matches.
   - **Labels**: one type label (`bug`, `enhancement`, etc.) plus applicable discovered area/component labels (`area:*`). Propose new labels to the user rather than creating them silently.
   - **Milestone**: apply `--milestone` if specified and valid; otherwise leave unassigned.

## Step 4 — Confirm Every Outward Action

Show the full title, repository, labels, milestone, and complete body.
Obtain explicit approval via `ask_user`. Issue creation is an outward-facing change; nothing is created before the user explicitly confirms.

## Step 5 — Create and Hand Off

Write the body to a file and execute:
```bash
gh issue create --repo <owner/repo> --title "..." --body-file "$tmpfile" --label "..." [--milestone "..."]
```
Run creation exactly once, verify with `gh issue view`, report the issue number and URL, and name the next pipeline step: `fix-issue`.

## Red Flags

- Filing without the Step 2 dedup search.
- Posting a comment on an existing duplicate without explicit user confirmation.
- Sliding into fixing the bug instead of filing it.
- Passing raw user text in `gh issue create --body "..."` instead of using `--body-file`.
- Creating repository labels or milestones nobody approved.
- Batch-filing multiple tickets; that is `backlog`'s job.
