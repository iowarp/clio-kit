---
name: clio-kit-fix-issue
description: 'Use when this workflow is requested: Resolves a tracker issue end to
  end: fetches it, diagnoses only when the issue does not name the code, fixes test-first,
  and self-reviews against the acceptance criteria, leaving a verified uncommitted
  change. Not for filing; use file-ticket. Not for committing or PRs; use ship. Triggers
  on "fix issue 123", "take this ticket".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/skills/git/fix-issue
  upstream-name: fix-issue
  upstream-eval-status: smoke-checked
license: Apache-2.0
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

# Fix Issue

Resolve one tracker issue: diagnose only as much as the issue leaves unknown, fix test-first, verify against the issue's own acceptance criteria. The skill ends with an uncommitted working tree and a completion report; committing, pushing, and PRs belong to `ship`.

See [diagnosis and RCA](references/diagnosis-and-rca.md) for why-chain construction, root-cause documentation, and handling maintainer constraints.

## Arguments

Arguments are passed in the user invocation message. Interpret them structurally from the prompt:

```text
/skill:clio-kit-fix-issue [--repo owner/repo] <issue-number|issue-url>
```

### Examples
- `/skill:clio-kit-fix-issue 42`
- `/skill:clio-kit-fix-issue #105`
- `/skill:clio-kit-fix-issue https://github.com/iowarp/clio-coder/issues/271`
- `/skill:clio-kit-fix-issue --repo acme/widgets 88`

### Positional Arguments
- `<issue-number|issue-url>`: The GitHub issue identifier to fix.
  - Allowed forms:
    - Pure integer: `42`
    - Hash-prefixed integer: `#42`
    - Full GitHub issue URL: `https://github.com/<owner>/<repo>/issues/<number>`
  - Required. If omitted, prompt the user for the issue number or URL.

### Options
- `--repo <owner/repo>`: Target GitHub repository.
  - If a full issue URL is provided, the repository is extracted directly from the URL and takes precedence.
  - If only an issue number is provided, `--repo` overrides the default.
  - Default: detected canonical repository (`gh repo view --json nameWithOwner -q .nameWithOwner` or Git remote tracking).
  - Validation: Must match `^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$`.

### Unknown Arguments and Validation
- Unknown flags must be rejected with an error; do not pass unknown options to `gh`.
- Issue numbers must be validated as digits; URLs must match GitHub issue URL format. Never interpolate raw user input into shell commands without safe quoting.

## Step 1 — Fetch and Gate

1. Deterministically resolve repository and issue ID.
2. Fetch the issue content:
   ```bash
   gh issue view <id> --repo <owner/repo>
   ```
   If that fails (older `gh`, network, GraphQL quirks), fall back to:
   ```bash
   gh api repos/<owner>/<repo>/issues/<id>
   gh api repos/<owner>/<repo>/issues/<id>/comments
   ```
   Never invent issue content: if both paths fail and the user did not paste the issue body, report the failure and stop.
3. Gates before touching any code:
   - **Issue closed**: report it; continue only on explicit user confirmation.
   - **Linked PR exists**: report it and confirm before duplicating effort.
   - **Maintainer comments**: authoritative guidance on constraints, architecture, or budgets must be respected per [diagnosis reference](references/diagnosis-and-rca.md).

## Step 2 — Diagnose Only the Unknown

Read the issue's evidence first:
- If the issue already cites the failing code (`file:line`, an exact stack trace, or commit SHA), verify the code directly by reading it and skip broad exploration.
- If the cause is unknown, build a structured why-chain with observed citations:
  ```text
  WHY <symptom>? -> because <cause> (evidence: file.ts:123)
  ROOT CAUSE: <exact code to change> (evidence: file.ts:789)
  ```
- Routine fixes keep their why-chain in the completion report. For complex bugs, format a standalone RCA. Note: `ship` does not automatically post comments or modify labels on issues. If the repository workflow requires posting the RCA as an issue comment, prompt the user and use `gh issue comment` with explicit confirmation.

## Step 3 — Fix Test-First

1. Write a focused reproducing test before editing implementation code.
2. Run the test and observe it fail for the stated defect reason.
3. Make the smallest necessary change to resolve the root cause.
4. Stay strictly within the issue scope. Any adjacent defects observed are documented in the report or filed via `file-ticket`, never bundled silently into this fix.

## Step 4 — Verify

1. Run the reproducing test to verify it now passes.
2. Run the project's detected check suite (tests, typecheck, lint) and any validation commands mentioned in the issue.
3. Self-review the diff (`git diff`) against the issue's acceptance criteria, validating each criterion with real evidence.

## Step 5 — Report and Stop

Leave the working tree uncommitted. Deliver a structured report:
- Root cause (why-chain or verified issue evidence)
- Files modified with net line diffs
- Acceptance criteria checklist with verification evidence
- Validation commands executed and their actual outputs
- Out-of-scope observations or follow-ups

Handoff: Inform the user that the change is verified and uncommitted, ready for `ship`.

## Red Flags

- Re-deriving a root cause the issue already accurately documents.
- Changing implementation code before a reproducing test fails.
- Bundling out-of-scope refactoring or fixes into the change.
- Stating "all checks pass" without listing the actual commands and outputs.
- Committing, pushing, or creating pull requests from this skill.
