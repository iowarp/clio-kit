# Issue Discovery and GitHub Integration

This reference covers repository discovery, template detection, taxonomy queries, and dedup searching for `file-ticket`.

## Target Repository Resolution

1. If the user passes `--repo owner/repo`, validate that it matches `^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$` and use it explicitly.
2. Otherwise, detect the target repository using `gh repo view`:
   ```bash
   gh repo view --json nameWithOwner -q .nameWithOwner
   ```
3. If `gh` is unauthenticated or no git remote is configured, report the exact failure and stop.

## Issue Template Discovery

Check for repository issue templates in priority order:
1. `.github/ISSUE_TEMPLATE/*.md` or `.github/ISSUE_TEMPLATE/*.yml`
2. `.github/issue_template.md`
3. `issue_template.md` (root or `docs/`)

If a matching template exists for the determined issue type (e.g. `bug_report.md` or `feature_request.md`), adapt the sections to match the template's required headings. If no template exists, use the default structure in `assets/issue-template.md`.

## Label and Milestone Discovery

Discover the repository's existing taxonomy before proposing labels:

```bash
# List all available labels in the target repository
gh label list --repo <owner/repo> --limit 100

# List open milestones in the target repository
gh api repos/<owner>/<repo>/milestones --jq '.[] | select(.state=="open") | {number: .number, title: .title, due_on: .due_on}'
```

Rules:
- Apply at most one type label (e.g., `bug`, `enhancement`, `documentation`, `question`) that exists in the repo.
- Apply area or component labels (e.g. `area:*`, `component:*`) only if they are already defined in the repo.
- If a relevant label or milestone is missing from the repository, propose creating or adding it during the confirmation step (`ask_user`); never create labels or milestones without explicit user confirmation.

## Dedup and Duplicate Handling

Search for existing open and closed issues matching key terms:

```bash
gh issue list --repo <owner/repo> --search "<key terms>" --state all --limit 20
```

- **Open duplicate found**: Do not file a duplicate. Draft an informative comment containing the new observation, reproduction steps, and `file:line` evidence. **Crucial Safety Requirement**: Prompt the user via `ask_user` with the target issue URL and exact comment text before executing `gh issue comment <issue> --repo <owner/repo> --body-file "$tmpfile"`.
- **Closed duplicate found**: A new issue may be filed if the defect regressed or the circumstance differs. Reference the closed issue in the Links section of the new ticket.
