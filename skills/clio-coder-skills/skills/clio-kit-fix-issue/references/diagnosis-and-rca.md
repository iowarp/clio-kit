# Diagnosis, Why-Chains, and Root Cause Analysis (RCA)

This reference outlines diagnosis techniques, why-chain construction, and root-cause documentation for `fix-issue`.

## Diagnosing Unknown Causes

When an issue does not already pinpoint the root cause:
1. **Reproduce first**: Write a reproduction test script or minimal test case before altering any application logic. Run it and verify it fails with the expected symptom.
2. **Trace the fault**: Follow the call stack or data flow backwards from the failure site to the originating trigger.
3. **Construct a why-chain**:
   ```
   WHY <observed failure>?  -> because <immediate condition> (evidence: path/file.ts:123)
   WHY <immediate condition>? -> because <underlying cause>   (evidence: path/file.ts:89)
   ROOT CAUSE: <the exact defect to fix>                    (evidence: path/file.ts:45)
   ```
   Every link in the chain must cite a verified line of code or observed runtime state. A link without an observed citation remains an unverified hypothesis.

## Routine Fixes vs Standalone RCA

- **Routine fixes**: The why-chain is included directly in the final completion summary presented to the user. No separate RCA file is required.
- **Complex or high-impact defects**: A standalone RCA document or detailed why-chain is written when:
  - The defect caused data corruption, downtime, or security exposure.
  - The diagnosis uncovered multiple interacting failure modes.
  - The fix requires significant architectural changes or breaking behavioral adjustments.

## RCA Destination and Issue Tracking

- An RCA or diagnosis artifact stays in the local repository or task output during development.
- **Do not assume downstream tools post comments automatically**: `ship` handles atomic commits and pull requests; it does not automatically post closing comments or attach labels to issues.
- If the maintainer or repository workflow calls for recording the RCA on the tracker issue (for example, under an `rca` status label), format the comment clearly and request explicit user confirmation (`ask_user`) before running:
  ```bash
  gh issue comment <id> --repo <owner/repo> --body-file "$rca_file"
  ```
- If the repository uses an `rca` label, propose applying it only after the user approves:
  ```bash
  gh issue edit <id> --repo <owner/repo> --add-label "rca"
  ```

## Respecting Maintainer Constraints

If an issue contains comments from maintainers specifying:
- Non-negotiable interfaces or constraints
- Performance budgets or dependencies that must not be added
- Out-of-bounds files or packages

These constraints are binding. If technical reality makes compliance impossible, stop, document the contradiction with evidence, and ask for user guidance before proceeding.
