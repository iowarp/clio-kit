---
name: materio-task-verifier
description: 'Use when this workflow is requested: Independently inspect research
  task outputs against their approved assumptions, artifacts, and evidence. Triggers
  on "materio task verifier".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/plugins/materio/skills/materio-task-verifier
  upstream-eval-status: unspecified
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Independently read one approved task specification, its actual output inventory, and summary. Check output completeness, assumptions, provenance, units, uncertainty, and scientific claims. Return checks performed, findings, and complete/blocked/needs_input with checkpoint kind. Keep files unchanged; expert sign-off remains a human-verify checkpoint.

Read the [shared research policy](assets/references/research-policy.md) for
state, provenance, execution boundaries, and advisory validation.

Read the linked references when needed. Resolve links from this skill directory.
If acting as a worker, return questions to the orchestrator; when acting as the
interactive assistant, collect the researcher answers directly using the host's
conversation facility.

- [WORKFLOW.md](assets/templates/WORKFLOW.md)
- [RESEARCH.md](assets/templates/RESEARCH.md)

# Independent task verification

Treat the workflow task as an output contract. Check every expected file against
the actual task directory and read the summary; executor self-report is not
evidence of existence or correctness. Preserve researcher-approved assumptions,
resource constraints and dependency requirements. Distinguish generated protocols
and unrun code from experiments or computed results. Inspect units, uncertainty,
source/claim traceability and explicit limits of validity. A citation entry alone
cannot establish a source supports a claim.

Report missing files, unsupported results or contradictory state as failed
validations. Advisory physics/citation/script warnings are findings for the
researcher, not permission to remove or rewrite work. Record a researcher's
accepted exception faithfully; do not invent consent. A passing file inspection
is not proof of general physical correctness or successful execution.

Describe read/ls checks as artifact inspection. They do not establish a
command-backed scientific validation. Keep any host grounding limitation visible;
do not invent command receipts or claim an ungrounded check passed. A rejected
host validation remains incomplete and is returned to the caller.


## Complete action guides

Read only the guide for the requested operation; it contains its interview and
state transitions. Resolve these links from this skill directory.

- [execute-task](assets/actions/execute-task.md)
