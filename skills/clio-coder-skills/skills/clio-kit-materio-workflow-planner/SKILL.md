---
name: clio-kit-materio-workflow-planner
description: 'Use when this workflow is requested: Build a resource-aware dependency
  plan with researcher assumptions and tagged defaults. Triggers on "clio kit materio
  workflow planner".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/plugins/materio/skills/materio-workflow-planner
  upstream-name: materio-workflow-planner
  upstream-eval-status: unspecified
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

Choose and customize one of the six research workflows with the researcher. Interview assumptions for each task type. Read .research/VIRTUAL-LAB.md and mark feasibility unchecked if absent. Write .research/WORKFLOW.md with stable task IDs, dependency ordering, researcher-origin assumptions and unconfirmed defaults, and all decision/human-action/human-verify checkpoints. Confirm defaults and alternatives before execution.

Read the [shared research policy](assets/references/research-policy.md) for
state, provenance, execution boundaries, and advisory validation.

Read the linked references when needed. Resolve links from this skill directory.
If acting as a worker, return questions to the orchestrator; when acting as the
interactive assistant, collect the researcher answers directly using the host's
conversation facility.

- [traditional-workflows.md](assets/references/traditional-workflows.md)
- [WORKFLOW.md](assets/templates/WORKFLOW.md)
- [VIRTUAL-LAB.md](assets/templates/VIRTUAL-LAB.md)
- [RESEARCH.md](assets/templates/RESEARCH.md)

## Plans Are Prompts

WORKFLOW.md is not a project management document. It IS the executable specification. Each task entry must contain everything the task-executor needs to run it without interpretation; specific inputs, expected outputs, and documented assumptions.

## Assumptions Are Research Decisions

Every assumption in a research workflow is a scientific decision. "We assume linear elastic behavior" is not just a simplification; it's a testable claim that could be wrong and should be documented. Good assumptions are:
- **Explicit**: Written down, not implied
- **Justified**: Why this assumption is reasonable
- **Testable**: How you'd know if the assumption breaks

Assumptions the researcher stated in the interview are theirs; record them verbatim with `[researcher]`. Assumptions you supplied are defaults; mark them `[planner default, confirmation pending]`.

## Task Atomicity

Each task should be completable in one focused work session (hours to days, not weeks). If a task is too large, split it. Signs of an oversized task:
- Multiple distinct outputs
- Multiple methods applied
- Results from one part needed before another part can proceed

## Traditional Workflows as Starting Points

Traditional workflows encode decades of research practice. They are starting points, not constraints. Honor the researcher's customizations and document why standard steps were modified or skipped.



If VIRTUAL-LAB.md is provided, check every task against its Resource-to-Task Mapping table:
- Required resource available → feasible
- Required resource has a gap → flag with ⚠ and the alternative from VIRTUAL-LAB.md
- Required resource completely unavailable with no alternative → mark `BLOCKED` and propose removing or replacing the task

Example flags:
- "⚠ Task 05 (TEM characterization): No in-house TEM. External facility available (2–3 week lead time); add booking step."
- "⚠ Task 07 (DFT with VASP): No VASP license. Alternative: Quantum ESPRESSO (free, installed on HPC)."

Every flag goes into the return block for the researcher's decision. Apply the alternative provisionally so the workflow is complete either way.

If VIRTUAL-LAB.md is missing, note in Workflow Decisions that feasibility was not checked.

## Step 5: Attach Assumptions

For each task, write the assumptions from `<assumptions>` verbatim, tagged `[researcher]`. Cover, per type, at least:

- **experimental**: sample preparation route and contamination risks; characterization tools and resolution; test conditions and standards (ASTM/ISO); sample size and replicates; success criteria
- **computational**: method and why; software; functional or force field; system size and timescale vs available resources; validation benchmark
- **data-analysis**: data source; statistical approach; outlier handling; visualizations needed; software and libraries
- **literature**: sub-topic scope; inclusion criteria; priority authors; output format
- **analytical**: governing model; boundary conditions; validation limit cases

Where the interview left a slot empty, supply a defensible default tagged `[planner default, confirmation pending]` and add it to the return block.



## Complete action guides

Read only the guide for the requested operation; it contains its interview and
state transitions. Resolve these links from this skill directory.

- [define-research-tasks](assets/actions/define-research-tasks.md)
