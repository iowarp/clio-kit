# Shared research policy

Research state lives under `.research/`. Keep supplied data and provenance; never
initialize a git repository. Commits require the explicit top-level JSON setting
`commit_research: true`. Defaults disable recording and network retrieval. The
[research state helper](../scripts/research_state.py) parses configuration and
provides initialization, task IDs, dependency validation, checkpoint archives,
data copies, and optional named-file recording. It never writes `.planning/`.
Use Python 3.10 or later. Report unavailable helper/checker execution as skipped;
never report a skipped check as passed. Pass arguments as separate argv values.

Task IDs are positive decimal integers formatted with at least two digits.
Keep existing IDs and output paths stable when reordering, archiving, or adding
tasks. Allocate a new ID above all active, archived, and on-disk IDs. Execution
order comes from dependencies, not numerical order. A task with an archived or
missing dependency requires a researcher decision and a revised dependency plan.

The interactive assistant owns researcher interviews and decisions. A worker
returns `needs_input` with checkpoint kind `decision` (choose between research
paths), `human-action` (supply data or perform physical work), or `human-verify`
(expert sign-off). Preserve the exact question, prior context, and supplied answer
when dispatching again. A worker checkpoint cannot approve itself.

Record assumptions with their source: researcher statements, or planner defaults
pending confirmation. Never silently change an approved scientific assumption.
Use VIRTUAL-LAB.md to identify available methods, resource gaps, alternatives,
lead times, and blocked tasks. If it is missing, label feasibility unchecked and
request the missing inventory before executing a task dependent on lab resources.

Prepared protocols, simulation input files, code, and figure specifications are
artifacts. They are not performed experiments, successful runs, or numerical
results. Mark the prepared artifact complete only when that is the task's approved
contract; keep physical work and expert review as explicit pending checkpoints.
Read actual output files and the task summary before marking completion. Do not
invent measurements, uncertainty, plots, successful execution, or researcher consent.

An optional host fleet may require an existing Git worktree to enforce its write
boundary. Clio's v4 fleet does. If no checkout exists, keep that boundary intact
and use direct execution followed by a separate read-only review. Give the executor
only the selected task output grant; give the reviewer no write grant. Do not
initialize git to make a fleet runnable. Hosts without worker delegation perform
a labeled readback in the main conversation and disclose that no independent
worker review occurred.

Read/ls inspection establishes that artifacts were reviewed. It is not
command-backed scientific validation or proof of successful simulation execution.
Preserve the host's grounding diagnostics; if a validation receipt is rejected,
report validation incomplete rather than overriding it. Advisory commands and real
scientific executions have separate receipts and limitations.

Inspect supplied papers, data, and the data index first. Fetch a provided URL only
when the researcher authorizes that corpus and project `web_search` is true.
The historical option name controls network retrieval and Crossref identity
queries, not a promise that the host supplies web search. Record metadata,
abstract, full-text, or primary-data inspection depth honestly. A limited corpus
can justify a local research question; it cannot prove global absence or novelty.
Report zero supported gaps when appropriate. Do not meet arbitrary source/gap quotas.

The [physics checker](../scripts/check_physics.py),
[script checker](../scripts/check_scripts.py), and
[citation checker](../scripts/verify_citations.py) are advisory. Exit 0 means no
reported findings in the checked inputs, 1 means findings for researcher review,
and 2 means incomplete coverage or a command/input failure. Record the inventory,
coverage, warnings, and accepted exceptions. A bibliographic identity match is
not evidence that a paper supports a scientific claim. Crossref absence does not
prove a citation or DOI is fabricated. Never delete citations automatically.

Checkpoint archives exclude data and prior checkpoints, preserve data on restore,
and verify every archived file before replacing research state. Restore replaces
other state exactly, so collect a researcher confirmation and save current state
first. Use the helper; do not execute ad hoc tar extraction or git tags.

For paper writing, create an author-reviewed brief and source/evidence import
handoff under `.research/handoff/`. The receiving wtf-p workflow owns its gated
new-paper, map-project, and create-outline actions and all `.planning/` writes.
Existing paper state means inspect/reuse/repair first. Keep locked and deferred
choices, exclusions, provenance, and completed-versus-planned work distinct.
