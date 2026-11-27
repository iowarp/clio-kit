---
name: clio-kit-materio-task-executor
description: 'Use when this workflow is requested: Produce type-specific artifacts
  for one approved materials-science task and verify the files. Triggers on "clio
  kit materio task executor".'
compatibility: Clio Coder procedures adapted for skill discovery. Host tools, external
  services and native Clio execution gates require separate configuration; see the
  host compatibility note.
metadata:
  bundle: clio-coder
  servers: none
  provenance: adapted
  eval-status: scenarios-recorded
  source: https://github.com/iowarp/clio-coder/tree/c841a46101d6d9df5fd3bcb1d337e59d92fb660d/library/plugins/materio/skills/materio-task-executor
  upstream-name: materio-task-executor
  upstream-eval-status: unspecified
---

## Host compatibility

This copy is adapted from Clio Coder. Apply the procedure using the current host's available tools and the user's authorized scope. The tool names, `/skill` invocations, `.clio-coder` paths, fleets, approval gates and completion gates below describe Clio Coder; they are not installed or enforced by this skill in another host. Use the host's actual skill invocation and equivalent tools. If no equivalent exists, report the missing capability. Do not assume a tool is unavailable merely because the original headless workflow says so. A referenced agent or skill must be installed before relying on it. Scientific MCPs must be configured separately.

Adapted skills use the `clio-kit-` prefix to distinguish them from upstream audited skills. For a companion skill named below, select its `clio-kit-` copy from this collection; native agents, fleets and external programs retain their original names.

Execute one researcher-approved task specification under its selected .research/tasks/task-NN/ directory. Read dependency outputs, lab constraints, and registered inputs. Write the type-specific artifacts and task-NN-SUMMARY.md; distinguish prepared artifacts from measured or computed results. Return complete, blocked, or needs_input with checkpoint kind decision/human-action/human-verify. The interactive caller reads files back, resolves findings, and updates workflow state.

Read the [shared research policy](assets/references/research-policy.md) for
state, provenance, execution boundaries, and advisory validation.

Read the linked references when needed. Resolve links from this skill directory.
If acting as a worker, return questions to the orchestrator; when acting as the
interactive assistant, collect the researcher answers directly using the host's
conversation facility.

- [research-domains.md](assets/references/research-domains.md)
- [traditional-workflows.md](assets/references/traditional-workflows.md)
- [WORKFLOW.md](assets/templates/WORKFLOW.md)
- [VIRTUAL-LAB.md](assets/templates/VIRTUAL-LAB.md)

# Canonical task output contracts

The five per-type file contracts and format blocks below are copied verbatim
from the restored canonical executor. The literature retrieval steps use available host capabilities; the scientific output contracts are unchanged.


<output_scope>
Apply the output contracts below within the caller's explicit write grant.
Update `.research/LITERATURE.md` only when that path is authorized; otherwise put
proposed updates in the task summary for the orchestrator. Record actual supplied
files and provided URLs inspected in Sources Reviewed. Never represent generated
protocols, scripts or figure descriptions as performed experiments or measured
results. The orchestrator handles writing tasks through the paper handoff workflow.
</output_scope>

When spawned, read the `<task>` block and identify its **type**. Execution behavior differs by type:

---

## Type: `literature`

**What you do:** Read supplied materials and researcher-provided URLs, then synthesize papers on the task topic.

**Execution:**
1. Read `<data_files>` first: every registered paper, BibTeX file, or note relevant to this task is the primary corpus. Read each one before searching anything.
2. If `<web_allowed>` is true, config permits fetching, and URL retrieval is available, fetch only the specific URLs the researcher provided in `<provided_urls>`. Use only retrieval capabilities actually provided by the host. If fetching is disabled/unavailable or no URLs were supplied, work from the supplied materials and record that coverage in Sources Reviewed. Request missing sources through a checkpoint rather than inventing papers or search results.
3. Use URL retrieval to inspect the provided abstract/full-text URL and record the actual inspection depth and provenance. Supplied files come first; do not claim a PDF was read when only its filename or metadata was accessible.
4. Synthesize findings into a structured summary
5. Identify which papers are most relevant to the research gap
6. Quote every paper title in the summary and put a DOI in the BibTeX entry when known. Never invent bibliographic details; mark uncertain entries `[unverified]`. The orchestrator runs a Crossref check and shows flags to the researcher.

**Output files:**
- `task-NN-literature-summary.md`; structured synthesis by sub-topic
- `task-NN-references.bib`; BibTeX entries for key papers (formatted correctly)
- Updates to `.research/LITERATURE.md` if new gaps found

**Output format in summary:**
```markdown
# Literature Summary: [task name]

## Key Findings by Sub-topic

### [Sub-topic]
[synthesis; what is known, what is debated]

## Most Relevant Papers
| Paper | Relevance | Key Finding |
|-------|-----------|-------------|
| [ref] | [HIGH/MED] | [1 line] |

## New Gaps Identified
[Any gaps found that weren't in the original LITERATURE.md]
```

---

## Type: `experimental`

**What you do:** Generate a detailed experimental protocol and data recording template.

**Execution:**
1. Read task description, assumptions, and expected outputs from WORKFLOW.md
2. Generate a step-by-step experimental protocol appropriate to the material and measurement
3. Include: safety considerations, equipment settings, sample preparation steps, measurement procedure, data recording format
4. Generate a data recording template (spreadsheet structure or table)
5. Flag any assumption that needs verification before lab work begins

**Output files:**
- `task-NN-protocol.md`; detailed experimental protocol
- `task-NN-data-template.md`; data recording structure with column headers and units
- `task-NN-checklist.md`; pre-experiment verification checklist

**Protocol format:**
```markdown
# Experimental Protocol: [task name]

## Safety Notes
[relevant hazards; chemicals, high voltage, high temperature, etc.]

## Materials and Equipment
- [list with specifications]

## Sample Preparation
Step 1: [action]; [expected result]; [how to verify]
Step 2: ...

## Measurement Procedure
Step 1: [action]; [parameters]; [what to record]

## Data to Record
[table with: measurement, units, conditions, notes column]

## Expected Results
[from task assumptions; what would confirm success]

## Troubleshooting
| Symptom | Likely Cause | Fix |
|---------|--------------|-----|
```

---

## Type: `computational`

**What you do:** Generate simulation/calculation input files and analysis scripts.

**Execution:**
1. Read task assumptions for: method, software, system, parameters
2. Generate appropriate input files or scripts for the specified software
3. Include convergence testing guidance
4. Generate post-processing analysis script

**Supported methods and outputs:**
- DFT (VASP): INCAR, POSCAR, KPOINTS, licensed POTCAR setup instructions, submission script
- DFT (Quantum ESPRESSO): pw.x namelists/cards, pseudopotential setup instructions, submission script
- MD (LAMMPS): input script, force field notes, analysis scripts (OVITO/MDAnalysis)
- Phase-field: parameter file + solver notes
- CALPHAD: Thermo-Calc/Pandat script outline
- High-throughput: screening workflow script (Python + Materials Project API)

**Output files:**
- `task-NN-input/`; input files or templates
- `task-NN-run.sh`; job submission script (SLURM or local)
- `task-NN-analysis.py`; post-processing script
- `task-NN-README.md`; how to run + expected outputs

---

## Type: `data-analysis`

**What you do:** Analyze uploaded data files and produce figures and results summary.

**Execution:**
1. Read DATA-INDEX.md to find relevant data files for this task
2. Inspect data structure (headers, units, data types)
3. Write analysis code appropriate to the data type:
   - Mechanical test data (stress-strain): Young's modulus, yield stress, UTS extraction
   - XRD data: peak identification, lattice parameter extraction, Rietveld outline
   - EIS data: equivalent circuit fitting outline
   - Microstructure images: grain size analysis, phase fraction outline
   - Computational output: energy, force, trajectory analysis
4. Generate figures with proper labels, units, and scientific formatting

**Output files:**
- `task-NN-analysis.py` (or .m for MATLAB); analysis code
- `task-NN-results-summary.md`; key numerical results with uncertainty estimates
- `task-NN-figures/`; placeholder descriptions for each figure (actual generation requires running the code)

**Results summary format:**
```markdown
# Results: [task name]

## Key Numerical Results
| Quantity | Value | Uncertainty | Units | Method |
|----------|-------|-------------|-------|--------|
| [e.g., Yield stress] | [X] | [±Y] | [MPa] | [0.2% offset] |

## Comparison to Expectations
[from task assumptions; did results match, exceed, fall short?]

## Anomalies or Concerns
[anything unexpected that needs investigation]
```

---

## Type: `analytical`

**What you do:** Develop mathematical models, derive equations, validate against theory or data.

**Execution:**
1. Identify the physical model appropriate to the task
2. State assumptions and governing equations
3. Derive/adapt the model for the specific material system
4. Validate against known limits or available data
5. Identify the parameter sensitivity (which inputs matter most)

**Output files:**
- `task-NN-model.md`; derivation, equations, assumptions, validity range
- `task-NN-validation.md`; comparison to limiting cases or data
- `task-NN-implementation.py`; numerical implementation if needed

Read supplied files and the data index first. Fetch only researcher-provided URLs
when project policy and the available host retrieval capability permit it. Record
actual inspection depth, failed/unreadable sources, and coverage. Metadata alone
is not full-text inspection, and a restricted corpus does not establish global
absence. Ask for usable text if an unreadable paper blocks the research.

## Complete action guides

Read only the guide for the requested operation; it contains its interview and
state transitions. Resolve these links from this skill directory.

- [execute-task](assets/actions/execute-task.md)
