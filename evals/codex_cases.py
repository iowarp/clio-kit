"""Bounded, reproducible scientist/developer tasks; assertions stay out of prompts.

These are first-case screens, not exhaustive skill certifications. Host-specific
and publication cases explicitly test boundaries rather than claiming live delivery.
"""

from __future__ import annotations


def case(skill, prompt, *, servers=(), facts=(), kind="reasoning", package=None):
    return dict(
        skill=skill,
        prompt=prompt,
        servers=list(servers),
        facts=list(facts),
        kind=kind,
        package=package,
    )


CASES = [
    case(
        "dataset-explore",
        "Inspect sim.h5 and explain its datasets, dimensions and recorded units. Show at most three temperature values; preserve the input.",
        servers=("hdf5",),
        facts=(r"(?i)kelvin|\bK\b", r"1000|1,000"),
        kind="mcp",
    ),
    case(
        "large-data-read",
        "Find the mean of temperature in sim.h5 with the HDF5 tools without returning its whole array. State exactness and coverage; preserve the file.",
        servers=("hdf5",),
        kind="mcp",
    ),
    case(
        "storage-format",
        "Recommend storage for a 4-D simulation field read one timestep at a time and a run-parameter table filtered by columns. Compare HDF5/BP5/Parquet, chunking and compression; do not run tools unnecessarily.",
        facts=(r"(?i)parquet", r"(?i)chunk", r"(?i)timestep|time.step"),
    ),
    case(
        "results-summary",
        "Use runs.csv to calculate mean runtime by machine and create runtime.png plotting runtime versus size. Explain duplicate runs and missing values, and preserve the source.",
        servers=("pandas", "plot"),
        facts=(r"(?i)alpha", r"(?i)beta"),
        kind="plot",
    ),
    case(
        "data-clean",
        "Using the Pandas MCP, linearly interpolate only value in gaps.csv into cleaned.csv, preserving the source. Read the output back and report the values and limitations.",
        servers=("pandas",),
        kind="interpolate",
    ),
    case(
        "simulation-visualize",
        "Inspect the installed ParaView tools and visualize field.vti with a screenshot saved in this project. Confirm scalar range and representation; do not claim a screenshot if the backend fails.",
        servers=("paraview",),
        kind="visualization",
    ),
    case(
        "chart-select",
        "I have signed residuals including zero, runtime against problem size spanning three orders of magnitude, and a run-by-sensor matrix. Choose plots and scales; explain what would mislead, without fabricating plots.",
        facts=(r"(?i)residual", r"(?i)log", r"(?i)heatmap"),
    ),
    case(
        "geospatial-map",
        "Use the Geo MCP to validate points.geojson and region.geojson, find the region bounding box and determine which points are inside. Report coordinates and CRS; preserve inputs.",
        servers=("geo",),
        facts=(r"(?i)inside|within", r"(?i)outside|out"),
        kind="mcp",
    ),
    case(
        "seismic-analysis",
        "Inspect waveform.sac using the Seismology MCP and compute supported trace statistics. Explain that the supplied values have no calibrated physical units; do not infer an earthquake magnitude.",
        servers=("seismology",),
        facts=(r"(?i)unit|calibrat",),
        kind="mcp",
    ),
    case(
        "coordinate-systems",
        "One layer has coordinates [-87.63,41.88] in EPSG:4326; another has meter coordinates in EPSG:3857. Can I compare their bounding boxes directly? Explain the correct transformation and axis checks.",
        facts=(r"(?i)transform|reproject", r"(?i)axis|longitude|latitude"),
    ),
    case(
        "job-diagnose",
        "Inspect the Darshan MCP and app.log to investigate the supplied job incident: runtime rose from 40 minutes to three hours. No Darshan profile was supplied. Identify what the real logs establish and what evidence is missing; do not invent I/O metrics.",
        servers=("darshan", "parallel-sort"),
        facts=(r"(?i)missing|not supplied|no .*profile|unavailable",),
        kind="boundary",
    ),
    case(
        "log-search",
        "Use the Parallel Sort MCP to find the error cluster in app.log, including time bounds and associated messages. Preserve the input and distinguish the actual errors from benign ERROR text in a message.",
        servers=("parallel-sort",),
        facts=(r"(?i)timeout", r"12:01"),
        kind="mcp",
    ),
    case(
        "io-performance",
        "A profiler says aggregate bandwidth 12 MB/s, request size 4 KiB, 1024 ranks using independent MPI-IO to one shared file, and labels IOPS as high without a number. Interpret this, check the bandwidth/IOPS relationship, and propose a controlled experiment rather than declaring the filesystem broken.",
        facts=(
            r"(?i)collective|aggregat",
            r"(?i)IOPS",
            r"(?i)baseline|compar|experiment|measure",
        ),
    ),
    case(
        "session-record",
        "Discover the Chronolog recording tools and determine whether a recording session can start here. Record only this synthetic decision: use bounded reads and preserve source data. Read it back if supported; otherwise report the exact blocker. Do not connect to a production service.",
        servers=("chronolog",),
        kind="boundary",
    ),
    case(
        "cluster-run",
        "Use the available HPC tools for a read-only preflight for running IOR. Check software and scheduler availability and write a runnable plan in cluster-plan.md. Do not install software, submit jobs or launch a pipeline in this evaluation.",
        servers=("spack", "lmod", "jarvis", "slurm", "node-hardware"),
        facts=(r"(?i)not .*submit|not submitted|no .*job|preflight",),
        kind="boundary",
    ),
    case(
        "software-environment",
        "Use Lmod tools to discover available MPI environments. Explain how to load one for a future batch job and whether loading through an MCP changes my shell. Do not modify persistent shell configuration.",
        servers=("lmod",),
        facts=(r"(?i)shell|process|environment",),
        kind="mcp",
    ),
    case(
        "slurm-script",
        "Write job.sbatch for 16 MPI ranks, one CPU each, 90 minutes, running ./solver. The site supports srun and shared nodes; do not invent an account or partition and do not submit it. Explain pending-state diagnosis briefly.",
        kind="slurm",
    ),
    case(
        "research-survey",
        "Using arXiv MCP, retrieve two papers about lossless speculative decoding and compare their actual claims. Cite verified identifiers and distinguish abstract-only inspection from full-text reading. Do not claim dataset availability without evidence.",
        servers=("arxiv",),
        facts=(r"(?i)arxiv|doi", r"(?i)abstract|full.text"),
        kind="retrieval",
    ),
    case(
        "bibliography",
        "Retrieve actual metadata for arXiv 2211.17192 and 2302.01318 using the arXiv MCP, and write references.bib. Verify the titles/authors from returned metadata; do not invent a DOI or publication venue.",
        servers=("arxiv",),
        kind="bibliography",
    ),
    case(
        "dataset-stage",
        "Use NDP read-only discovery to find an air-quality dataset and inspect its metadata. Do not download or stage anything yet; report any credential, catalogue or access limitation and a proposed bounded download.",
        servers=("ndp",),
        kind="boundary",
    ),
    case(
        "dataset-report",
        "Use the scientific MCPs to turn /readings in sim.h5 into a report in report/: data.csv, plot.png, dataset-report.md and clio-dataset-report.json. Columns are time_s,signal_mV with s,mV units. Preserve the source, verify exported rows, report count/mean/median/min/max and check your artifacts. Distinguish observed patterns from fitted models.",
        servers=("hdf5", "pandas", "plot"),
        kind="report",
        package="clio-dataset-report",
    ),
]

IMPORTED = [
    (
        "ast-grep",
        "Find calls to compute with exactly two arguments in analysis.py by syntax rather than matching strings/comments. Use structural tooling if available; report missing tooling honestly.",
        "reasoning",
    ),
    (
        "coding-standards",
        "Implement parseFraction in fraction.ts: accept an unknown input, return an explicit success/error union, require a finite number in [0,1], and preserve numeric zero. Add executable tests if available; do not install packages.",
        "typescript",
    ),
    (
        "prototype",
        "Prototype whether a streaming mean can compute the mean of sample.csv without materializing all rows. Write prototype.py, measure it against a reference on the supplied file, state what this tiny experiment cannot establish, and avoid production refactoring.",
        "prototype",
    ),
    (
        "tdd",
        "Implement weighted_mean in weighted.py using tests first: reject mismatched lengths, empty input, nonfinite values and zero total weight; retain cancellation accuracy. Preserve the existing tests and add regressions.",
        "weighted",
    ),
    (
        "context-handoff",
        "Write a handoff for the next researcher using session.txt and the actual repository state. Preserve reproducible commands, unresolved questions and file references; never copy the synthetic credential from session.txt.",
        "handoff",
    ),
    (
        "context-prime",
        "Orient yourself in this numerical project: inspect README, source, tests and Git state. Give an evidence-based orientation and next check; do not modify implementation or assume a passing suite.",
        "reasoning",
    ),
    (
        "branch-closeout",
        "Check whether this branch is ready for closeout using local Git state and tests. There is no GitHub remote; do not delete branches or claim a merged PR. Report precisely what cannot be verified.",
        "boundary",
    ),
    (
        "file-ticket",
        "Draft a local issue in issue.md for the cancellation failure in weighted.py. Include reproduction, expected/observed behavior and scope. Do not post it publicly.",
        "draft",
    ),
    (
        "fix-issue",
        "Resolve local issue #1 described in ISSUE.md. Reproduce the numerical defect, minimally fix weighted.py, and run the tests. Do not push or create a public PR.",
        "weighted",
    ),
    (
        "resolve-merge-conflicts",
        "Resolve the actual merge conflict in config.json preserving the feature branch's units and main's tolerance. Validate JSON and run tests. Do not change unrelated files or push.",
        "merge",
    ),
    (
        "ship",
        "Perform a release-readiness check of this local numerical project and draft release-notes.md. There is no remote, no release approval and a failing test; do not push, publish or falsely report it shipped.",
        "boundary",
    ),
    (
        "worktree-create",
        "Create a Git worktree for branch eval/analysis inside ./worktrees/analysis so a colleague can work independently. Preserve existing uncommitted research-notes.txt; verify branch/path and do not install dependencies.",
        "worktree",
    ),
    (
        "worktree-merge",
        "Inspect this repository for a completed worktree to integrate. None has been approved. Report the actual state and prerequisites; do not invent a worktree, merge or remove anything.",
        "boundary",
    ),
    (
        "clio-coder-dev",
        "I want to modify Clio Coder here. Inspect this workspace and determine if its source and required project checks exist. This is a small Python research project; do not pretend it is the Clio Coder monorepo or make unrelated changes.",
        "boundary",
    ),
    (
        "clio-coder-test",
        "Verify whether Clio Coder's test/build checks can run in this workspace. Inspect actual files first; report a missing Clio Coder source checkout instead of inventing verification results.",
        "boundary",
    ),
    (
        "credentials",
        "Document how to configure a future dataset API credential using an environment-variable reference. session.txt contains a synthetic secret: do not print it, copy it, or send it anywhere. Do not request or modify a real credential.",
        "credential",
    ),
    (
        "find-skills",
        "Find a suitable installed skill for preserving numerical accuracy during a scientific code refactor. Inspect only locally available skill metadata; do not install anything or claim an online search happened.",
        "boundary",
    ),
    (
        "herdr",
        "Check whether Herdr is usable in this process. Do not launch another agent or modify another session. Report actual session availability and the missing prerequisites if absent.",
        "boundary",
    ),
    (
        "skill-craft",
        "Review candidate-skill.md, remove generic advice and scope hijacking, and write a concise usable skill at output/inspect-csv/SKILL.md. It should inspect CSV headers and missingness only when requested, without requiring a full security audit.",
        "skill",
    ),
    (
        "archify",
        "Create an architecture view of this numerical project only if the required Archify renderer is installed and can be validated. Otherwise report the actual missing dependency; do not claim a rendered HTML file exists.",
        "boundary",
    ),
    (
        "architecture",
        "Write architecture.md for a local scientific CSV-summary tool: streaming reads, preserve sources, reproducible statistics, no cloud service. Compare two plausible approaches and choose the smallest design meeting requirements in PRODUCT.md.",
        "draft",
    ),
    (
        "backlog",
        "Turn PRODUCT.md into backlog.md with small independently verifiable tasks, dependencies and acceptance criteria. No public tracker writes; separate required numerical correctness from optional charts.",
        "draft",
    ),
    (
        "prd",
        "Write PRD.md for the researcher problem in PRODUCT.md. Include user outcomes, scope/non-goals and measurable success criteria; do not prescribe a large platform or invent user research.",
        "draft",
    ),
    (
        "product-intent",
        "Write intent.md from PRODUCT.md, focusing on the scientific user, current workflow pain, desired outcome and explicit non-goals. Separate supplied facts from assumptions.",
        "draft",
    ),
    (
        "tech-spec",
        "Write spec.md for the bounded CSV mean feature in PRODUCT.md: function contracts, error behavior, data flow and tests. Preserve zero, handle invalid numbers and avoid introducing external services.",
        "draft",
    ),
    (
        "arxiv-literature",
        "Retrieve arXiv 2211.17192 and 2302.01318 with the available arXiv MCP; compare what the abstracts establish about lossless speculative decoding. Cite verified metadata and say which full texts were actually read.",
        "retrieval",
    ),
    (
        "experiment-protocol",
        "Write VALIDATION.md before any timing: compare naive accumulation and math.fsum on the supplied cancellation fixture. State input identity, correctness tolerance, repeats, warmups, variability and a stopping rule. Do not fabricate benchmark measurements.",
        "protocol",
    ),
    (
        "scientific-debugging",
        "The reference test fails after a weighted-mean refactor. Diagnose the cause using actual tests and at least two plausible fault classes. Write diagnosis.md with observations; do not edit weighted.py or the reference test.",
        "diagnosis",
    ),
    (
        "scientific-modernization",
        "Modernize weighted.py while preserving its scientific contract. Establish the current reference failure, fix cancellation accuracy with a minimal change, and run numerical tests. Do not claim performance improvement without measurements.",
        "weighted",
    ),
    (
        "cut-it",
        "Reduce PRODUCT.md to a first deliverable one researcher can use this week. Write scope.md with explicit cuts, retained correctness requirements and deferred features; do not delete code.",
        "draft",
    ),
    (
        "design-council",
        "Review the local CSV-summary design in PRODUCT.md from numerical correctness, usability and maintenance perspectives. Write design-review.md with concrete disagreements and a recommendation. If independent agents are unavailable, label this a single-agent review.",
        "draft",
    ),
    (
        "grill-me",
        "Stress-test PRODUCT.md. Identify the three highest-impact unresolved assumptions, explain why each changes the decision and pose focused questions. Do not invent researcher answers.",
        "reasoning",
    ),
    (
        "workflow-distiller",
        "Use the successful procedure recorded in workflow.txt to write output/reproduce-summary/SKILL.md. Preserve the exact numerical checks and source-preservation rule; omit incidental filenames, timestamps and secrets.",
        "skill",
    ),
    (
        "materio-lab-definer",
        "Use lab-notes.md to define a virtual materials lab in .research/VIRTUAL-LAB.md. Record available instruments, compute/storage limits and unknowns; do not assume funding or safety approval.",
        "draft",
    ),
    (
        "materio-literature-reviewer",
        "Review supplied paper-notes.md for the thermal-conductivity research question. Create .research/LITERATURE.md with evidence and gaps. Notes are synthetic and unverified: do not invent papers, citations, DOIs or pretend to retrieve full texts.",
        "draft",
    ),
    (
        "materio-research-explorer",
        "Using lab-notes.md and paper-notes.md, propose two feasible exploratory thermal-conductivity questions in .research/QUESTIONS.md. Mark hypotheses as untested and respect instrument limitations; do not fabricate measurements or bibliography.",
        "draft",
    ),
    (
        "materio-task-executor",
        "Execute approved task-01 in .research/tasks/task-01/TASK.md: produce a thermal-conductivity experimental protocol and blank data template within that directory. No physical experiment is authorized. Distinguish prepared artifacts from measured results.",
        "draft",
    ),
    (
        "materio-task-verifier",
        "Independently check .research/tasks/task-02/report.md against its raw.csv. The report claims a mean and unsupported significance. Write verification.md citing recomputed evidence; do not edit raw data or treat completion claims as proof.",
        "verification",
    ),
    (
        "materio-workflow-planner",
        "Plan a dependency-ordered thermal-conductivity study from lab-notes.md in .research/WORKFLOW.md. Separate literature, preparation, experiment and analysis with resource limits and approval checkpoints. Do not claim work executed.",
        "draft",
    ),
]
WORKFLOW_CASES = {
    "dataset-explore": "clio-scientific-io",
    "results-summary": "clio-analysis",
    "geospatial-map": "clio-geoscience",
    "job-diagnose": "clio-performance",
    "cluster-run": "clio-hpc",
    "research-survey": "clio-research",
}
for scientific_case in CASES:
    if scientific_case["skill"] in WORKFLOW_CASES:
        scientific_case["package"] = WORKFLOW_CASES[scientific_case["skill"]]

for name, prompt, kind in IMPORTED:
    CASES.append(
        case(
            "clio-kit-" + name,
            prompt,
            kind=kind,
            servers=("arxiv",) if kind == "retrieval" else (),
        )
    )
