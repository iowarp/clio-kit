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
        "Inspect the installed ParaView tools, open field.vti, and save a PNG figure of it in this project. Confirm the scalar range by measuring it and name the representation; do not claim a figure if ParaView is not found.",
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
        "storage-compose",
        "Write a CLIO Core compose section for one node with DRAM and an NVMe file tier where data fsynced by the application must survive a runtime crash. State what an acknowledged put means and what a kill -9 can lose. Do not start a runtime.",
        facts=(
            r"(?i)persistence_level",
            r"(?i)metadata_log_path",
            r"(?i)fsync|SyncTag",
        ),
    ),
    case(
        "clio-runtime-config",
        "I pip installed iowarp-core on two machines. The runtime runs on host A with the default config. On host B I set CLIO_X=TCP and my Python client hangs for 30 seconds, then cannot connect. What is wrong and how do I configure both sides? Do not start a runtime.",
        facts=(
            r"(?i)CLIO_IPC_MODE",
            r"(?i)CLIO_SERVER_ADDR",
            r"(?i)CLIO_BIND_ADDR|0\.0\.0\.0",
        ),
    ),
    case(
        "clio-bdev",
        "I have three NVMe drives on one node. Give me a CLIO Core compose config where my CTE data survives one drive dying, and explain how to test it. Do not start a runtime.",
        facts=(
            r"(?i)clio_safe_bdev|safe_bdev",
            r"(?i)max_failures",
            r"(?i)persistence_level",
        ),
    ),
    case(
        "clio-cte-core",
        "I pip-installed iowarp-core, gave the CTE core at 512.0 an NVMe file tier, and now every PutBlob from Python returns 11. What is wrong and how do I fix the config? Do not start a runtime.",
        facts=(
            r"(?i)persistence_level",
            r"(?i)target",
        ),
    ),
    case(
        "clio-cte-caches",
        "With the default CLIO Core config my CTE writes vanished after the node rebooted. What does an acknowledged put mean, and how do I make writes durable? Do not start a runtime.",
        facts=(
            r"(?i)write-back|write_back",
            r"(?i)fsync|SyncTag",
            r"(?i)flush_period_ms",
        ),
    ),
    case(
        "clio-replication",
        "We run CLIO Core with replication_factor 2 on 4 nodes. When one node dies, reads of its blobs hang 30 s and then fail. What is wrong? Do not start a runtime.",
        facts=(
            r"(?i)failover_to_successor",
            r"(?i)CLIO_NET_DEAD_FAIL_FAST|CLIO_PROBE_SILENCE_S",
        ),
    ),
    case(
        "clio-compressor",
        "I pip-installed iowarp-core. How do I get zstd compression on everything written through my CLIO filesystem mount? Do not build or start anything.",
        facts=(
            r"(?i)CLIO_CTE_ENABLE_COMPRESS|\+compress",
            r"(?i)default_compress",
        ),
    ),
    case(
        "clio-indexer",
        "SemanticSearch from Python with clio_cte_core_ext over TCP always returns an empty list on my pip install of iowarp-core. Why, and how do I search my blobs by keyword? Do not start a runtime.",
        facts=(
            r"(?i)564",
            r"(?i)CLIO_CTE_POOL",
        ),
    ),
    case(
        "clio-fs",
        "I pip-installed iowarp-core on a 4-node Linux cluster. How do I mount the CLIO filesystem on every node, and what does fsync guarantee if a node dies? Do not mount anything.",
        facts=(
            r"(?i)clio_cte_fuse",
            r"(?i)fsync",
            r"(?i)EIO",
        ),
    ),
    case(
        "clio-cae",
        "Write a CLIO OMNI file that imports /data/sim.h5, only the HDF5 datasets under /fields, and show the command that runs it. Do not run it.",
        facts=(
            r"(?i)include_patterns",
            r"(?i)clio-cae|ParseOmni",
        ),
    ),
    case(
        "chimod-dev",
        "Plan a new CLIO Core chimod named counter with one Increment method returning the new value. List the files to write, the clio_mod.yaml contents, the codegen command and how the test initializes the runtime. Do not build anything.",
        facts=(r"(?i)clio[-_]run refresh", r"(?i)kIncrement:\s*1\d", r"(?i)CLIO_INIT"),
    ),
    case(
        "gpu-vector",
        "A gpu_vector kernel matches its baseline with the page cache larger than the data, and its log shows evictions=0. Is it correct out of core? Say what run would show that and which verbs must follow each write.",
        facts=(r"(?i)evict", r"(?i)Flush", r"(?i)Unpin"),
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
        "slicing",
        "Slice part.stl here for a Prusa MK4 in PLA at 0.2 mm and tell me how long it takes to print and how much filament it uses.",
        servers=("slicer",),
        facts=(r"(?i)prusaslicer|orcaslicer|SLICER_BIN",),
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
