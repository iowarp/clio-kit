---
name: diagnosing-a-slow-job
description: Use when investigating I/O bottlenecks using a genuine Darshan profile and application logs. Triggers on "Darshan log", "why was this job slow", "I/O bottleneck". Not for metric interpretation alone; use interpreting-io-performance-numbers.
clio-kit:
  bundle: clio-performance
  servers: clio-darshan, clio-parallel-sort
  provenance: designed
  eval-status: scenarios-recorded
---

# Diagnose Scientific I/O Performance

A Darshan log records what a finished job did to the filesystem. This skill uses
that evidence to develop and test a bottleneck hypothesis. It spans the profiler and the log tools, and the order is not
optional.

## Load before anything else

`clio-darshan:load_darshan_log` parses the log and holds the result. **Every other
darshan tool operates on what that call loaded** — none of them take a log path.
Calling them first is an error, not an empty answer.

One log is loaded at a time. To look at a second run, load it; the first is
replaced. `clio-darshan:compare_darshan_logs` is the tool that holds two at once.

## Steps

**1. Load the log.** `clio-darshan:load_darshan_log`.

**2. Get the shape of the run.** `clio-darshan:get_job_summary` — runtime, process
count, total I/O volume. Compute the crude number yourself: volume ÷ runtime. This is
average application throughput, not the fraction of time blocked on I/O.
Compare I/O time on the critical path and an equivalent filesystem baseline;
high throughput can coexist with an I/O-bound job.

**3. Get the real metrics.** `clio-darshan:get_io_performance_metrics` — bandwidth,
IOPS, request sizes. Request size is the field that usually explains everything;
see `interpreting-io-performance-numbers` for what counts as bad.

**4. Look at how the files were touched.**
`clio-darshan:analyze_file_access_patterns` — read/write mix, sequential vs random.

**5. Go down a layer, and pick the right layer.**
- `clio-darshan:analyze_posix_operations` for raw read/write syscall counts.
- `clio-darshan:analyze_mpiio_operations` for collective vs independent MPI-IO.

For an MPI code, inspect MPI-IO when that module is present. MPI programs can
also issue POSIX I/O directly; missing MPI-IO counters do not establish that no
I/O occurred. Avoid double-counting bytes observed at both layers.

**6. Ask for the summary.** `clio-darshan:identify_io_bottlenecks` produces the
ranked findings.

> There is a tool with this exact name on the HDF5 server too. They are not the
> same: darshan's reads a finished job's profile, HDF5's inspects a file's layout.
> Use the fully qualified name.

**7. Cross-check against the timeline.** `clio-darshan:get_timeline_analysis`. A
job whose I/O is one spike at the end is a different problem from one that is slow
throughout, and the metrics alone cannot tell them apart.

**8. Confirm against the job's own logs.** Darshan says what happened to the
filesystem; it does not say what the application thought it was doing. Use
`clio-parallel-sort:filter_by_time_range` around the spike from step 7 and
`clio-parallel-sort:filter_by_log_level` for errors in that window. See
`searching-large-log-files`.

**9. Report.** `clio-darshan:generate_io_summary_report` for the written
summary. Check its `detailed_analysis`: each section reports its own success
separately, and a report can be produced from a partial set. Empty
`key_findings` and `recommendations` mean the sections that feed them did not
run, not that the job was healthy.

## Comparing two runs

When the question is "it was fast last week", go straight to
`clio-darshan:compare_darshan_logs` with both logs rather than loading each and
remembering the numbers. It is the only tool that sees both at once.

## What not to do

- Do not call any darshan tool before `load_darshan_log`.
- Do not name a cause from bandwidth alone — check request size and the timeline.
- Do not confuse `clio-darshan:identify_io_bottlenecks` with the HDF5 tool of the
  same name.
- Do not report POSIX syscall counts as the problem for an MPI code without
  checking the MPI-IO layer first.
- Do not reach for `clio-chronolog` here. It records LLM interactions, not job
  execution; it has nothing to say about a Slurm job's runtime.

## Completion check

Separate observed counters from hypotheses. Report volume, timing denominator, access pattern, relevant log window and missing evidence. Propose one controlled comparison; do not present a profiler heuristic as an established cause.
