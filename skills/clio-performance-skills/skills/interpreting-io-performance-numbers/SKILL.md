---
name: interpreting-io-performance-numbers
description: Use when interpreting bandwidth, IOPS, request sizes and MPI-IO counters already provided. Triggers on "is 40 MB/s bad", "small writes", "collective versus independent". Calls no tools. Not for loading a profile; use diagnosing-a-slow-job.
clio-kit:
  bundle: clio-performance
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Interpret I/O Performance Metrics

A profiler hands back numbers. Nothing in the output says whether they are good.
This is the missing half.

## Bandwidth is not the headline number

Bandwidth needs a denominator and scope: bytes per I/O-active second differs
from bytes per wall-clock second; per-rank and aggregate numbers also differ.
`total bytes / runtime` gives application-average throughput, not an I/O time
fraction. A low value can reflect compute time, waiting, small requests or low
volume. A high value can still mean the critical path is storage-limited.

Use measured I/O timing and an equivalent baseline on the same filesystem,
node/rank count, request size and load. Summed per-rank I/O times overlap, so do
not divide their sum by wall time and call it a utilization percentage.
There is no universal bandwidth threshold that proves a job healthy.

## Request size is evidence, not a diagnosis

Small requests can amplify per-operation latency. Compare request-size
histograms, access patterns and metadata counts before recommending buffering
or aggregation. The best request size depends on the filesystem and workload;
verify a proposed change with a representative measurement.

## IOPS and bandwidth trade against each other

High IOPS with low bandwidth means many small operations — the small-request
problem above. Low IOPS with high bandwidth means large sequential transfers,
which can indicate large transfers but does not prove optimal performance. Reading either alone gives the wrong answer.

## Sequential versus random

Sequential access lets the filesystem read ahead. Random access defeats it, and
each read pays full latency. On a parallel filesystem, "random" often means
strided — each rank stepping through a shared file at an offset — which looks
random to the storage even though every rank is orderly.

## Collective versus independent MPI-IO

This is the one that most often has a real fix behind it.

- **Independent**: every rank issues its own requests. With 1,000 ranks writing
  small pieces of one file, the storage sees 1,000 uncoordinated small writes.
- **Collective**: the MPI library aggregates across ranks, and may use collective buffering so fewer processes issue larger requests.
  The behavior depends on the MPI implementation and configuration.

A profile dominated by independent operations, with small request sizes and many
ranks, is the classic fixable case: switching to collective calls can change
throughput by an order of magnitude without touching the science.

Independent is not always wrong. Ranks writing to genuinely separate files have
nothing to aggregate.

## Metadata

Thousands of opens, stats and closes with little data moved is a metadata-bound
job. Bandwidth tuning does nothing for it — the fix is fewer files. A run
creating one file per rank per timestep is the usual culprit, and it gets worse
with scale, not better.

## What not to do

- Do not call a bandwidth number bad without knowing the volume and runtime.
- Do not tune I/O for a job that spends most of its time computing.
- Do not read IOPS or bandwidth in isolation — the pair is the signal.
- Do not treat strided access as sequential because each rank is orderly.
- Do not recommend collective I/O for ranks writing to separate files.

## Completion check

State units (MB versus MiB), timing denominator, scope (rank/node/job), workload and reference measurement. Explain what additional timing or baseline would distinguish latency, bandwidth, metadata and compute limits.
