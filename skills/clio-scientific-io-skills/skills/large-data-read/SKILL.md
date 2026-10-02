---
name: large-data-read
description: Use when choosing bounded reads and distinguishing sampled summaries from exact large-data statistics. Triggers on "file is huge", "compute this mean", "out of memory". Not for initial format discovery; use dataset-explore.
metadata:
  bundle: clio-scientific-io
  servers: clio-hdf5, clio-parquet, clio-adios
  provenance: designed
  eval-status: scenarios-recorded
---

# Dataset Reading

Avoid transferring a whole dataset into the agent context to compute one
number. Prefer a server-side reduction when it meets the requested coverage
and accuracy. An exact reduction may still need to read every value.

## Work out the size first

For a supplied HDF5 file, resolve its path and call `open_file` directly. Resource
discovery scans configured directories; an empty listing does not mean that the supplied file is missing.

From `get_shape` and `get_dtype`: multiply the dimensions, multiply by the item
size. A `(2000, 2000, 500)` array of float64 is 16 GB. Use it to choose a bounded read, and allow for temporary arrays and conversion
overhead. Shape and dtype require separate tool calls.

## If you want a statistic, do not read the data

| Question | Tool |
|---|---|
| Mean/min/max over all elements of each HDF5 dataset | `clio-hdf5:hdf5_aggregate_stats` |
| Aggregate over a Parquet column | `clio-parquet:aggregate_column_tool` |
| Discover structure across many HDF5 files | `clio-hdf5:hdf5_parallel_scan` |

`hdf5_aggregate_stats` reduces all elements of each selected dataset; it has no
column or axis selector. Do not use its scalar mean as one column's mean in a
multicolumn array, especially when columns have different units. Select a tool
that implements the requested column reduction, or report the missing capability.

Open the HDF5 file before requesting aggregate statistics. For datasets larger
than 500 MiB, `hdf5_aggregate_stats` may use a strided sample. Read the response's
`SAMPLED` or `FULL DATA` label, processed/total element counts and coverage.
Sample sum/count/min/max describe selected values only, not whole-dataset
totals or bounds. Cross-dataset aggregation is omitted when any result is sampled.
Striding can miss patterns; it is not a random representative sample. Older
server versions may omit these labels, so verify coverage against the full shape.
One- and multidimensional arrays use different stride logic. Use the actual
response's counts and coverage; before a call, leave those quantities unverified.
Multidimensional sampling also does not guarantee a small allocation.
For exact results, use a separately verified chunked calculation with complete
coverage or report that the available tool cannot establish the exact answer.
Do not request thousands of `hdf5_stream_data` summaries to reconstruct an exact
mean: they are rounded summaries, not lossless sums. Establish that a suitable
local reader is available before choosing a fallback. Inspect its actual callable
and selection/coverage behavior; an importable library or a source file containing
the existing aggregate does not establish that a chunked column reader exists.

## If you genuinely need values, take a bounded piece

- `clio-hdf5:read_partial_dataset` — reads the requested slice internally, but
  returns shape, dtype and only the first five flattened values. Keep the slice
  small. If the selected region contains more than five elements, this response
  cannot supply all of them for a CSV, a whole-region monotonicity check or an
  exact reduction. Verify an export tool's contract before using it for that handoff.
- `clio-parquet:read_slice_tool` — a row range with only the columns you need.
- `clio-parquet:get_column_preview_tool` — paginated values from one column.
- `clio-adios:read_variable_at_step` — one variable at one step, which is already
  potentially the entire spatial domain. This tool has no spatial slice
  argument: check the variable shape before calling it; one timestep can exceed
  memory or context limits.

## Reading a lot, deliberately

- `clio-hdf5:hdf5_batch_read` — several datasets in parallel, one call rather
  than a serial loop.

> `hdf5_batch_read` and `hdf5_aggregate_stats` both take `paths` as a
> **comma-separated string** (or a JSON array encoded as a string), not a JSON list, despite the plural name. Passing a
> JSON array fails with "Input should be a valid string".
- `clio-hdf5:hdf5_stream_data` — returns chunk summaries and stops at
  `max_chunks`; it is not an exact whole-dataset reduction. For multidimensional
  data, `chunk_size` slices the first axis, so account for all remaining
  dimensions when estimating memory. Check processed coverage explicitly.
- `clio-hdf5:read_full_dataset` — reads the full dataset internally and returns a
  description, not its values. Internal chunking does not establish bounded total
  memory. Do not use it as a value-export or exact-reduction fallback.

## Align reads to the chunks

`clio-hdf5:get_chunks` reports the on-disk chunk shape. Reading across the chunk
grain makes the library fetch and decompress whole chunks to hand back a sliver.
A slice along the chunked dimension can be many times faster than the same number
of elements taken across it.

`clio-hdf5:optimize_access_pattern` will suggest a better shape for a pattern you
describe, and `clio-hdf5:identify_io_bottlenecks` inspects the file's own layout.

> That last name also exists on the darshan server, where it means something
> different — a finished job's profile rather than a file's layout. Use the
> fully qualified name.

## What not to do

- Do not call `read_full_dataset` to compute a statistic.
- Do not read a dataset without checking its size first.
- Do not loop single reads where `hdf5_batch_read` takes them together.
- Do not read every Parquet column to aggregate one.
- Do not slice across the chunk grain when the same data can be taken along it.

## Tool discovery across agents

Names such as `clio-hdf5:open_file` identify a server and its tool in this
guide. Your agent may expose a different prefix. Match the server and tool
against its live MCP inventory, then use the advertised name and input schema.
If a required server is unavailable, report it before attempting the workflow.

## Completion check

Report full dataset shape/bytes, exact selected region, sampled or processed count, and whether a statistic is exact or approximate. Verify coverage before claiming full-data results; a small response does not guarantee bounded server memory.
