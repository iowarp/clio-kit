---
name: summarizing-and-plotting-results
description: Use when calculating tabular summaries and plotting transformed CSV or Excel data. Triggers on "summarize this table", "plot these results", "grouped mean". Not for mesh or volume rendering; use visualizing-3d-simulation-output.
clio-kit:
  bundle: clio-analysis
  servers: clio-pandas, clio-plot
  provenance: designed
  eval-status: scenarios-recorded
---

# Summarize and Plot Scientific Results

Two servers, and the handoff between them is where this goes wrong.

## The plotting tools read files, not dataframes

Every `clio-plot` tool takes a **path to a CSV or Excel file**. There is no way to
hand it the result of a pandas operation in memory.

Use the transform's returned `output_file` when it exists. Otherwise save the
returned records to an absolute CSV path, then plot that file. Never point the
plot at the original input after a transformation.

## Two tools spell the file argument differently

Almost every tool on both servers takes `file_path`. Two do not, and they are
the two you reach for early:

| Tool | Argument |
|---|---|
| `clio-pandas:profile_csv` | `data_path` |
| `clio-plot:plot_timeseries` | `data_path` |

Fifteen pandas tools and six plot tools use `file_path`, so the pattern learned
from the rest is wrong for exactly these. Passing `file_path` to either returns
a validation error naming a missing required argument, which reads like the file
is missing rather than the key being wrong.

## Steps

**1. Look before loading.**

`clio-pandas:profile_csv` gives row and column counts, per-column dtype, null
counts and numeric ranges from a bounded retained sample (default 5,000 rows;
scan cap 250,000). It reads CSV rows and is not a full-data profile. State
these limits; later file-based operations read their inputs independently.

`clio-plot:data_info` answers a similar question and is the cheaper choice when
plotting is all that is wanted.

**2. Load only what you need.**

`clio-pandas:load_data` takes column selection and a row limit. Use both. Reading
40 columns to plot 2 is the same waste as reading a whole array for one number.

**3. Profile properly.**

`clio-pandas:profile_data` — shape, types, missing values, distributions, quality
checks. This is where you find out the column is 30% null before the mean is
computed from it. If it needs fixing, see `cleaning-and-validating-a-dataset`.

**4. Aggregate before plotting.**

`clio-pandas:groupby_operations` for grouped aggregates,
`clio-pandas:statistical_summary` for descriptives,
`clio-pandas:correlation_analysis` for relationships between columns,
`clio-pandas:pivot_table` to reshape into the layout the chart wants.

A million-point scatter is unreadable and slow. Aggregate to the resolution the
figure can actually show.

**5. Hand off the transformed file or records explicitly.**

`groupby_operations` returns an `output_file` and a `results` array. Pass the
returned `output_file` to the plotting tool. If a different destination is
needed, call `save_data(data={"data": result["results"]},
file_path="/absolute/path/means.csv", index=False)`. Do not pass the entire
response envelope: metadata and nested results are not a dataframe.

Verify the saved column names and at least one independently calculated
aggregate before plotting. A successful image render alone proves no numerical
correctness.

**6. Plot from that file.**

| Question | Tool |
|---|---|
| How does y change with x | `clio-plot:line_plot` |
| Several series over time | `clio-plot:plot_timeseries` |
| Compare across categories | `clio-plot:bar_plot` |
| Are these two related | `clio-plot:scatter_plot` |
| How is one variable distributed | `clio-plot:histogram_plot` |
| Which of many columns move together | `clio-plot:heatmap_plot` |

Match the chart to the question, not to preference — see
`choosing-the-right-chart`.

## When the data is not a table

Mesh and volume data does not belong here. `clio-plot` reads CSV and Excel; a
simulation field goes to ParaView instead — see
`visualizing-3d-simulation-output`.

## What not to do

- Do not plot the original input after a transformation; use its output file.
- Do not load every column to use two.
- Do not compute a mean before checking the null count.
- Do not plot a million raw points instead of an aggregate.
- Do not reach for a chart type before deciding what the figure has to show.
- Do not assume `file_path` on `profile_csv` or `plot_timeseries`; both take
  `data_path`.

## Completion check

Report the transformed data path, grouping and missing-value rules, one checked aggregate, and the figure path. Confirm the saved image exists and uses the transformed columns. A row-limited preview is not a full-data statistic.
