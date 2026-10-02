---
name: chart-select
description: Use when choosing a chart, axes, bins or color scale for scientific data. Triggers on "which chart", "log scale", "color map". Calls no tools. Not for rendering a chart; use results-summary.
metadata:
  bundle: clio-analysis
  servers: none
  provenance: designed
  eval-status: scenarios-recorded
---

# Chart Selection

The chart type follows from the question. Working the other way — picking a
familiar chart and fitting the data to it — is how a figure ends up answering
something nobody asked.

## Question to chart

| The question | Chart |
|---|---|
| How does y change with x, where x is ordered | line |
| How does this change over time | time series line |
| How do a few categories compare | bar |
| Are these two variables related | scatter |
| How is one variable distributed | histogram |
| Which of many variables move together | heatmap of correlations |

Two rules behind the table:

**Lines imply continuity.** A line between two points says the values in between
are meaningful. Over categories they are not, and the line is a claim about
nothing. Use bars.

**Bars imply a magnitude from zero.** A bar chart with a truncated y axis
exaggerates differences — the length of the bar is the message, and it is wrong.
Truncating a *line* axis is fine, because a line's message is the shape.

## Distributions

A histogram's bin width is a choice that changes the conclusion. Too few bins
hides a second peak; too many turn a distribution into noise. Try more than one
before believing a shape — a bimodal distribution reported as unimodal is a
finding lost to a default.

A histogram shows one variable. Comparing distributions across groups is a
different figure, not several histograms the reader has to hold at once.

## Log scales

Use a log axis when the data spans orders of magnitude, or when the interesting
relationship is multiplicative. Runtime against problem size, particle counts,
error against resolution — all naturally log.

An ordinary log axis cannot show zero or negative values. State how those
observations are handled; never silently discard them.

Distinguish the axes: on a **semilog-y** plot, `y = A exp(kx)` is a straight
line; on a **log-log** plot, `y = A x^p` is a straight line. These are different
relationships. Label both axes and do not infer a fitted model from appearance
alone. Use the user's stated data for a conceptual question; an unrelated file
in the working directory does not contradict their description.

## Scatter plots that hide their data

Ten thousand overlapping points is a solid blob. The relationship is in there and
invisible. Aggregate first — bin, or plot a summary — rather than drawing every
point. See `results-summary`.

Correlation describes association, not causation. Report the coefficient with
sample size, missing-data treatment and uncertainty when available. A p-value
does not measure effect size, and dependent observations require an appropriate
analysis rather than an automatic significance claim.

## Colour

A colour map for continuous values should be **perceptually uniform** — equal
steps in value look like equal steps in colour. Viridis and Plasma are. The
classic rainbow is not: it has bright bands at yellow and cyan that read as
features in the data and are entirely an artefact of the palette.

Diverging data — anomalies, differences, anything with a meaningful zero — wants
a diverging map centred on that zero. A sequential map on differences buries the
sign, which was the point.

Roughly one person in twelve with male colour vision cannot distinguish red from
green. Do not encode a distinction in that pair alone.

## Heatmaps

A correlation heatmap is a good summary and a poor diagnostic: it shows the
strength of linear relationships and nothing else. Two variables with a clear
curved relationship can show near-zero correlation. Look at the scatter of any
pair the analysis will rest on.

## Every figure needs

Axis labels with units, a caption saying what the reader should take from it, and
a stated scale where it is not linear. A figure that needs the surrounding text
to be interpretable will be misread the moment it is shown on its own.

## What not to do

- Do not draw a line across categories.
- Do not truncate the y axis of a bar chart.
- Do not accept the default bin count without trying another.
- Do not use a log axis on data containing zeros without saying what happened.
- Do not plot every point when the cloud is opaque.
- Do not use a rainbow colour map for continuous data.
- Do not present a correlation coefficient or p-value as proof of causation.
- Do not ship a figure without units on the axes.

## Completion check

State the question, chart type, axis units, treatment of missing/nonpositive values, and why the encoding supports the comparison. Do not imply a p-value proves causality; report sample size and uncertainty when available.
