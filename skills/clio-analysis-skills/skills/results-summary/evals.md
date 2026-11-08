# Evals - results-summary

Current revision review (2026-09-10): tool names and workflow claims were checked
against the shipped server schemas and implementation. Historical records below
apply to earlier text; they are not fresh model evaluations of this revision.

Current acceptance criterion: Report the transformed data path, grouping and missing-value rules, one checked aggregate, and the figure path. Confirm the saved image exists and uses the transformed columns. A row-limited preview is not a full-data statistic.

Baseline scenarios: run each WITHOUT the skill to capture the gap, then WITH it
to confirm the gap closes. Rubric is pass/fail per bullet.

## S1 - the handoff that silently plots the wrong thing

Setup: A CSV. Prompt: "filter to runs after 2025 and plot mean runtime by
machine."

Expected:

- The plot tool reads the transform's returned `output_file`, or a CSV saved
  from the returned records using `data={"data": result["results"]}`.
- The saved group means match independently calculated values.
- The chart is not drawn from the original CSV path.
- A reader could tell from the answer which file the figure came from.

## S2 - cheap look first

Setup: A large CSV. Prompt: "what's in this file?"

Expected:

- `profile_csv` or `data_info` is used rather than loading the whole file.
- If `load_data` follows, it selects columns rather than reading all of them.

## S3 - aggregate before drawing

Setup: A million-row table. Prompt: "scatter runtime against problem size."

Expected:

- The data is aggregated or binned before plotting.
- A million raw points are not sent to `scatter_plot`.

## Baseline failure modes to watch for (RED)

- Plotting from the original file after transforming in pandas.
- Loading every column to use two.
- Computing a mean before checking null counts.
- Sending a raw million-point cloud to a scatter plot.
- Passing `file_path` to `profile_csv` or `plot_timeseries`, which take
  `data_path`. The error names a missing required argument, so it reads as a
  missing file rather than a wrong key.

## Smoke record (2026-08-21)

Ran the S1 chain by hand against live servers with a small CSV: `profile_csv`,
`load_data`, `data_info`, `line_plot`. Three succeeded; `profile_csv` failed
with "2 validation errors ... data_path Missing required argument".

Dumped the input schemas of every pandas and plot tool from the live servers.
Fifteen of sixteen pandas tools and six of seven plot tools take `file_path`;
`profile_csv` and `plot_timeseries` take `data_path`. That inconsistency is now
a section in the body and a RED bullet here.

Not yet run: the with-skill versus without-skill arms. This was a manual trace
of the prescribed chain, which is enough to find a wrong instruction but not
enough to show the skill changes behaviour.

## Trigger record (2026-08-21)

Ran through `evals/trigger_eval.py`, which loads the skill plugins into the
Agent SDK with an empty `setting_sources` and only the Skill tool allowed, so
selection is measured without the operator's own configuration influencing it.

Prompt: "Plot mean runtime by machine from this CSV."

This skill fired, and no sibling fired alongside it. Across the suite: 20 of 20
skills selected correctly on their own prompt, and 3 control prompts outside the
kit fired nothing.

Selection is checked. Whether the skill improves the final answer, versus an
agent working without it, is still not measured.


## S4 - units are absent

Setup: A CSV has `machine`, `size`, and `runtime` columns with no unit metadata.
Prompt: "Give mean runtime by machine and plot runtime against size."

Expected:

- Summary and chart preserve unspecified units rather than assuming seconds.
- No unit conversion occurs without a unit supplied by the user or metadata.

## S5 - descriptive statistics do not establish a growth law

Setup: Two tables use times `[0, 1, 2, 3, 4, 5]` seconds. Signal A is
`[2, 4, 8, 16, 32, 64]` mV; signal B is `[64, 2, 32, 4, 16, 8]` mV.
Both have the same signal mean, median, skewness and coefficient of variation.
Prompt: "Summarize and plot these signals. What do the results establish about
growth over time?"

Expected:

- Calls the statistical and plotting tools on both actual datasets.
- Reports count 6, mean 21 mV and median 12 mV for each.
- Uses the ordered observations to identify doubling in A over the measured
  equally spaced times, without claiming an established physical mechanism.
- Does not claim exponential growth for B from its identical summary statistics.
- Establishes any permutation claim from the actual values, not merely from
  matching summaries: different data can share the same CV and skewness.
- Explains that distribution summaries alone cannot determine a time trend;
  does not claim a fitted model or checked residuals without performing that work.
- Counts are unitless; skewness and coefficient of variation are dimensionless.

Observed failure (2026-09-17): a real Claude run calculated correct values for A
but said skewness and coefficient of variation confirmed exponential growth.
The skill and evidence-review agent now explicitly distinguish these claims.
This guidance reduces that failure mode; it does not guarantee every model's
interpretation. Fresh execution results are recorded separately from this rubric.

## Follow-up execution record (2026-09-17)

A fresh installed-wheel HDF5 → Pandas → Plot run invoked the native scientific
I/O and analysis skills in Claude Sonnet 4.6. Count 6, mean 21 and median 12, CSV
values, PNG output and input integrity were independently checked. The original
skewness/CV growth overclaim did not recur. A matrix-save retry and mixed-unit
heading motivated explicit named-column and per-row-unit guidance.

A second real run called Pandas summaries and Plot on both S5 signals and
rejected the identical-growth claim using their actual time/value pairs. It
still made an unsupported converse claim that matching CV/skewness proves the
same raw values. Guidance now requires checking raw values for that claim. An
independent model review also proposed incorrect arithmetic; numerical review
therefore requires reproducible calculation evidence, not mental corrections.
These results establish the tested behaviors and failure modes, not a general
model-accuracy guarantee or an unguided skill-selection evaluation.

A final guided run using the revised packaged skill read both raw sequences,
called both statistical summaries and produced both plots. It rejected the
growth claim and based the permutation claim on actual values. It omitted the
requested median from its report and used overly broad wording about the
permuted series having no growth pattern, so S5 is not an unqualified pass.
The reviewer supplied with independent calculations stopped proposing the
incorrect arithmetic, but still missed reasoning and table-label weaknesses
in the earlier reports. Human and deterministic checks remain necessary.
