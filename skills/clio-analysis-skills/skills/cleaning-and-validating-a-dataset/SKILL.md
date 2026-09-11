---
name: cleaning-and-validating-a-dataset
description: Use when profiling missing values, duplicates, outliers and validation rules before analysis. Triggers on "clean this data", "missing values", "validate this CSV". Not for plotting; use summarizing-and-plotting-results.
metadata:
  bundle: clio-analysis
  servers: clio-pandas
  provenance: designed
  eval-status: scenarios-recorded
---

# Clean and Validate Tabular Data

Every repair here changes the numbers that come out. Look before repairing, and
say what was done.

## Profile first, always

`clio-pandas:profile_data` — shape, types, missing values, distributions, quality
checks. On a raw CSV, `clio-pandas:profile_csv` answers the same first questions
using a bounded sample; its counts/statistics need not cover the whole file.

Read three things before touching anything: **how much is missing and where**,
**whether dtypes match what the columns mean**, and **whether ranges are
physically possible**.

## Missing data: how it is missing decides the fix

`clio-pandas:handle_missing_data` takes two separate arguments, and conflating
them is the usual first failure. `strategy` is one of `detect`, `impute`,
`remove` or `analyze`, and defaults to `detect`, which only reports. `method` is
the imputation itself: `mean`, `median`, `mode`, `forward_fill`,
`backward_fill`, `interpolate`. Passing `strategy="median"` fails with "Unknown
strategy"; the call you want is `strategy="impute", method="median"`.

The methods are not interchangeable. `interpolate` fills interior numeric gaps
linearly by row position: `[1, missing, 3, missing, 5]` becomes
`[1, 2, 3, 4, 5]`. Leading/trailing gaps, entirely missing columns and
non-numeric gaps remain missing; inspect the reported `imputed_count` and the
output. Select the intended `columns`. This is not time-weighted interpolation:
for irregular timestamps, use a verified time-aware calculation instead.
`forward_fill` propagates the previous observed value; `backward_fill` uses
the next observed value. Both follow row order, so sort and separate independent
series first. `mode` fills numeric or categorical gaps from observed values.
Mean/median operate only on numeric columns. Entirely missing columns stay
missing, and `imputed_count` counts actual fills. Preserve the input and assert
expected values before accepting any transformation.

- **Scattered gaps** — inspect the measurement process; apparent randomness
  does not establish a missingness mechanism. Mean/median imputation changes
  variance and relationships. Choose a method for the scientific purpose and
  report sensitivity to that choice.
- **In runs** — a sensor dropout. Choose the scientific method explicitly; a mean invents a
  plateau at a value that was never measured.
- **Concentrated in one group** — imputing hides a systematic problem. Something
  about that group failed to record, and the fill will look like a real finding.
- **Most of a column** — flag inadequate coverage. Excluding the column,
  collecting more data or using a justified model requires an explicit decision;
  do not silently drop a scientifically essential variable.

Dropping rows is honest and biased: it silently removes exactly the cases with
missing data, which are rarely a random sample.

## Duplicates and outliers

`clio-pandas:clean_data` removes duplicates, detects outliers via IQR or Z-score,
and optimises dtypes in one pass.

Outlier *detection* is not outlier *removal*. An extreme value can be a sensor
fault or the event the whole run was about. Look at flagged rows before deleting
them. Z-score thresholds can be misleading on skewed
distributions. IQR is a robust screening rule, not proof that flagged values
are invalid; inspect the domain and distribution before changing any values.

Duplicate rows are sometimes real — two identical measurements at different times
where the timestamp was not kept. Check what makes a row unique before deduping.

## State the rules and check them

`clio-pandas:validate_data` checks columns against explicit rules: min/max range,
type, nullability, uniqueness, regex. This is stronger than eyeballing a profile,
because it is a claim that can fail later on new data.

Write the rules from what the data means: a fraction is in [0, 1], a temperature
in Kelvin is not negative, an ID is unique and non-null. A validation rule that
encodes physics catches the corrupt file that a statistical check accepts.

## Types and memory

`clio-pandas:optimize_memory` reduces footprint through dtype optimisation and
suggests chunking. Worth running before an expensive operation on a wide table.

Check what it did. A column downcast to a narrower integer type will silently
overflow if later values exceed the new range, and an ID stored as a float has
already lost precision.

## Filtering

`clio-pandas:filter_data` supports comparison, membership, pattern matching and
null checks across several columns. Filter before aggregating, not after: an
average over rows you meant to exclude is wrong in a way that looks fine.

## Then say what you did

Every step above changes the result. Report which columns were imputed and how,
how many rows were removed and why, and what was excluded. A cleaned dataset
with no record of the cleaning is not reproducible.

## What not to do

- Do not impute before knowing whether the gaps are random, in runs, or grouped.
- Do not mean-fill a skewed column.
- Do not delete outliers without looking at them.
- Do not deduplicate before knowing what identifies a row.
- Do not report results from cleaned data without saying what was cleaned.

## Tool discovery across agents

Names such as `clio-hdf5:open_file` identify a server and its tool in this
guide. Your agent may expose a different prefix. Match the server and tool
against its live MCP inventory, then use the advertised name and input schema.
If a required server is unavailable, report it before attempting the workflow.

## Completion check

Keep the source file. Report the output path, before/after row counts, missing counts, transformations, and validation failures. Use each returned output file for the next step; these tools do not share an in-memory dataframe.
