---
name: creating-dataset-report
description: 'Use when creating a report from a small numeric HDF5 dataset with a CSV, statistics and figure. Triggers on "dataset report", "inspect summarize and plot". Not for large-file streaming or model fitting.'
metadata:
  bundle: clio-dataset-report
  servers: clio-hdf5, clio-pandas, clio-plot
  provenance: designed
  eval-status: scenarios-recorded
---

# Create a verified dataset report

Deliver `data.csv`, `plot.png`, `dataset-report.md` and
`clio-dataset-report.json` in a new output directory. This bounded workflow
supports a numeric table with a selected value column: source and CSV at most
64 MiB, CSV at most 100,000 rows. Stop and explain when those limits do not fit;
do not silently sample and report whole-data statistics.

Discover the HDF5, Pandas and Plot MCP tools. Inspect HDF5 metadata, dataset
shape and supplied column names/units before reading. Ask for the dataset path
or column meaning when ambiguous; do not infer physical units from values.

Before data operations, run the helper beside this skill (resolve its absolute
path from this installed skill directory):

```bash
python3 /absolute/path/to/this/skill/scripts/verify_report.py prepare --source /absolute/input.h5 --output /absolute/new-report --column signal_mV
```

It records the source hash and planned output paths. Preserve this baseline on
retries. Hashing reads the source file; it is intentionally limited to 64 MiB.

1. Open the HDF5 source read-only. Inspect the chosen dataset and export its
   numeric values with `export_dataset` as JSON. Close the file afterward.
2. Map the exported matrix to the metadata's named columns. Use Pandas
   `save_data` with `data={"time_s": [...], "signal_mV": [...]}` and
   `index=False` to write `data.csv`. A `columns/data` envelope loses names.
   Compare saved rows and column names against the exported values; this is
   the provenance check the helper cannot perform.
3. Call Pandas `statistical_summary` on the selected CSV column. Record unrounded
   `count`, `mean`, `median` (the returned `50%`), `min` and `max` in the
   manifest's `statistics` object, preserving its other fields.
4. Use Plot `line_plot` for an actual ordered axis or `histogram_plot` for a
   distribution; follow the live tool schema. Save `plot.png` and inspect it.
5. Write `dataset-report.md`: source/dataset, columns and units, row count,
   the five statistics, figure, data handoff and limitations. Counts have no
   units; mean/median/min/max retain the measurement units. State observed
   patterns separately from models. CV/skewness alone cannot establish growth;
   no fit, mechanism or extrapolation is established by this workflow.
6. Run the helper's `check /absolute/new-report/clio-dataset-report.json`.
   Fix concrete failures without changing the original baseline. The native
   plugin's PostToolUse hook runs the same read-only check after Write/Edit of
   the manifest or report. Shell/MCP writes do not trigger this hook, so the
   explicit final check remains required. A hook message is feedback, not a
   prevention mechanism or a scientific certification.
7. In Claude Code, use the installed `clio-agents:scientific-evidence-reviewer`
   with the raw data, report and helper output. Outside that host, review the
   same evidence explicitly; mark independent agent review unavailable if so.
   Treat reviewer corrections as claims requiring evidence, not authority.

Report output paths and the final check status. A PASS establishes CSV numeric
checks, source integrity and output presence; it does not establish HDF5→CSV
fidelity, plot semantics, the truth of prose or a fitted scientific model.
