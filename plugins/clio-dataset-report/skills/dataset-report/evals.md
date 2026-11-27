# Evaluation scenarios

- Known HDF5 table: time 0..5, signal 2,4,8,16,32,64; supplied s/mV units.
  Expect actual HDF5→Pandas→Plot calls, exact CSV rows, count 6, mean 21,
  median 12, min 2, max 64, valid figure, unchanged source and helper PASS.
- Alter the manifest mean to 22: helper and installed hook must report FAIL.
  Restore 21: subsequent check must PASS without changing the source baseline.
- Modify the source after preparation: FAIL. Missing PNG, non-finite CSV values,
  missing statistics or over-limit input must not yield PASS.
- Unrelated Write: hook produces no report-check message or file changes.
- Reviewer must distinguish observed doubling from a fitted growth model;
  reviewer correctness is assessed independently of tool success.
- If a time interval is reported for the known table, show successive timestamp
  differences (1 second). The final answer must not introduce a different rate
  or infer a temporal trend from CV/skewness.

Run deterministic fixtures and the native client's hook runtime. A scripted
model proves client integration, not live scientific reasoning. Record live
model results separately, including omissions or unsupported interpretations.

## Local execution — 2026-09-17

The installed plugin's dependencies resolved to HDF5, Pandas, Plot and
clio-agents. Actual MCP calls produced the expected CSV, statistics and PNG.
The native Claude hook reported a wrong mean and then PASS after correction;
this used a scripted model endpoint, not live scientific reasoning. Uninstalling
the task preserved an explicitly installed shared Plot plugin.

A fresh live Claude run was attempted but rejected by the account's five-hour
quota before skill execution. Live end-to-end model/reviewer evaluation of this
new composite skill therefore remains pending. Earlier component evaluations
do not substitute for this test.
